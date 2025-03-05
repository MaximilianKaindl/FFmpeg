/*
 * Copyright (c) 2024
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * DNN Torch backend implementation.
 */

extern "C" {
#include "dnn_io_proc.h"
#include "dnn_backend_common.h"
#include "../dnn_filter_common.h"
#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "queue.h"
#include "safe_queue.h"
#include "libavutil/avassert.h"
#include "libavutil/detection_bbox.h"
#include "libavutil/avstring.h"
}

#include <torch/torch.h>
#include <torch/script.h>

#if (HAVE_LIBTORCH_CUDA == 1)
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#endif

typedef struct THClxpContext {
    torch::Tensor *tokenized_text;
    torch::Tensor *attention_mask;
    int *softmax_units;
    int softmax_units_count;
} THClxpContext;

typedef struct THModel {
    DNNModel model;
    DnnContext *ctx;
    torch::jit::Module *jit_model;
    SafeQueue *request_queue;
    Queue *task_queue;
    Queue *lltask_queue;

    THClxpContext *clxp_ctx;

} THModel;

typedef struct THInferRequest {
    torch::Tensor *output;
    torch::Tensor *input_tensor;
} THInferRequest;

typedef struct THRequestItem {
    THInferRequest *infer_request;
    LastLevelTaskItem **lltasks;
    int lltask_count;
    DNNAsyncExecModule exec_module;
} THRequestItem;


#define OFFSET(x) offsetof(THOptions, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM
static const AVOption dnn_th_options[] = {
    {"optimize","turn on graph executor optimization",
        OFFSET(optimize), AV_OPT_TYPE_INT, {.i64 = 0}, 0,1,FLAGS},
    {NULL}};

static int extract_lltask_from_task(DNNFunctionType func_type, TaskItem *task,
                                    Queue *lltask_queue,
                                    DNNExecBaseParams *exec_params)
{
    THModel *th_model = (THModel *)task->model;
    DnnContext *ctx = th_model->ctx;

    switch (func_type) {
    case DFT_PROCESS_FRAME:
    case DFT_ANALYTICS_CLAP: {
        LastLevelTaskItem *lltask =
            (LastLevelTaskItem *)av_malloc(sizeof(*lltask));
        if (!lltask) {
            av_log(ctx, AV_LOG_ERROR,
                   "Failed to allocate memory for LastLevelTaskItem\n");
            return AVERROR(ENOMEM);
        }
        task->inference_todo = 1;
        task->inference_done = 0;
        lltask->bbox_index = 0;
        lltask->task = task;
        if (ff_queue_push_back(lltask_queue, lltask) < 0) {
            av_log(ctx, AV_LOG_ERROR, "Failed to push back lltask_queue.\n");
            av_freep(&lltask);
            return AVERROR(ENOMEM);
        }
        return 0;
    }
    case DFT_ANALYTICS_CLIP:{
        const AVDetectionBBoxHeader *header;
        AVFrame *frame = task->in_frame;
        AVFrameSideData *sd;
        LastLevelTaskItem *lltask;
        DNNExecZeroShotClassificationParams *params = (DNNExecZeroShotClassificationParams *)exec_params;

        if(params->target == NULL){
            LastLevelTaskItem *lltask = (LastLevelTaskItem *)av_malloc(sizeof(*lltask));
            if (!lltask) {
                av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for LastLevelTaskItem\n");
                return AVERROR(ENOMEM);
            }
            task->inference_todo = 1;
            task->inference_done = 0;
            lltask->bbox_index = 0;
            lltask->task = task;
            if (ff_queue_push_back(lltask_queue, lltask) < 0) {
                av_log(ctx, AV_LOG_ERROR, "Failed to push back lltask_queue.\n");
                av_freep(&lltask);
                return AVERROR(ENOMEM);
            }
            return 0;
        }

        task->inference_todo = 0;
        task->inference_done = 0;

        if (!ff_dnn_contain_valid_detection_bbox(frame)) {
            return 0;
        }

        sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DETECTION_BBOXES);
        header = (const AVDetectionBBoxHeader *)sd->data;

        for (uint32_t i = 0; i < header->nb_bboxes; i++) {
            const AVDetectionBBox *bbox = av_get_detection_bbox(header, i);
            if (bbox->w * bbox->h <= 0) {
                continue;
            }
            if (params->target) {
                if (av_strncasecmp(bbox->detect_label, params->target, sizeof(bbox->detect_label)) != 0) {
                    continue;
                }
            }

            lltask = (LastLevelTaskItem *)av_malloc(sizeof(*lltask));
            if (!lltask) {
                return AVERROR(ENOMEM);
            }
            task->inference_todo++;
            lltask->task = task;
            lltask->bbox_index = i;
            if (ff_queue_push_back(lltask_queue, lltask) < 0) {
                av_freep(&lltask);
                return AVERROR(ENOMEM);
            }
        }
        return 0;
    }
    default:{
        av_assert0(!"should not reach here");
        return AVERROR(EINVAL);
    }
    }
}

static void th_free_request(THInferRequest *request)
{
    if (!request)
        return;
    if (request->output) {
        delete(request->output);
        request->output = NULL;
    }
    if (request->input_tensor) {
        delete(request->input_tensor);
        request->input_tensor = NULL;
    }
    return;
}

static inline void destroy_request_item(THRequestItem **arg)
{
    THRequestItem *item;
    if (!arg || !*arg) {
        return;
    }
    item = *arg;
    th_free_request(item->infer_request);
    av_freep(&item->infer_request);
    av_freep(&item->lltasks);
    ff_dnn_async_module_cleanup(&item->exec_module);
    av_freep(arg);
}

static void free_clip_context(THClxpContext *clxp_ctx)
{
    if (!clxp_ctx)
        return;
    if (clxp_ctx->tokenized_text) {
        delete clxp_ctx->tokenized_text;
        clxp_ctx->tokenized_text = nullptr;
    }
    if (clxp_ctx->attention_mask) {
        delete clxp_ctx->attention_mask;
        clxp_ctx->attention_mask = nullptr;
    }
    av_freep(&clxp_ctx);
}

static void dnn_free_model_th(DNNModel **model)
{
    THModel *th_model;
    if (!model || !*model)
        return;

    th_model = (THModel *)(*model);
    while (ff_safe_queue_size(th_model->request_queue) != 0) {
        THRequestItem *item =
            (THRequestItem *)ff_safe_queue_pop_front(th_model->request_queue);
        destroy_request_item(&item);
    }
    ff_safe_queue_destroy(th_model->request_queue);

    while (ff_queue_size(th_model->lltask_queue) != 0) {
        LastLevelTaskItem *item =
            (LastLevelTaskItem *)ff_queue_pop_front(th_model->lltask_queue);
        av_freep(&item);
    }
    ff_queue_destroy(th_model->lltask_queue);

    while (ff_queue_size(th_model->task_queue) != 0) {
        TaskItem *item = (TaskItem *)ff_queue_pop_front(th_model->task_queue);
        av_frame_free(&item->in_frame);
        av_frame_free(&item->out_frame);
        av_freep(&item);
    }
    ff_queue_destroy(th_model->task_queue);
    delete th_model->jit_model;
#if (CONFIG_LIBTOKENIZERS == 1)
    free_clip_context(th_model->clxp_ctx);
#endif
    av_freep(&th_model);
    *model = NULL;
}

static int get_input_th(DNNModel *model, DNNData *input, const char *input_name)
{
    input->dt = DNN_FLOAT;
    input->order = DCO_RGB;
    input->layout = DL_NCHW;
    input->dims[0] = 1;
    input->dims[1] = 3;
    input->dims[2] = -1;
    input->dims[3] = -1;
    return 0;
}

static void deleter(void *arg)
{
    av_freep(&arg);
}


#if (CONFIG_LIBTOKENIZERS == 1)

static int get_tokenized_batch(THClxpContext *clxp_ctx, const char **labels,
    int label_count, const char *tokenizer_path,
    DnnContext *ctx, const c10::Device &device)
{
    // Early validation to avoid unnecessary processing
    if (!labels || label_count <= 0) {
        av_log(ctx, AV_LOG_ERROR, "Label file invalid.\n");
        return AVERROR(EINVAL);
    }

    if (!tokenizer_path) {
        av_log(ctx, AV_LOG_ERROR, "Tokenizer path not provided.\n");
        return AVERROR(EINVAL);
    }

    TokenizerEncodeResult *results = NULL;
    int ret;

    ret = ff_dnn_create_tokenizer_and_encode_batch(
        tokenizer_path, labels, label_count, &results, ctx);

    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to tokenize batch text\n");
        return ret;
    }

    const int64_t token_dimension = ctx->torch_option.token_dimension;

    // Create the tensors directly with the final batch dimensions
    // Shape: [batch_size, token_dimension]
    auto tokenized_text = torch::zeros({label_count, token_dimension}, 
            torch::TensorOptions().dtype(torch::kInt64));
    auto attention_mask = torch::zeros({label_count, token_dimension}, 
            torch::TensorOptions().dtype(torch::kInt64));

    // Get accessors for direct, efficient memory access
    auto tokens_accessor = tokenized_text.accessor<int64_t, 2>();
    auto attention_accessor = attention_mask.accessor<int64_t, 2>();

    // Fill the tensors directly
    for (int i = 0; i < label_count; i++) {
        const int current_token_count = results[i].len;

        // Fill only the valid token positions, leaving zeros elsewhere
        for (int j = 0; j < current_token_count && j < token_dimension; j++) {
            tokens_accessor[i][j] = static_cast<int64_t>(results[i].token_ids[j]);
            attention_accessor[i][j] = 1;
        }
    }

    clxp_ctx->tokenized_text = new torch::Tensor(tokenized_text);
    clxp_ctx->attention_mask = new torch::Tensor(attention_mask);

    if(clxp_ctx->tokenized_text->device() != device){
        *clxp_ctx->tokenized_text = clxp_ctx->tokenized_text->to(device);
    }
    if(clxp_ctx->attention_mask->device() != device){
        *clxp_ctx->attention_mask = clxp_ctx->attention_mask->to(device);
    }

    ff_dnn_tokenizer_free_results(results, label_count);

    return 0;
}

static int copy_softmax_units(THModel *th_model, const int *softmax_units, int softmax_units_count)
{
    if (softmax_units && softmax_units_count > 0) {
        th_model->clxp_ctx->softmax_units =
            (int *)av_malloc_array(softmax_units_count, sizeof(int));
        if (!th_model->clxp_ctx->softmax_units) {
            av_log(th_model->ctx, AV_LOG_ERROR,
                   "Failed to allocate memory for softmax units\n");
            return AVERROR(ENOMEM);
        }
        memcpy(th_model->clxp_ctx->softmax_units, softmax_units,
               softmax_units_count * sizeof(int));
        th_model->clxp_ctx->softmax_units_count = softmax_units_count;
    } else {
        th_model->clxp_ctx->softmax_units = NULL;
        th_model->clxp_ctx->softmax_units_count = 0;
    }
    return 0;
}

static int test_clip_inference(THModel *th_model, const c10::Device &device){
    // Try given resolution
    if(th_model->ctx->torch_option.input_resolution >= 0){
        try {
            torch::Tensor test_input = torch::zeros(th_model->ctx->torch_option.input_resolution);
            if (test_input.device() != device) {
                test_input = test_input.to(device);
            }
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(test_input);
            inputs.push_back(*th_model->clxp_ctx->tokenized_text);
            auto output = th_model->jit_model->forward(inputs);
        } catch (const std::exception &e) {
            av_log(th_model->ctx, AV_LOG_ERROR,
                    "CLIP Input Resolution %ld did not work\n", th_model->ctx->torch_option.input_resolution);
            return AVERROR(EINVAL);
        }
        return 0;
    }

    // Common CLIP input dimensions to test
    std::vector<int64_t> test_dims[] = {
        {1, 3, 224, 224},
        {1, 3, 256, 256},
        {1, 3, 384, 384},
        {1, 3, 378, 378},
    };
    bool found_dims = false;
    int64_t resolution = 0;

    for (const auto &dims : test_dims) {
        // Create test input tensor
        torch::Tensor test_input = torch::zeros(dims);
        if (test_input.device() != device) {
            test_input = test_input.to(device);
        }
        try {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(test_input);
            inputs.push_back(*th_model->clxp_ctx->tokenized_text);
            auto output = th_model->jit_model->forward(inputs);
        } catch (const std::exception &e) {
            av_log(th_model->ctx, AV_LOG_WARNING,
                    "CLIP Input Resolution %ld did not work\n", dims[2]);
            continue;
        }
        resolution = dims[2];
        found_dims = true;
        break;
    }
    if (!found_dims || resolution <= 0) {
        av_log(th_model->ctx, AV_LOG_ERROR,
            "Failed to determine input resolution for CLIP model\n");
        return AVERROR(EINVAL);
    }
    // Log the resolution chosen for the CLIP model
    av_log(th_model->ctx, AV_LOG_INFO, 
        "Using input resolution %ldx%ld for CLIP model\n", 
        resolution, resolution);
    th_model->ctx->torch_option.input_resolution = resolution;
    return 0;
}

static int test_clap_inference(THModel *th_model, int64_t sample_rate, int64_t sample_duration, const c10::Device &device){
    try {
        // Create dummy audio tensor to test model compatibility
        int target_samples = sample_rate * sample_duration;
        torch::Tensor dummy_audio = torch::zeros({1, target_samples});

        // Try to move the tensor to the correct device
        if (dummy_audio.device() != device) {
            dummy_audio = dummy_audio.to(device);
        }

        // Test inference with dummy audio using forward method<
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(dummy_audio);
        inputs.push_back(*th_model->clxp_ctx->tokenized_text);
        inputs.push_back(*th_model->clxp_ctx->attention_mask);

        auto audio_features = th_model->jit_model->forward(inputs);
    } catch (const c10::Error &e) {
        av_log(th_model->ctx, AV_LOG_ERROR,
                "Error during CLIP model initialization: %s\n", e.what());
        return AVERROR(EINVAL);
    } catch (const std::exception &e) {
        av_log(th_model->ctx, AV_LOG_ERROR,
                "Error during CLAP model inference testing\n");
        return AVERROR(EINVAL);
    }
    return 0;
}

static int init_clxp_model(THModel *th_model, DNNFunctionType func_type,
                           const char **labels, int label_count,
                           const char *tokenizer_path,
                           const AVFilterContext *filter_ctx)
{
    c10::Device device = (*th_model->jit_model->parameters().begin()).device();
    th_model->clxp_ctx = (THClxpContext *)av_mallocz(sizeof(THClxpContext));
    if (!th_model->clxp_ctx) {
        av_log(th_model->ctx, AV_LOG_ERROR,
               "Failed to allocate memory for CLIP context\n");
        return AVERROR(ENOMEM);
    }

    int ret = get_tokenized_batch(th_model->clxp_ctx, labels, label_count,
                                  tokenizer_path, th_model->ctx, device);
    if (ret < 0) {
        av_log(th_model->ctx, AV_LOG_ERROR,
               "Failed to tokenize batch text for CLIP model\n");
        return ret;
    }
    return 0;
}
#endif

static torch::Tensor calculate_similarity(torch::Tensor &tensor1,
                                          torch::Tensor &tensor2,
                                          bool normalize, float logit_scale,
                                          DnnContext *ctx)
{
    try {
        if (normalize) {
            tensor1 = torch::nn::functional::normalize(
                tensor1, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));

            tensor2 = torch::nn::functional::normalize(
                tensor2, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
        }

        // Compute similarity matrix
        torch::Tensor similarity =
            logit_scale * torch::matmul(tensor2, tensor1.transpose(0, 1));
        return similarity.transpose(0, 1);
    } catch (const c10::Error &e) {
        if (ctx) {
            av_log(ctx, AV_LOG_ERROR, "Similarity computation failed: %s\n",
                   e.what());
        }
        return torch::Tensor(); // Return empty tensor properly
    }
}

static torch::Tensor apply_softmax(torch::Tensor input_tensor,
                                   const int *softmax_units,
                                   int softmax_units_count, DnnContext *ctx)
{
    try {
        // Check for empty or invalid input tensor
        if (input_tensor.numel() == 0 || input_tensor.dim() < 2) {
            av_log(ctx, AV_LOG_ERROR, "Invalid input tensor for softmax\n");
            return input_tensor;
        }

        // If no specific units are provided, apply softmax to the entire tensor
        if (!softmax_units || softmax_units_count <= 0) {
            return torch::nn::functional::softmax(
                input_tensor, torch::nn::functional::SoftmaxFuncOptions(1));
        }

        torch::Tensor result = input_tensor.clone();
        int offset = 0;

        // Apply softmax to each specified segment
        for (int i = 0; i < softmax_units_count; i++) {
            int length = softmax_units[i];
            if (length <= 0 || offset + length > input_tensor.size(1)) {
                continue;
            }

            // Select the segment to apply softmax
            torch::Tensor segment = result.slice(1, offset, offset + length);

            // Apply softmax along dimension 1 (across labels in segment)
            torch::Tensor softmax_segment = torch::nn::functional::softmax(
                segment, torch::nn::functional::SoftmaxFuncOptions(1));

            // Put softmaxed segment back into result tensor
            result.slice(1, offset, offset + length) = softmax_segment;

            // Move offset forward
            offset += length;
        }
        return result;
    } catch (const c10::Error &e) {
        if (ctx) {
            av_log(ctx, AV_LOG_ERROR, "Error applying softmax: %s\n", e.what());
        }
        return input_tensor; // Return original tensor on error
    }
}

static torch::Tensor handle_short_audio_tensor(torch::Tensor audio_tensor,
                                               int target_samples)
{
    int nb_samples = audio_tensor.size(0);
    int repeat_factor = (target_samples + nb_samples - 1) / nb_samples;

    // Repeat tensor along dimension 0 to fill required length
    torch::Tensor repeated = audio_tensor.repeat({repeat_factor});

    // Take only the needed samples
    return repeated.slice(0, 0, target_samples);
}

static torch::Tensor handle_long_audio_tensor(torch::Tensor audio_tensor,
                                              int target_samples,
                                              TaskItem *task)
{
    int nb_samples = audio_tensor.size(0);
    int max_start = nb_samples - target_samples;

    // Use a deterministic seed based on frame properties
    unsigned int seed =
        (unsigned int)((uintptr_t)task ^
                       (uintptr_t)(task->in_frame->pts ? task->in_frame->pts
                                                       : nb_samples));

    // Determine start position - center-biased for better representation
    int start_idx;

    // Prefer center segments for better representation, with some randomness
    if (seed % 3 == 0) { // ~33% chance for center segment
        start_idx = (nb_samples - target_samples) / 2;
    } else {
        // Otherwise use seeded position
        start_idx = seed % (max_start + 1);
    }

    // Extract the segment using slice operation
    return audio_tensor.slice(0, start_idx, start_idx + target_samples);
}

static int prepare_audio_tensor(const THModel *th_model,
                                const THRequestItem *request)
{
    THInferRequest *infer_request = request->infer_request;
    LastLevelTaskItem *lltask = request->lltasks[0];
    TaskItem *task = lltask->task;
    DnnContext *ctx = th_model->ctx;
    int ret = 0;

    const int target_samples = th_model->ctx->torch_option.sample_rate * th_model->ctx->torch_option.sample_duration;

    // Validate input frame
    if (!task->in_frame || !task->in_frame->data[0]) {
        av_log(ctx, AV_LOG_ERROR, "Invalid input frame or data\n");
        return AVERROR(EINVAL);
    }

    // Get audio data from the frame
    float *audio_data = (float *)task->in_frame->data[0];
    int nb_samples = task->in_frame->nb_samples;

    // Validate audio parameters
    if (task->in_frame->sample_rate != th_model->ctx->torch_option.sample_rate) {
        av_log(ctx, AV_LOG_ERROR,
               "Sample rate mismatch. Expected %ld Hz, got %d Hz\n",
               th_model->ctx->torch_option.sample_rate, task->in_frame->sample_rate);
        return AVERROR(EINVAL);
    }

    if (task->in_frame->format != AV_SAMPLE_FMT_FLT) {
        av_log(ctx, AV_LOG_ERROR,
               "Unsupported sample format. Expected float\n");
        return AVERROR(EINVAL);
    }

    try {
        torch::Tensor audio_tensor =
            torch::from_blob(audio_data, {nb_samples}, torch::kFloat32).clone();

        c10::Device device =
            (*th_model->jit_model->parameters().begin()).device();
        if (audio_tensor.device() != device) {
            audio_tensor = audio_tensor.to(device);
        }

        // Create target tensor based on the audio length
        torch::Tensor processed_tensor;

        if (nb_samples < target_samples) {
            // Handle short audio using tensor repeat operation
            processed_tensor =
                handle_short_audio_tensor(audio_tensor, target_samples);
        } else if (nb_samples > target_samples) {
            // Handle long audio using tensor slice operation
            processed_tensor =
                handle_long_audio_tensor(audio_tensor, target_samples, task);
        } else {
            // Exact length, just use the tensor as is
            processed_tensor = audio_tensor;
        }

        processed_tensor = processed_tensor.reshape({1, -1});

        // Assign to output
        *infer_request->input_tensor = processed_tensor;
    } catch (const c10::Error &e) {
        av_log(ctx, AV_LOG_ERROR, "Audio tensor processing failed: %s\n",
               e.what());
        return AVERROR(EINVAL);
    }

    return ret;
}

static int preprocess_image_tensor(const THModel *th_model,
                                   torch::Tensor *input_tensor,
                                   const c10::Device &device)
{
    DnnContext *ctx = th_model->ctx;
    try {
        if (input_tensor->device() != device) {
            *input_tensor = input_tensor->to(device);
        }
        *input_tensor = torch::nn::functional::interpolate(
            *input_tensor,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{ctx->torch_option.input_resolution, ctx->torch_option.input_resolution})
                .mode(torch::kBicubic)
                .align_corners(false));
        return 0;
    } catch (const c10::Error &e) {
        av_log(ctx, AV_LOG_ERROR, "Image encoding error: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

static int fill_model_input_th(THModel *th_model, THRequestItem *request)
{
    LastLevelTaskItem *lltask = NULL;
    TaskItem *task = NULL;
    THInferRequest *infer_request = request->infer_request;
    DNNData input = {0};
    DnnContext *ctx = th_model->ctx;
    int ret, width_idx, height_idx, channel_idx;
    std::vector<torch::Tensor> batch_tensors;

    ret = get_input_th(&th_model->model, &input, NULL);
    if (ret != 0) {
        goto err;
    }
    width_idx = dnn_get_width_idx_by_layout(input.layout);
    height_idx = dnn_get_height_idx_by_layout(input.layout);
    channel_idx = dnn_get_channel_idx_by_layout(input.layout);
    infer_request->input_tensor = new torch::Tensor();
    infer_request->output = new torch::Tensor();

    if (th_model->model.func_type == DFT_ANALYTICS_CLAP) {
        lltask =
            (LastLevelTaskItem *)ff_queue_pop_front(th_model->lltask_queue);
        if (!lltask) {
            return -1;
        }
        request->lltasks[request->lltask_count++] = lltask;
        task = lltask->task;
        return prepare_audio_tensor(th_model, request);
    }

    while (ff_queue_size(th_model->lltask_queue) != 0) {
        lltask =
            (LastLevelTaskItem *)ff_queue_pop_front(th_model->lltask_queue);
        if (!lltask) {
            break;
        }
        request->lltasks[request->lltask_count++] = lltask;
        task = lltask->task;

        input.dims[height_idx] = task->in_frame->height;
        input.dims[width_idx] = task->in_frame->width;
        input.data = av_malloc(input.dims[height_idx] * input.dims[width_idx] *
                               input.dims[channel_idx] * sizeof(float));
        if (!input.data) {
            ret = AVERROR(ENOMEM);
            goto err;
        }
        switch (th_model->model.func_type) {
        case DFT_PROCESS_FRAME:
        case DFT_ANALYTICS_CLIP:
            input.scale = 255;
            if (task->do_ioproc) {
                if (th_model->model.frame_pre_proc != NULL) {
                    th_model->model.frame_pre_proc(task->in_frame, &input,
                                                   th_model->model.filter_ctx);
                } else {
                    ff_proc_from_frame_to_dnn(task->in_frame, &input, ctx);
                }
            }
            break;
        default:
            avpriv_report_missing_feature(NULL, "model function type %d",
                                          th_model->model.func_type);
            ret = AVERROR(EINVAL);
            goto err;
        }

        try {
            auto tensor = torch::from_blob(input.data,
                                           {1, input.dims[channel_idx],
                                            input.dims[height_idx],
                                            input.dims[width_idx]},
                                           deleter, torch::kFloat32)
                              .clone();
            if (th_model->model.func_type == DFT_ANALYTICS_CLIP) {
                preprocess_image_tensor(th_model, &tensor, torch::kCPU);
            }
            batch_tensors.push_back(tensor);
            input.data = NULL; // Ownership transferred to tensor
        } catch (const c10::Error &e) {
            av_log(ctx, AV_LOG_ERROR, "Error creating tensor: %s\n", e.what());
            ret = AVERROR(EINVAL);
            goto err;
        }

        av_freep(&input.data);
    }

    // Stack tensors into batch
    try {
        if (!batch_tensors.empty()) {
            *infer_request->input_tensor = torch::cat(batch_tensors, 0);
        } else {
            av_log(ctx, AV_LOG_ERROR, "No tensors to process\n");
            ret = AVERROR(EINVAL);
            goto err;
        }
    } catch (const c10::Error &e) {
        av_log(ctx, AV_LOG_ERROR, "Error creating batch tensor: %s\n",
               e.what());
        ret = AVERROR(EINVAL);
        goto err;
    }
    return 0;

err:
    if (input.data) {
        av_freep(&input.data);
    }
    th_free_request(infer_request);
    return ret;
}

static int th_start_inference(void *args)
{
    THRequestItem *request = (THRequestItem *)args;
    THInferRequest *infer_request = NULL;
    LastLevelTaskItem *lltask = NULL;
    TaskItem *task = NULL;
    THModel *th_model = NULL;
    DnnContext *ctx = NULL;
    std::vector<torch::jit::IValue> inputs;
    torch::NoGradGuard no_grad;

    if (!request) {
        av_log(NULL, AV_LOG_ERROR, "THRequestItem is NULL\n");
        return AVERROR(EINVAL);
    }
    infer_request = request->infer_request;
    lltask = request->lltasks[0];
    task = lltask->task;
    th_model = (THModel *)task->model;
    ctx = th_model->ctx;

    if (ctx->torch_option.optimize)
        torch::jit::setGraphExecutorOptimize(true);
    else
        torch::jit::setGraphExecutorOptimize(false);

    if (!infer_request->input_tensor || !infer_request->output) {
        av_log(ctx, AV_LOG_ERROR, "input or output tensor is NULL\n");
        return DNN_GENERIC_ERROR;
    }
    // Transfer tensor to the same device as model
    c10::Device device = (*th_model->jit_model->parameters().begin()).device();

    if (infer_request->input_tensor->device() != device)
        *infer_request->input_tensor = infer_request->input_tensor->to(device);
    inputs.push_back(*infer_request->input_tensor);

#if (CONFIG_LIBTOKENIZERS == 1)
    if (th_model->model.func_type == DFT_ANALYTICS_CLIP) {
        inputs.push_back(*th_model->clxp_ctx->tokenized_text);
    } else if (th_model->model.func_type == DFT_ANALYTICS_CLAP) {
        inputs.push_back(*th_model->clxp_ctx->tokenized_text);
        inputs.push_back(*th_model->clxp_ctx->attention_mask);
    }
#endif

    auto result = th_model->jit_model->forward(inputs);

    if (th_model->model.func_type == DFT_PROCESS_FRAME) {
        *infer_request->output = result.toTensor();
    } else if (th_model->model.func_type == DFT_ANALYTICS_CLIP ||
               th_model->model.func_type == DFT_ANALYTICS_CLAP) {
        if (result.isTuple()) {
            auto result_tuple = result.toTuple();
            torch::Tensor media_embeddings;
            torch::Tensor text_embeddings;
            float logit_scale = th_model->ctx->torch_option.logit_scale;
            if(th_model->ctx->torch_option.forward_order == 1){
                media_embeddings = result_tuple->elements()[1].toTensor();
                text_embeddings = result_tuple->elements()[0].toTensor();
                *infer_request->output = calculate_similarity(media_embeddings, text_embeddings, th_model->ctx->torch_option.normalize, logit_scale, ctx);
            } else {
                media_embeddings = result_tuple->elements()[0].toTensor();
                text_embeddings = result_tuple->elements()[1].toTensor();
                *infer_request->output = calculate_similarity(media_embeddings, text_embeddings, th_model->ctx->torch_option.normalize, logit_scale, ctx);
            }
            *infer_request->output = apply_softmax(
                *infer_request->output, th_model->clxp_ctx->softmax_units,
                th_model->clxp_ctx->softmax_units_count, ctx);
        }
    } else {
        avpriv_report_missing_feature(ctx, "model function type %d",
                                      th_model->model.func_type);
        return AVERROR(EINVAL);
    }

    return 0;
}

static void infer_completion_callback(void *args)
{
    THRequestItem *request = (THRequestItem *)args;
    LastLevelTaskItem *lltask = request->lltasks[0];
    TaskItem *task = lltask->task;
    DNNData outputs = {0};
    THInferRequest *infer_request = request->infer_request;
    THModel *th_model = (THModel *)task->model;
    torch::Tensor *output = infer_request->output;

    c10::IntArrayRef sizes = output->sizes();
    outputs.order = DCO_RGB;
    outputs.layout = DL_NCHW;
    outputs.dt = DNN_FLOAT;
    if (th_model->model.func_type == DFT_ANALYTICS_CLIP ||
        th_model->model.func_type == DFT_ANALYTICS_CLAP) {
        // CLIP outputs are similarity scores [batch_size, num_labels]
        if (sizes.size() != 2) {
            av_log(th_model->ctx, AV_LOG_ERROR,
                   "Invalid CLIP output dimensions\n");
            goto err;
        }
        outputs.dims[0] = sizes[0]; // batch_size
        outputs.dims[1] = sizes[1]; // number of labels
        outputs.order = th_model->model.func_type == DFT_ANALYTICS_CLIP
                            ? DCO_RGB
                            : DCO_NONE; // doesn't matter for similarity scores
        outputs.dt = DNN_FLOAT;
    } else if (sizes.size() == 4 &&
               th_model->model.func_type == DFT_PROCESS_FRAME) {
        // 4 dimensions: [batch_size, channel, height, width]
        // this format of data is normally used for video frame SR
        outputs.dims[0] = sizes.at(0); // N
        outputs.dims[1] = sizes.at(1); // C
        outputs.dims[2] = sizes.at(2); // H
        outputs.dims[3] = sizes.at(3); // W
    } else {
        avpriv_report_missing_feature(th_model->ctx,
                                      "Support of this kind of model");
        goto err;
    }

    // Process each item in the batch
    for (int i = 0; i < request->lltask_count; i++) {
        LastLevelTaskItem *lltask = request->lltasks[i];
        TaskItem *task = lltask->task;

        // Extract single item from batch
        torch::Tensor single_output;
        try {
            single_output = output->select(0, i);

            // Move to CPU if needed
            if (single_output.device() != torch::kCPU) {
                single_output = single_output.to(torch::kCPU);
            }

            outputs.data = single_output.data_ptr();
        } catch (const c10::Error &e) {
            av_log(th_model->ctx, AV_LOG_ERROR,
                   "Error processing output tensor: %s\n", e.what());
            goto err;
        }

        switch (th_model->model.func_type) {
        case DFT_PROCESS_FRAME:
            if (task->do_ioproc) {
                outputs.scale = 255;
                if (th_model->model.frame_post_proc != NULL) {
                    th_model->model.frame_post_proc(task->out_frame, &outputs,
                                                    th_model->model.filter_ctx);
                } else {
                    ff_proc_from_dnn_to_frame(task->out_frame, &outputs,
                                              th_model->ctx);
                }
            } else {
                task->out_frame->width =
                    outputs.dims[dnn_get_width_idx_by_layout(outputs.layout)];
                task->out_frame->height =
                    outputs.dims[dnn_get_height_idx_by_layout(outputs.layout)];
            }
            break;
#if (CONFIG_LIBTOKENIZERS == 1)
        case DFT_ANALYTICS_CLIP:
        case DFT_ANALYTICS_CLAP:
            if (task->do_ioproc) {
                if (!th_model->model.classify_post_proc) {
                    av_log(th_model->ctx, AV_LOG_ERROR,
                           "CLIP filter needs to provide post proc\n");
                    goto err;
                }
                th_model->model.classify_post_proc(task->in_frame, &outputs,
                                                   lltask->bbox_index,
                                                   th_model->model.filter_ctx);
            }
            break;
#endif
        default:
            avpriv_report_missing_feature(th_model->ctx,
                                          "model function type %d",
                                          th_model->model.func_type);
            goto err;
        }
        task->inference_done++;
        av_freep(&request->lltasks[i]);
    }
err:
    av_freep(&request->lltasks);
    request->lltask_count = 0;

    if (ff_safe_queue_push_back(th_model->request_queue, request) < 0) {
        destroy_request_item(&request);
        av_log(th_model->ctx, AV_LOG_ERROR,
               "Unable to push back request_queue\n");
    }
}

static int execute_model_th(THRequestItem *request, Queue *lltask_queue)
{
    THModel *th_model = NULL;
    LastLevelTaskItem *lltask;
    TaskItem *task = NULL;
    int ret = 0;

    if (ff_queue_size(lltask_queue) == 0) {
        destroy_request_item(&request);
        return 0;
    }

    lltask = (LastLevelTaskItem *)ff_queue_peek_front(lltask_queue);
    if (lltask == NULL) {
        av_log(NULL, AV_LOG_ERROR, "Failed to get LastLevelTaskItem\n");
        ret = AVERROR(EINVAL);
        goto err;
    }
    task = lltask->task;
    th_model = (THModel *)task->model;

    ret = fill_model_input_th(th_model, request);
    if (ret != 0) {
        goto err;
    }
    if (task->async) {
        avpriv_report_missing_feature(th_model->ctx, "LibTorch async");
    } else {
        ret = th_start_inference((void *)(request));
        if (ret != 0) {
            goto err;
        }
        infer_completion_callback(request);
        return (task->inference_done == task->inference_todo)
                   ? 0
                   : DNN_GENERIC_ERROR;
    }

err:
    th_free_request(request->infer_request);
    if (ff_safe_queue_push_back(th_model->request_queue, request) < 0) {
        destroy_request_item(&request);
    }
    return ret;
}

static int get_output_th(DNNModel *model, const char *input_name,
                         int input_width, int input_height,
                         const char *output_name, int *output_width,
                         int *output_height)
{
    int ret = 0;
    THModel *th_model = (THModel *)model;
    DnnContext *ctx = th_model->ctx;
    TaskItem task = {0};
    THRequestItem *request = NULL;
    DNNExecBaseParams exec_params = {
        .input_name = input_name,
        .output_names = &output_name,
        .nb_output = 1,
        .in_frame = NULL,
        .out_frame = NULL,
    };
    ret = ff_dnn_fill_gettingoutput_task(&task, &exec_params, th_model,
                                         input_height, input_width, ctx);
    if (ret != 0) {
        goto err;
    }

    ret = extract_lltask_from_task(th_model->model.func_type, &task,
                                   th_model->lltask_queue, NULL);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR,
               "unable to extract last level task from task.\n");
        goto err;
    }

    request = (THRequestItem *)ff_safe_queue_pop_front(th_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        ret = AVERROR(EINVAL);
        goto err;
    }

    ret = execute_model_th(request, th_model->lltask_queue);
    *output_width = task.out_frame->width;
    *output_height = task.out_frame->height;

err:
    av_frame_free(&task.out_frame);
    av_frame_free(&task.in_frame);
    return ret;
}

static THInferRequest *th_create_inference_request(void)
{
    THInferRequest *request =
        (THInferRequest *)av_malloc(sizeof(THInferRequest));
    if (!request) {
        return NULL;
    }
    request->input_tensor = NULL;
    request->output = NULL;
    return request;
}

static THModel *init_model_th(DnnContext *ctx, DNNFunctionType func_type,
                              AVFilterContext *filter_ctx)
{
    DNNModel *model = NULL;
    THModel *th_model = NULL;
    THRequestItem *item = NULL;
    const char *device_name = ctx->device ? ctx->device : "cpu";

    th_model = (THModel *)av_mallocz(sizeof(THModel));
    if (!th_model)
        return NULL;
    model = &th_model->model;
    th_model->ctx = ctx;

    if(ctx->torch_option.forward_order < 0){
        //set default value for forward_order
        ctx->torch_option.forward_order = func_type == DFT_ANALYTICS_CLAP ? 1 : 0;
        // Log the default value for forward_order
        av_log(ctx, AV_LOG_INFO,
            "Using default forward_order=%d for %s input\n",
            ctx->torch_option.forward_order,
            func_type == DFT_ANALYTICS_CLAP ? "audio" : "video");
    }
    if(ctx->torch_option.logit_scale <= 0){
        //set default value for logit_scale
        ctx->torch_option.logit_scale = func_type == DFT_ANALYTICS_CLAP ? 33.37 : 4.6052;
        // Log the default value for logit_scale
        av_log(ctx, AV_LOG_INFO,
            "Using default logit_scale=%.4f for %s input\n",
            ctx->torch_option.logit_scale,
            func_type == DFT_ANALYTICS_CLAP ? "audio" : "video");
    }
    if(ctx->torch_option.normalize < 0){
        ctx->torch_option.normalize = func_type == DFT_ANALYTICS_CLAP ? 1 : 0;
        // Log the default value for logit_scale
        av_log(ctx, AV_LOG_INFO,
            "Using default normalize=%d for %s input\n",
            ctx->torch_option.normalize,
            func_type == DFT_ANALYTICS_CLAP ? "audio" : "video");
    }

    c10::Device device = c10::Device(device_name);
    if (device.is_xpu()) {
        if (!at::hasXPU()) {
            av_log(ctx, AV_LOG_ERROR, "No XPU device found\n");
            goto fail;
        }
        at::detail::getXPUHooks().init();
#if (HAVE_LIBTORCH_CUDA == 1)
    } else if (device.is_cuda()) {
        if (!torch::cuda::is_available()) {
            av_log(ctx, AV_LOG_ERROR, "CUDA is not available!\n");
            goto fail;
        }
        // Initialize CUDA
        try {
            int device_idx = 0;
            const char *device_num = strstr(device_name, ":");
            if (device_num) {
                char *endptr = NULL;
                device_idx = strtol(device_num + 1, &endptr, 10);
                if (*endptr != '\0' && !isspace(*endptr)) {
                    av_log(ctx, AV_LOG_ERROR,
                           "Invalid device number format: %s\n",
                           device_num + 1);
                    goto fail;
                }
            }
            if (device_idx >= static_cast<int>(torch::cuda::device_count())) {
                av_log(
                    ctx, AV_LOG_ERROR,
                    "Requested CUDA device %d but only %ld devices available\n",
                    device_idx, torch::cuda::device_count());
                goto fail;
            }
            c10::cuda::set_device(device_idx);
            c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
            torch::cuda::synchronize();
        } catch (const c10::Error &e) {
            av_log(ctx, AV_LOG_ERROR, "CUDA initialization failed: %s\n",
                   e.what());
            goto fail;
        }
#endif
    } else if (!device.is_cpu()) {
        av_log(ctx, AV_LOG_ERROR, "Not supported device:\"%s\"\n", device_name);
        goto fail;
    }

    try {
        th_model->jit_model = new torch::jit::Module;
        (*th_model->jit_model) = torch::jit::load(ctx->model_filename);
        th_model->jit_model->to(device);
    } catch (const c10::Error &e) {
        av_log(ctx, AV_LOG_ERROR, "Failed to load torch model\n");
        goto fail;
    }

    th_model->request_queue = ff_safe_queue_create();
    if (!th_model->request_queue) {
        goto fail;
    }

    item = (THRequestItem *)av_mallocz(sizeof(THRequestItem));
    if (!item) {
        goto fail;
    }
    item->lltasks = NULL;
    item->infer_request = th_create_inference_request();
    if (!item->infer_request) {
        av_log(NULL, AV_LOG_ERROR,
               "Failed to allocate memory for Torch inference request\n");
        goto fail;
    }
    item->exec_module.start_inference = &th_start_inference;
    item->exec_module.callback = &infer_completion_callback;
    item->exec_module.args = item;

    if (ff_safe_queue_push_back(th_model->request_queue, item) < 0) {
        goto fail;
    }
    item = NULL;

    th_model->task_queue = ff_queue_create();
    if (!th_model->task_queue) {
        goto fail;
    }

    th_model->lltask_queue = ff_queue_create();
    if (!th_model->lltask_queue) {
        goto fail;
    }

    model->get_input = &get_input_th;
    model->get_output = &get_output_th;
    model->filter_ctx = filter_ctx;
    model->func_type = func_type;
    return th_model;

fail:
    if (item) {
        destroy_request_item(&item);
        av_freep(&item);
    }
    dnn_free_model_th(&model);
    return NULL;
}

static DNNModel *dnn_load_model_th(DnnContext *ctx, DNNFunctionType func_type,
                                   AVFilterContext *filter_ctx)
{
    THModel *th_model = init_model_th(ctx, func_type, filter_ctx);
    if (!th_model) {
        return NULL;
    }
    return &th_model->model;
}

static DNNModel *dnn_load_model_with_tokenizer_th(
    DnnContext *ctx, DNNFunctionType func_type, const char **labels,int label_count,
     int *softmax_units, int softmax_units_count,
     const char *tokenizer_path, AVFilterContext *filter_ctx)
{
    int ret;
    THModel *th_model = init_model_th(ctx, func_type, filter_ctx);
    if (th_model == NULL) {
        return NULL;
    }
#if (CONFIG_LIBTOKENIZERS == 1)
    // Check if this is a CLXP model and initialize accordingly
    auto model = &th_model->model;
    if ((func_type == DFT_ANALYTICS_CLIP || func_type == DFT_ANALYTICS_CLAP)) {
        ret = init_clxp_model(th_model, func_type, labels, label_count,
            tokenizer_path,filter_ctx);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Failed to initialize CLXP model\n");
            dnn_free_model_th(&model);
            return NULL;
        }
        ret = copy_softmax_units(th_model, softmax_units, softmax_units_count);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Failed to copy softmax units\n");
            dnn_free_model_th(&model);
            return NULL;
        }
    }
    c10::Device device = (*th_model->jit_model->parameters().begin()).device();

    if(func_type == DFT_ANALYTICS_CLIP) {
        ret = test_clip_inference(th_model,device);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Failed to test CLIP inference\n");
            dnn_free_model_th(&model);
            return NULL;
        }
    }
    else if(func_type == DFT_ANALYTICS_CLAP){
        ret = test_clap_inference(th_model, th_model->ctx->torch_option.sample_rate, th_model->ctx->torch_option.sample_duration,device);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Failed to test CLAP inference\n");
            dnn_free_model_th(&model);
            return NULL;
        }
    }
#endif
    return &th_model->model;
}

static int dnn_execute_model_th(const DNNModel *model,
                                DNNExecBaseParams *exec_params)
{
    THModel *th_model = (THModel *)model;
    DnnContext *ctx = th_model->ctx;
    TaskItem *task;
    THRequestItem *request;
    int ret = 0;

    ret = ff_check_exec_params(ctx, DNN_TH, model->func_type, exec_params);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "exec parameter checking fail.\n");
        return ret;
    }

    task = (TaskItem *)av_malloc(sizeof(TaskItem));
    if (!task) {
        av_log(ctx, AV_LOG_ERROR, "unable to alloc memory for task item.\n");
        return AVERROR(ENOMEM);
    }

    ret = ff_dnn_fill_task(task, exec_params, th_model, 0, 1);
    if (ret != 0) {
        av_freep(&task);
        av_log(ctx, AV_LOG_ERROR, "unable to fill task.\n");
        return ret;
    }

    ret = ff_queue_push_back(th_model->task_queue, task);
    if (ret < 0) {
        av_freep(&task);
        av_log(ctx, AV_LOG_ERROR, "unable to push back task_queue.\n");
        return ret;
    }

    ret = extract_lltask_from_task(model->func_type, task,
                                   th_model->lltask_queue, exec_params);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR,
               "unable to extract last level task from task.\n");
        return ret;
    }

    if (task->inference_todo == 0) {
        return 0;
    }

    request = (THRequestItem *)ff_safe_queue_pop_front(th_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }

    request->lltasks = (LastLevelTaskItem **)av_malloc_array(
        task->inference_todo, sizeof(*request->lltasks));
    if (!request->lltasks) {
        av_log(ctx, AV_LOG_ERROR, "unable to create lltasks.\n");
        return AVERROR(EINVAL);
    }
    request->lltask_count = 0;

    return execute_model_th(request, th_model->lltask_queue);
}

static DNNAsyncStatusType dnn_get_result_th(const DNNModel *model, AVFrame **in,
                                            AVFrame **out)
{
    THModel *th_model = (THModel *)model;
    return ff_dnn_get_result_common(th_model->task_queue, in, out);
}

static int dnn_flush_th(const DNNModel *model)
{
    THModel *th_model = (THModel *)model;
    THRequestItem *request;

    if (ff_queue_size(th_model->lltask_queue) == 0)
        // no pending task need to flush
        return 0;

    request = (THRequestItem *)ff_safe_queue_pop_front(th_model->request_queue);
    if (!request) {
        av_log(th_model->ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }

    return execute_model_th(request, th_model->lltask_queue);
}


extern const DNNModule ff_dnn_backend_torch = {
    .clazz          = DNN_DEFINE_CLASS(dnn_th),
    .type           = DNN_TH,
    .load_model     = dnn_load_model_th,
    .load_model_with_tokenizer = dnn_load_model_with_tokenizer_th,
    .execute_model  = dnn_execute_model_th,
    .get_result     = dnn_get_result_th,
    .flush          = dnn_flush_th,
    .free_model     = dnn_free_model_th,
};
