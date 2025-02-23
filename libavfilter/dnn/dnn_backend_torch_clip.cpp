/*
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

#include "dnn_backend_torch_clip.h"
#if (CONFIG_LIBTOKENIZERS == 1)

extern "C" {
#include "libavutil/mem.h"
#include "libavutil/log.h"
#include "libswscale/swscale.h"
#include "libavformat/avio.h"
}

static torch::Tensor get_tokens(const THModel *th_model, const std::string& prompt) {
    DnnContext *ctx = th_model->ctx;
    const int expected_length = EMBEDDING_SIZE_CLIP;

    try {
        if (!th_model->clip_ctx || !th_model->clip_ctx->tokenizer) {
            throw std::runtime_error("Tokenizer not initialized");
        }

        int32_t start_token = th_model->clip_ctx->tokenizer->TokenToId(START_TOKEN_CLIP);
        int32_t end_token = th_model->clip_ctx->tokenizer->TokenToId(END_TOKEN_CLIP);

        // Create vector with correct size, filled with padding tokens
        std::vector<int64_t> padded_ids(expected_length, PADDING_TOKEN_CLIP);

        // Add start token
        padded_ids[0] = start_token;

        try {
            // Get tokens from the tokenizer
            std::vector<int> tokens = th_model->clip_ctx->tokenizer->Encode(prompt);

            // Calculate how many tokens we can copy (leaving space for start and end tokens)
            const size_t max_text_tokens = expected_length - 2;

            const size_t num_tokens = tokens.size();
            if(num_tokens > max_text_tokens) {
                av_log(ctx, AV_LOG_WARNING, "Input text is too long, truncating to %ld tokens\n", max_text_tokens);
            }
            // Copy tokens after the start token
            size_t i;
            for (i = 0; i < num_tokens; i++) {
                padded_ids[i + 1] = tokens[i];
            }
            padded_ids[i+1] = end_token;

            auto tensor = torch::from_blob(
                padded_ids.data(),
                {1, expected_length}, 
                torch::kInt64
            ).clone(); 

            return tensor;

        } catch (const std::exception& e) {
            av_log(ctx, AV_LOG_ERROR, "Token encoding failed: %s\n", e.what());
            // Return empty tensor with correct dimensions on error
            return torch::zeros({1, expected_length}, torch::kInt64);
        }

    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Token generation failed: %s\n", e.what());
        return torch::zeros({1, expected_length}, torch::kInt64);
    }
}

static int load_bytes_from_file(const std::string& path, std::string& data, DnnContext* log_ctx) {
    AVIOContext *ctx = NULL;
    int ret;
    int64_t size;
    
    ret = avio_open(&ctx, path.c_str(), AVIO_FLAG_READ);
    if (ret < 0) {
        av_log(log_ctx, AV_LOG_ERROR, "Cannot open file: %s\n", path.c_str());
        return ret;
    }

    size = avio_size(ctx);
    if (size < 0) {
        av_log(log_ctx, AV_LOG_ERROR, "Failed to determine file size: %s\n", path.c_str());
        return size;
    }

    try {
        data.resize(size);
        ret = avio_read(ctx, (unsigned char*)data.data(), size);
        if (ret < 0) {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to read file: %s\n", path.c_str());
            return ret;
        }
        if (ret != size) {
            av_log(log_ctx, AV_LOG_ERROR, "Incomplete read: %s\n", path.c_str());
            return AVERROR(EIO);
        }
    } catch (const std::exception& e) {
        av_log(log_ctx, AV_LOG_ERROR, "Exception while reading file %s: %s\n", 
               path.c_str(), e.what());
        return AVERROR(ENOMEM);
    }

    return 0;
}

int create_tokenizer(const THModel *th_model, const std::string& tokenizer_path) {
    //Dont create tokenizer if it already exists
    if (th_model->clip_ctx->tokenizer) {
        return 0;
    }

    std::string blob;
    int ret = load_bytes_from_file(tokenizer_path, blob, th_model->ctx);
    if (ret < 0) {
        return ret;
    }

    try {
        th_model->clip_ctx->tokenizer = Tokenizer::FromBlobJSON(blob);
    } catch (const c10::Error& e) {
        av_log(th_model->ctx, AV_LOG_ERROR, "Error creating tokenizer: %s\n", e.what());
        return AVERROR(EINVAL);
    }
    return 0;
}

int init_clip_model(THModel *th_model, const AVFilterContext *filter_ctx) {
    try {
        //Should throw exception if not existing
        auto encode_image = th_model->jit_model->get_method("encode_image");
        auto encode_text = th_model->jit_model->get_method("encode_text");
        th_model->is_clip_model = true;
        th_model->clip_ctx = (THClipContext *)av_mallocz(sizeof(THClipContext));
        th_model->clip_ctx->logit_scale = std::exp(std::log(1.0f / 0.07f));
        av_log(th_model->ctx, AV_LOG_INFO, 
               "Successfully initialized CLIP model\n");
        return 0;

    } catch (const c10::Error& e) {
        av_log(th_model->ctx, AV_LOG_ERROR, 
               "Error during CLIP model initialization: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

int apply_clip_image_preprocessing(torch::Tensor *input_tensor) {
    // Resize using bicubic interpolation
    auto resized = torch::nn::functional::interpolate(
        *input_tensor,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{224, 224})
            .mode(torch::kBicubic)
            .align_corners(false)
    );
    
    //Manual center crop if needed
    auto h = resized.size(2);
    auto w = resized.size(3);
    int64_t crop_size = 224;

    int64_t h_start = (h - crop_size) / 2;
    int64_t w_start = (w - crop_size) / 2;

    auto cropped = resized.slice(2, h_start, h_start + crop_size)
                        .slice(3, w_start, w_start + crop_size);

    // Apply CLIP specific normalization
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input_tensor->device());
    auto mean = torch::tensor({0.48145466, 0.4578275, 0.40821073}, options).view({1, 3, 1, 1});
    auto std = torch::tensor({0.26862954, 0.26130258, 0.27577711}, options).view({1, 3, 1, 1});
    
    *input_tensor = (resized - mean) / std;
    return 0;
}

int encode_image_clip(const THModel *th_model, torch::Tensor *input_tensor, const c10::Device& device) {
    DnnContext *ctx = th_model->ctx;
    try {               
        if (input_tensor->device() != device) {
            *input_tensor = input_tensor->to(device);
        }

        apply_clip_image_preprocessing(input_tensor);

        // Get image features
        auto image_features = th_model->jit_model->run_method(
            "encode_image",
            *input_tensor,
            true  // normalize
        );

        if (!image_features.isTensor()) {
            av_log(ctx, AV_LOG_ERROR, "Model returned invalid non-tensor output\n");
            return AVERROR(EINVAL);
        }

        // Update input tensor with the encoded features
        *input_tensor = image_features.toTensor();
        return 0;

    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Image encoding error: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

int encode_images_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device) {
    THInferRequest *infer_request = request->infer_request;
    DnnContext *ctx = th_model->ctx;
    std::vector<torch::Tensor> image_features;

    try {
        // Process each image in the batch
        for (int i = 0; i < request->lltask_count; i++) {
            // Get current tensor (should be [C, H, W])
            auto current_tensor = infer_request->input_tensor->select(0, i);
            
            // Unsqueeze to add batch dimension [1, C, H, W]
            current_tensor = current_tensor.unsqueeze(0);
            
            int ret = encode_image_clip(th_model, &current_tensor, device);
            if (ret < 0) {
                av_log(ctx, AV_LOG_ERROR, "Image encoding failed for batch item %d\n", i);
                return ret;
            }
            
            image_features.push_back(current_tensor.squeeze(0));  // Remove batch dimension for stacking
        }

        // Stack all image features
        if (!image_features.empty()) {
            *infer_request->input_tensor = torch::stack(image_features);
        } else {
            av_log(ctx, AV_LOG_ERROR, "No valid images to process in batch\n");
            return AVERROR(EINVAL);
        }
        return 0;
        
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Batch processing error: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

int encode_text_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device) {
    THInferRequest *infer_request = request->infer_request;
    DnnContext *ctx = th_model->ctx;
    THClipContext *clip_ctx = th_model->clip_ctx;
    infer_request->text_embeddings = new std::vector<torch::Tensor>();

    try {
        infer_request->text_embeddings->reserve(clip_ctx->labels.size());

        for (const auto& label : clip_ctx->labels) {
            torch::Tensor tokens = get_tokens(th_model, label);

            if (tokens.device() != device) 
                tokens = tokens.to(device);

            auto text_embedding = th_model->jit_model->run_method(
                "encode_text",
                tokens, 
                true // normalize
            );

            if (!text_embedding.isTensor()) {
                av_log(ctx, AV_LOG_ERROR, "Model returned invalid non-tensor output for text encoding\n");
                return AVERROR(EINVAL);
            }
            infer_request->text_embeddings->push_back(text_embedding.toTensor());
        }
        return 0;
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Text encoding error: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

int forward_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device)
{
    int ret;
    ret = encode_images_clip(th_model, request, device);
    if (ret < 0) {
        av_log(th_model->ctx, AV_LOG_ERROR, "Image encoding failed in CLIP preprocessing\n");
        return ret;
    }
    ret = encode_text_clip(th_model, request, device);
    if (ret < 0) {
        av_log(th_model->ctx, AV_LOG_ERROR, "Text encoding failed in CLIP preprocessing\n");
        return ret;
    }
    ret = process_clip_similarity(th_model, request, device);
    if (ret < 0) {
        av_log(th_model->ctx, AV_LOG_ERROR, "Error in CLIP Similarity calculation\n");
        return ret;
    }
    return 0;
}

int fill_model_input_clip(const THModel *th_model, const THRequestItem *request, const DNNData& input)
{
    DnnContext *ctx = th_model->ctx;
    THInferRequest *infer_request = request->infer_request;
    *infer_request->output = infer_request->input_tensor->clone().detach();

    // Verify the clone worked
    if (!infer_request->output->defined() || infer_request->output->sizes() != infer_request->input_tensor->sizes()) {
        av_log(ctx, AV_LOG_ERROR, "Tensor cloning failed\n");
        return AVERROR(EINVAL);
    }

    int ret;
    ret = create_tokenizer(th_model, th_model->clip_ctx->tokenizer_path);
    if(ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Error creating tokenizer\n");
        return ret;
    }
    return 0;
}

int set_params_clip(const THModel *th_model, const char **labels, const int& label_count, const char *tokenizer_path) {
    if (!labels || label_count <= 0) {
        av_log(th_model->ctx, AV_LOG_ERROR, "Label file invalid.\n");
        return AVERROR(EINVAL);
    }

    std::vector<std::string> label_vector;
    label_vector.reserve(label_count); 

    for (int i = 0; i < label_count; i++) {
        if (labels[i]) {
            label_vector.emplace_back(labels[i]);
        }
    }
    th_model->clip_ctx->labels = label_vector;
    th_model->clip_ctx->tokenizer_path = tokenizer_path;
    return 0;
}

static torch::Tensor calculate_clip_similarity_matrix(const torch::Tensor& image_features, const torch::Tensor& text_embedding, DnnContext *ctx) {
    try {
        return torch::matmul(image_features, text_embedding.transpose(0,1));    
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Similarity computation failed: %s\n", e.what());
        return {};
    }
}

int process_clip_similarity(const THModel *th_model, const THRequestItem *request, const c10::Device& device) {
    DnnContext *ctx = th_model->ctx;
    THInferRequest *infer_request = request->infer_request;
    int64_t batch_size = infer_request->input_tensor->size(0);
    std::vector<std::vector<float>> all_similarity_scores(batch_size);
    auto embedding_count = infer_request->text_embeddings->size();
    std::vector<torch::Tensor> batch_tensors;

    try {
        // Process each item in batch
        for (int64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            auto image_features = infer_request->input_tensor->select(0, batch_idx);
            std::vector<float>& similarity_scores = all_similarity_scores[batch_idx];
            similarity_scores.reserve(embedding_count);

            for (size_t i = 0; i < embedding_count; i++) {
                if((*infer_request->text_embeddings)[i].device() != device) {
                    (*infer_request->text_embeddings)[i] = (*infer_request->text_embeddings)[i].to(device);
                }
                auto similarity = calculate_clip_similarity_matrix(image_features, (*infer_request->text_embeddings)[i], ctx);
                auto similarity_value = similarity.item<float>();
                similarity_scores.push_back(similarity_value);

                av_log(ctx, AV_LOG_DEBUG, "BBox %ld, Label %s: logit_value=%.4f\n",
                       batch_idx, th_model->clip_ctx->labels[i].c_str(), similarity_value);
            }
            batch_tensors.push_back(torch::tensor(similarity_scores));
        }
        
        // Stack all scores into a single batch tensor
        auto scores_tensor = torch::stack(batch_tensors);
        infer_request->output = new torch::Tensor(scores_tensor);
        
        if (!infer_request->output->defined()) {
            av_log(ctx, AV_LOG_ERROR, "Failed to create output tensor\n");
            return AVERROR(EINVAL);
        }
        return 0;

    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "CLIP similarity computation error: %s\n", e.what());
        return AVERROR(EINVAL);
    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Error computing similarities: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

void free_clip_context(THClipContext *clip_ctx) {
    if (!clip_ctx)
        return;

    clip_ctx->labels.clear();
    clip_ctx->tokenizer.release();
    av_freep(clip_ctx);
}
#endif