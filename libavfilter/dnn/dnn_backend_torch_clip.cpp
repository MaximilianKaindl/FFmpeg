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

int get_clip_input_res(THModel *th_model, const c10::Device &device) {
    // Common CLIP input dimensions to test
    std::vector<int64_t> test_dims[] = {
        {1, 3, 224, 224},  
        {1, 3, 256, 256}, 
        {1, 3, 384, 384},
        {1, 3, 378, 378},
    };
    bool found_dims = false;
    int64_t resolution = 0;

    for (const auto& dims : test_dims) {
        // Create test input tensor
        torch::Tensor test_input = torch::zeros(dims);
        
        if(encode_image_clip(th_model, &test_input, device, false) < 0) {
            continue;
        }
        resolution = dims[2];
        found_dims = true;
        break;
    }
    if(!found_dims){
        return AVERROR(EINVAL);
    }
    return resolution;
}

int init_clip_model(THModel *th_model, DNNFunctionType func_type, const AVFilterContext *filter_ctx, const c10::Device &device) {
    th_model->clip_ctx = (THClipContext *)av_mallocz(sizeof(THClipContext));
    if (!th_model->clip_ctx) {
        av_log(th_model->ctx, AV_LOG_ERROR, "Failed to allocate memory for CLIP context\n");
        return AVERROR(ENOMEM);
    }
    if(func_type == DFT_ANALYTICS_CLAP){
        th_model->is_clap_model = true;
    }
    else{
        try {
            //Should throw exception if not existing
            auto encode_text = th_model->jit_model->get_method("encode_text");
            auto encode_image = th_model->jit_model->get_method("encode_image");
            th_model->is_clip_model = true;
            th_model->clip_ctx->resolution = get_clip_input_res(th_model, device);
            if(th_model->clip_ctx->resolution <= 0){
                av_log(th_model->ctx, AV_LOG_ERROR, 
                        "Failed to determine input resolution for CLIP model\n");
                return AVERROR(EINVAL);
            }
        } catch (const c10::Error& e) {
            av_log(th_model->ctx, AV_LOG_ERROR, 
                   "Error during CLIP model initialization: %s\n", e.what());
            return AVERROR(EINVAL);
        }
        av_log(th_model->ctx, AV_LOG_INFO, 
            "Successfully initialized CLIP model\n");
    }
    
    return 0;
}

static int resample_audio(AVFrame *frame, int target_sample_rate, float **resampled_data, int *resampled_nb_samples) {
    SwrContext *swr_ctx = NULL;
    int ret = 0;
    AVChannelLayout out_ch_layout = AV_CHANNEL_LAYOUT_MONO;
    AVChannelLayout in_ch_layout;
    av_channel_layout_copy(&in_ch_layout, &frame->ch_layout);

    ret = swr_alloc_set_opts2(&swr_ctx,
                            &out_ch_layout,          // out_ch_layout
                            AV_SAMPLE_FMT_FLT,        // out_sample_fmt
                            target_sample_rate,       // out_sample_rate
                            &in_ch_layout,           // in_ch_layout
                            (AVSampleFormat)frame->format, // in_sample_fmt
                            frame->sample_rate,       // in_sample_rate
                            0, NULL);

    av_channel_layout_uninit(&in_ch_layout);
    
    if (ret < 0) {
        return AVERROR(ENOMEM);
    }
    
    // Initialize the resampler
    ret = swr_init(swr_ctx);
    if (ret < 0) {
        swr_free(&swr_ctx);
        return ret;
    }
    
    // Calculate output number of samples
    *resampled_nb_samples = av_rescale_rnd(frame->nb_samples,
                                          target_sample_rate,
                                          frame->sample_rate,
                                          AV_ROUND_UP);
    
    // Allocate output buffer
    *resampled_data = (float*)av_malloc(*resampled_nb_samples * sizeof(float));
    if (!*resampled_data) {
        swr_free(&swr_ctx);
        return AVERROR(ENOMEM);
    }
    
    // Do the actual resampling
    ret = swr_convert(swr_ctx,
                      (uint8_t**)resampled_data, *resampled_nb_samples,
                      (const uint8_t**)frame->data, frame->nb_samples);
    
    swr_free(&swr_ctx);
    
    if (ret < 0) {
        av_freep(resampled_data);
        return ret;
    }
    
    return 0;
}

int encode_audio_clap(const THModel *th_model, const THRequestItem *request, const c10::Device& device) {
    THInferRequest *infer_request = request->infer_request;
    LastLevelTaskItem *lltask = request->lltasks[0];
    TaskItem *task = lltask->task;
    DnnContext *ctx = th_model->ctx;
    float *resampled_data = NULL;
    int resampled_nb_samples = 0;
    
    try {
        // Resample audio to 48kHz if needed
        int ret = resample_audio(task->in_frame, CLAP_SAMPLE_RATE,
                               &resampled_data, &resampled_nb_samples);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Audio resampling failed\n");
            return ret;
        }
        
        // Create input tensor from resampled audio
        *infer_request->input_tensor = torch::from_blob(resampled_data,
                                           {1, resampled_nb_samples},
                                           torch::kFloat32).clone();
        
        if (infer_request->input_tensor->device() != device) 
            *infer_request->input_tensor = infer_request->input_tensor->to(device);
        
        av_freep(&resampled_data);
        return 0;
        
    } catch (const c10::Error& e) {
        av_freep(&resampled_data);
        av_log(ctx, AV_LOG_ERROR, "Audio encoding error: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

int encode_image_clip(const THModel *th_model, torch::Tensor *input_tensor, const c10::Device& device, bool preprocessing) {
    DnnContext *ctx = th_model->ctx;
    try {               
        if (input_tensor->device() != device) {
            *input_tensor = input_tensor->to(device);
        }

        if(preprocessing){
            *input_tensor = torch::nn::functional::interpolate(
                *input_tensor,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{th_model->clip_ctx->resolution, th_model->clip_ctx->resolution})
                    .mode(torch::kBicubic)
                    .align_corners(false)
            );
        }

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
            
            int ret = encode_image_clip(th_model, &current_tensor, device, true);
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

int forward_clap(const THModel *th_model, const THRequestItem *request, const c10::Device& device)
{
    THInferRequest *infer_request = request->infer_request;
    DnnContext *ctx = th_model->ctx;
    THClipContext *clip_ctx = th_model->clip_ctx;
    infer_request->text_embeddings = new std::vector<torch::Tensor>();
    int ret;

    ret = encode_audio_clap(th_model, request, device);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Audio encoding failed in CLAP preprocessing\n");
        return ret;
    }

    try {
        // Store all similarity results for each label
        std::vector<torch::Tensor> similarities;
        
        for (const auto& label : clip_ctx->labels) {
            torch::Tensor tokens = get_tokens(th_model, label);

            if (tokens.device() != device) 
                tokens = tokens.to(device);
            
            // Create attention mask (all ones to match the model expectation)
            torch::Tensor attention_mask = torch::ones({1, tokens.size(1)}, torch::kInt64).to(device);
            av_log(ctx, AV_LOG_INFO, "About to call forward with tensor shapes: audio=[%ld, %ld], tokens=[%ld, %ld], mask=[%ld, %ld]\n",
               infer_request->input_tensor->size(0), infer_request->input_tensor->size(1),
               tokens.size(0), tokens.size(1),
               attention_mask.size(0), attention_mask.size(1));
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(*infer_request->input_tensor);  // audio
            inputs.push_back(tokens);                        // input_ids
            inputs.push_back(attention_mask);                // attention_mask instead of zeros
            
            // Run the model and store the result
            auto similarity = th_model->jit_model->forward(inputs).toTensor();
            similarities.push_back(similarity);
        }
        
        // Create a tensor with all similarities and assign to output
        if (!similarities.empty()) {
            auto all_similarities = torch::cat(similarities, 1);
            *infer_request->output = torch::Tensor(all_similarities);
        } else {
            av_log(ctx, AV_LOG_ERROR, "No similarity results computed\n");
            return AVERROR(EINVAL);
        }
        
        return 0;
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "CLAP forward error: %s\n", e.what());
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