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
#include "dnn_tokenizer.h"

extern "C" {
#include "libavutil/mem.h"
#include "libavutil/log.h"
#include "libavfilter/dnn/dnn_io_proc.h"
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

int encode_audio_clap(const THModel *th_model, const THRequestItem *request) {
    THInferRequest *infer_request = request->infer_request;
    LastLevelTaskItem *lltask = request->lltasks[0];
    TaskItem *task = lltask->task;
    DnnContext *ctx = th_model->ctx;
    float *resampled_data = NULL;
    int resampled_nb_samples = 0;
    // Resample audio to 48kHz if needed
    int ret = ff_frame_to_dnn_clap(task->in_frame, CLAP_SAMPLE_RATE, &resampled_data, &resampled_nb_samples);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Audio resampling failed\n");
        return ret;
    }

    try {
        // Create input tensor from resampled audio
        *infer_request->input_tensor = torch::from_blob(resampled_data,
                                           {1, resampled_nb_samples},
                                           torch::kFloat32).clone();
    } catch (const c10::Error& e) {
        av_freep(&resampled_data);
        av_log(ctx, AV_LOG_ERROR, "Audio encoding error: %s\n", e.what());
        return AVERROR(EINVAL);
    }
    av_freep(&resampled_data);
    return 0;
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
            // Use the get_tokens function (not get_tokens_with_mask) as CLIP doesn't need attention mask
            std::vector<int64_t> tokens_vec = get_tokens(
                clip_ctx->tokenizer,
                label,
                DEFAULT_MAX_LENGTH,
                ctx
            );
            
            // Convert vector to tensor and add batch dimension
            torch::Tensor tokens = torch::tensor(tokens_vec, torch::kInt64).unsqueeze(0);
            
            // Move tensor to the correct device
            if (tokens.device() != device) {
                tokens = tokens.to(device);
            }
            
            av_log(ctx, AV_LOG_DEBUG, "Encoding text '%s' for CLIP\n", label.c_str());
            
            // Run text through the model's encode_text method - no attention mask needed
            auto text_embedding = th_model->jit_model->run_method(
                "encode_text",
                tokens, 
                true // normalize
            );

            if (!text_embedding.isTensor()) {
                av_log(ctx, AV_LOG_ERROR, "Model returned invalid non-tensor output for text encoding: %s\n", label.c_str());
                return AVERROR(EINVAL);
            }
            
            infer_request->text_embeddings->push_back(text_embedding.toTensor());
        }
        
        return 0;
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Text encoding error: %s\n", e.what());
        return AVERROR(EINVAL);
    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Exception in encode_text_clip: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

int forward_clap(const THModel *th_model, const THRequestItem *request, const c10::Device& device)
{
    THInferRequest *infer_request = request->infer_request;
    DnnContext *ctx = th_model->ctx;
    THClipContext *clip_ctx = th_model->clip_ctx;
    infer_request->text_embeddings = new std::vector<torch::Tensor>();
    
    try {
        // Use the actual audio tensor from input_tensor instead of creating dummy audio
        torch::Tensor audio_tensor = *infer_request->input_tensor;
        if (audio_tensor.device() != device) {
            audio_tensor = audio_tensor.to(device);
        }
        // Store all similarity results for each label
        std::vector<torch::Tensor> similarities;
        
        for (const auto& label : clip_ctx->labels) {
            // Use the get_tokens_with_mask function to get both tokens and attention mask
            if (clip_ctx->tokenizer) {
                // Get tokens and attention mask using the helper function
                auto [tokens_vec, attention_mask_vec] = get_tokens_with_mask(
                    clip_ctx->tokenizer,
                    label,
                    DEFAULT_MAX_LENGTH,
                    ctx
                );

                torch::Tensor tokens = torch::tensor(tokens_vec, torch::kInt64);
                torch::Tensor attention_mask = torch::tensor(attention_mask_vec, torch::kInt64);
                
                // Move tensors to the correct device
                if (tokens.device() != device) {
                    tokens = tokens.to(device);
                }
                if (attention_mask.device() != device) {
                    attention_mask = attention_mask.to(device);
                }
                
                // Convert vectors to tensors
                tokens = tokens.unsqueeze(0);
                attention_mask = attention_mask.unsqueeze(0);
                av_log(ctx, AV_LOG_DEBUG, "Processing label '%s' with proper tokenization\n", label.c_str());
                
                // Prepare inputs in the order expected by the model
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(audio_tensor);
                inputs.push_back(tokens);
                inputs.push_back(attention_mask);
                
                av_log(ctx, AV_LOG_INFO, "Running forward pass for audio and label: %s\n", 
                       label.c_str());
                
                // Execute forward pass
                try {
                    auto result = th_model->jit_model->forward(inputs);
                    av_log(ctx, AV_LOG_INFO, "Forward call succeeded for label: %s\n", label.c_str());
                    
                    if (result.isTensor()) {
                        similarities.push_back(result.toTensor());
                    } else {
                        av_log(ctx, AV_LOG_ERROR, "Model returned non-tensor output for label: %s\n", 
                               label.c_str());
                    }
                } catch (const c10::Error& e) {
                    av_log(ctx, AV_LOG_ERROR, "Forward call failed for label %s: %s\n", 
                           label.c_str(), e.what());
                    return AVERROR(EINVAL);
                }
            } else {
                av_log(ctx, AV_LOG_ERROR, "No tokenizer available for processing text\n");
                return AVERROR(EINVAL);
            }
        }
        
        // Create a tensor with all similarities and assign to output
        if (!similarities.empty()) {
            auto all_similarities = torch::cat(similarities, 1);
            infer_request->output = new torch::Tensor(all_similarities);
            return 0;
        } else {
            av_log(ctx, AV_LOG_ERROR, "No similarity results computed\n");
            return AVERROR(EINVAL);
        }
    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Exception in forward_clap: %s\n", e.what());
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

    if (!th_model->clip_ctx->tokenizer && !th_model->clip_ctx->tokenizer_path.empty()) {
        th_model->clip_ctx->tokenizer = create_tokenizer(th_model->clip_ctx->tokenizer_path, th_model->ctx);
        if (!th_model->clip_ctx->tokenizer) {
            av_log(th_model->ctx, AV_LOG_ERROR, "Error creating tokenizer\n");
            return AVERROR(EINVAL);
        }
    }

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