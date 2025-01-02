#ifndef AVFILTER_DNN_TORCH_CLIP_BACKEND_H
#define AVFILTER_DNN_TORCH_CLIP_BACKEND_H

#include "dnn_backend_torch_common.h"

#if (CONFIG_LIBTOKENIZERS == 1)
#include <string>
#include <memory>
#include <vector>
#include <torch/script.h>
#include <tokenizers_cpp.h>

using tokenizers::Tokenizer;

typedef struct THClipContext {
    std::unique_ptr<Tokenizer> tokenizer;
    std::vector<std::string> labels;
    std::string tokenizer_path;
    float logit_scale;
} THClipContext;

// Core CLIP functions
int init_clip_model(THModel *th_model, const AVFilterContext *filter_ctx);
int fill_model_input_clip(const THModel *th_model, const THRequestItem *request, const DNNData& input);
int forward_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device);
int process_clip_similarity(const THModel *th_model, const THRequestItem *request, const c10::Device& device);

// Helper functions
torch::Tensor get_tokens(const THModel *th_model, const std::string& prompt);
int create_tokenizer(const THModel *th_model, const std::string& tokenizer_path);
int encode_image_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device);
int encode_text_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device);
torch::Tensor calculate_clip_similarity_matrix(const torch::Tensor& image_features, 
                                             const torch::Tensor& text_embedding,
                                             const float& logit_scale,
                                             const DnnContext *ctx,
                                             float temperature);

// Parameter setting and cleanup
int set_params_clip(const THModel *th_model, const char **labels, const int& label_count,
                   const char *tokenizer_path);
void free_clip_context(THClipContext *clip_ctx);

#endif // CONFIG_LIBTOKENIZERS
#endif // AVFILTER_DNN_TORCH_CLIP_BACKEND_H