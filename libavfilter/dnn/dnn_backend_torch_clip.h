#ifndef AVFILTER_DNN_TORCH_CLIP_BACKEND_H
#define AVFILTER_DNN_TORCH_CLIP_BACKEND_H

#include <string>
#include <memory>
#include "dnn_backend_torch_common.h"

#include <tokenizers_cpp.h>
using tokenizers::Tokenizer;

typedef struct THClipContext {
    std::unique_ptr<Tokenizer> tokenizer;
    std::vector<std::string> labels;
    std::string tokenizer_path;
    float logit_scale;
} THClipContext;

// Core CLIP model functions
int init_clip_model(THModel *th_model, AVFilterContext *filter_ctx);
int fill_model_input_clip(THModel *th_model, THRequestItem *request, DNNData input);
int process_clip_inference(THModel *th_model, THInferRequest *infer_request, 
                         const c10::Device& device, DnnContext *ctx);

// Setup functions
int set_params_clip(THModel *th_model, const char **labels, int label_count, 
                   const char *tokenizer_path);

// Cleanup functions                   
void free_clip_context(THClipContext *clip_ctx);

#endif