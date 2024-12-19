#ifndef AVFILTER_DNN_TORCH_CLIP_BACKEND_H
#define AVFILTER_DNN_TORCH_CLIP_BACKEND_H

#include <string>
#include <vector>
#include <memory>
#include <tokenizers_cpp.h>
#include "dnn_backend_torch_common.h"

using tokenizers::Tokenizer;

typedef struct THClipContext {
    std::unique_ptr<Tokenizer> tokenizer;
    std::vector<std::string> labels;
    std::string tokenizer_path;
} THClipContext;

int create_tokenizer(THModel *th_model, std::string tokenizer_path);
int init_clip_model(THModel *th_model, AVFilterContext *filter_ctx);
void free_clip_context(THClipContext *clip_ctx);
int fill_model_input_clip(THModel *th_model, THRequestItem *request, DNNData input);
int encode_image_clip(THModel *th_model, THRequestItem *request);
int encode_text_clip(THModel *th_model, THRequestItem *request);
int set_params_clip(THModel *th_model, const char **labels, int label_count, const char *tokenizer_path);
int process_clip_inference(THModel *th_model, THInferRequest *infer_request, const c10::Device& device, DnnContext *ctx);

#endif