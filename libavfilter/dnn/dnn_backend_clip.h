#ifndef AVFILTER_DNN_CLIP_BACKEND_H
#define AVFILTER_DNN_CLIP_BACKEND_H

#include <string>
#include <vector>
#include <memory>
#include <tokenizers_cpp.h>
#include <torch/torch.h>

extern "C" {
#include "libavutil/frame.h"
#include "../dnn_interface.h"
}

using tokenizers::Tokenizer;

typedef struct THClipContext {
    std::unique_ptr<Tokenizer> tokenizer;
    std::vector<std::string> labels;
    std::string tokenizer_path;
} THClipContext;

int init_clip_model(void *model_ptr, AVFilterContext *filter_ctx);
void free_clip_context(THClipContext *clip_ctx);
int fill_model_input_clip(void *model_ptr, void *request);
int encode_image_clip(void *model_ptr, void *request);
int encode_text_clip(void *model_ptr, void *request);
int set_params_clip(void *model_ptr, const char **labels, int label_count, const char *tokenizer_path);
int process_clip_inference(THModel *th_model, THInferRequest *infer_request, const c10::Device& device, DnnContext *ctx);
std::string LoadBytesFromFile(const std::string& path)
#endif