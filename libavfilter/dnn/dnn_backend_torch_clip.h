#ifndef AVFILTER_DNN_TORCH_CLIP_BACKEND_H
#define AVFILTER_DNN_TORCH_CLIP_BACKEND_H

#if CONFIG_LIBTOKENIZERS
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

int init_clip_model(THModel *th_model, AVFilterContext *filter_ctx);
int fill_model_input_clip(THModel *th_model, THRequestItem *request, DNNData input);
torch::Tensor get_clip_tokens_tensor(THModel *th_model, THRequestItem *request);

int process_clip_similarity(THModel *th_model, THRequestItem *request, c10::Device device);
int extract_clip_outputs(THModel *th_model, THRequestItem *request, const c10::ivalue::Tuple* output);

int set_params_clip(THModel *th_model, const char **labels, int label_count, 
                   const char *tokenizer_path);

void free_clip_context(THClipContext *clip_ctx);

#endif
#endif