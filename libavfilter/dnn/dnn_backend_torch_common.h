#ifndef AVFILTER_DNN_BACKEND_TORCH_COMMON_H
#define AVFILTER_DNN_BACKEND_TORCH_COMMON_H

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>

struct THClipContext;

extern "C" {
#include "dnn_io_proc.h"
#include "dnn_backend_common.h"
#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "queue.h"
#include "safe_queue.h"
}

typedef struct THModel {
    DNNModel model;
    DnnContext *ctx;
    torch::jit::Module *jit_model;
    SafeQueue *request_queue;
    Queue *task_queue;
    Queue *lltask_queue;
    bool is_clip_model; 
    THClipContext *clip_ctx;
} THModel;

typedef struct THInferRequest {
    torch::Tensor *output;
    torch::Tensor *input_tensor;
    std::vector<torch::Tensor> *text_embeddings;
} THInferRequest;

typedef struct THRequestItem {
    THInferRequest *infer_request;
    LastLevelTaskItem *lltask;
    DNNAsyncExecModule exec_module;
} THRequestItem;

#endif