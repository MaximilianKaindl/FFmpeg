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
 #ifndef AVFILTER_DNN_DNN_BACKEND_TORCH_COMMON_H
 #define AVFILTER_DNN_DNN_BACKEND_TORCH_COMMON_H
 
 struct THClipContext;
 
 extern "C" {
 #include "dnn_io_proc.h"
 #include "dnn_backend_common.h"
 #include "libavutil/opt.h"
 #include "libavutil/mem.h"
 #include "queue.h"
 #include "safe_queue.h"
 #include "libavutil/avassert.h"
 #include "libavutil/detection_bbox.h"
 #include "libavutil/avstring.h"
 }
 
 #include <torch/script.h>
 #include <torch/torch.h>
 
 #if (CONFIG_LIBTORCH_CUDA == 1)
 #include <c10/cuda/CUDAStream.h>
 #include <ATen/cuda/CUDAContext.h>
 #endif
 
 
 typedef struct THModel {
     DNNModel model;
     DnnContext *ctx;
     torch::jit::Module *jit_model;
     SafeQueue *request_queue;
     Queue *task_queue;
     Queue *lltask_queue;       
 
     #if CONFIG_LIBTOKENIZERS
     bool is_clip_model;
     bool is_clap_model;
     THClipContext *clip_ctx;
     #endif
 
 } THModel;
 
 typedef struct THInferRequest {
     torch::Tensor *output;
     torch::Tensor *input_tensor;
     
     #if CONFIG_LIBTOKENIZERS
     std::vector<torch::Tensor> *text_embeddings;
     #endif
 
 } THInferRequest;
 
 typedef struct THRequestItem {
     THInferRequest *infer_request;
     LastLevelTaskItem **lltasks;    
     int lltask_count;
     DNNAsyncExecModule exec_module;
 } THRequestItem;
 
 #endif