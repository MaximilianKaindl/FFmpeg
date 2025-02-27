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

#ifndef AVFILTER_DNN_DNN_BACKEND_TORCH_CLIP_H
#define AVFILTER_DNN_DNN_BACKEND_TORCH_CLIP_H

#include "dnn_backend_torch_common.h"

#if (CONFIG_LIBTOKENIZERS == 1)
#include <string>
#include <memory>
#include <vector>
#include <torch/script.h>

typedef struct THClipContext {
    torch::Tensor *tokenized_text;
    torch::Tensor *attention_mask;
    int64_t resolution;
} THClipContext;

#define CLXP_EMBEDDING_DIMS 77
#define CLAP_SAMPLE_RATE 44100

int init_clip_model(THModel *th_model, DNNFunctionType func_type, const char **labels, int label_count,
                    const char *tokenizer_path, const AVFilterContext *filter_ctx);

int preprocess_image_tensor(const THModel *th_model, torch::Tensor *input_tensor, const c10::Device &device);
int prepare_audio_tensor(const THModel *th_model, const THRequestItem *request);
int prepare_images_tensors(const THModel *th_model, const THRequestItem *request, const c10::Device &device);
int encode_text_clip(const THModel *th_model, const THRequestItem *request, const c10::Device &device);
torch::Tensor calculate_clip_similarity_matrix(const torch::Tensor &image_features, const torch::Tensor &text_embedding, DnnContext *ctx);
void free_clip_context(THClipContext *clip_ctx);

#endif
#endif