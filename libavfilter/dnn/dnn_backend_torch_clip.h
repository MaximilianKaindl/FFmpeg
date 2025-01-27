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
#ifndef AVFILTER_DNN_BACKEND_TORCH_CLIP_BACKEND_H
#define AVFILTER_DNN_BACKEND_TORCH_CLIP_BACKEND_H

#include "dnn_backend_torch_common.h"

#if (CONFIG_LIBTOKENIZERS == 1)
#include <string>
#include <memory>
#include <vector>
#include <torch/script.h>
#include <tokenizers_cpp.h>

extern "C"{
#include "libavutil/detection_bbox.h"
}

using tokenizers::Tokenizer;

typedef struct THClipContext {
    std::unique_ptr<Tokenizer> tokenizer;
    std::vector<std::string> labels;
    std::string tokenizer_path;
    float logit_scale;
} THClipContext;

const std::string START_TOKEN_CLIP = "<|startoftext|>";
const std::string END_TOKEN_CLIP = "<|endoftext|>";
constexpr int32_t PADDING_TOKEN_CLIP = 0;
#define EMBEDDING_SIZE_CLIP 77

// Core CLIP functions
int init_clip_model(THModel *th_model, const AVFilterContext *filter_ctx);
int fill_model_input_clip(const THModel *th_model, const THRequestItem *request, const DNNData& input);
int forward_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device);
int process_clip_similarity(const THModel *th_model, const THRequestItem *request, const c10::Device& device);
int scale_frame_for_clip(AVFrame *in_frame, AVFrame **scaled_frame, DnnContext *ctx);

// Helper functions
int create_tokenizer(const THModel *th_model, const std::string& tokenizer_path);
int encode_image_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device);
int encode_text_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device);

// Parameter setting and cleanup
int set_params_clip(const THModel *th_model, const char **labels, const int& label_count,
                   const char *tokenizer_path);
void free_clip_context(THClipContext *clip_ctx);

#endif
#endif