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
 #include <tokenizers_cpp.h>
 
 using tokenizers::Tokenizer;
 
 typedef struct THClipContext {
     std::unique_ptr<Tokenizer> tokenizer;
     std::vector<std::string> labels;
     std::string tokenizer_path;
     int64_t resolution;
 } THClipContext;
 
 #define CLAP_SAMPLE_RATE 48000
 
 int init_clip_model(THModel *th_model, DNNFunctionType func_type, const AVFilterContext *filter_ctx, const c10::Device &device);
 int forward_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device);
 int process_clip_similarity(const THModel *th_model, const THRequestItem *request, const c10::Device& device);
 int forward_clap(const THModel *th_model, const THRequestItem *request, const c10::Device& device);
 
 int encode_image_clip(const THModel *th_model, torch::Tensor *input_tensor, const c10::Device& device, bool preprocessing);
 int encode_audio_clap(const THModel *th_model, const THRequestItem *request);
 int encode_images_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device);
 int encode_text_clip(const THModel *th_model, const THRequestItem *request, const c10::Device& device);
 
 int set_params_clip(const THModel *th_model, const char **labels, const int& label_count,
                    const char *tokenizer_path);
 void free_clip_context(THClipContext *clip_ctx);
 
 #endif
 #endif