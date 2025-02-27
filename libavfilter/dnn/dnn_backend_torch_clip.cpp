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

#include "dnn_backend_torch_clip.h"
#if (CONFIG_LIBTOKENIZERS == 1)

extern "C" {
#include "dnn_tokenizer.h"
#include "libavutil/mem.h"
#include "libavutil/log.h"
#include "libavfilter/dnn/dnn_io_proc.h"
#include "libswresample/swresample.h"
#include "libavutil/samplefmt.h"
#include "libswscale/swscale.h"
}

int get_clip_input_res(THModel *th_model, const c10::Device &device) {
  // Common CLIP input dimensions to test
  std::vector<int64_t> test_dims[] = {
      {1, 3, 224, 224},
      {1, 3, 256, 256},
      {1, 3, 384, 384},
      {1, 3, 378, 378},
  };
  bool found_dims = false;
  int64_t resolution = 0;

  for (const auto &dims: test_dims) {
    // Create test input tensor
    torch::Tensor test_input = torch::zeros(dims);
    if (test_input.device() != device) {
      test_input = test_input.to(device);
    }
    try {
        auto image_features = th_model->jit_model->run_method(
            "encode_image",
            test_input,
            true // normalize
        );
    } catch (const c10::Error &e) {
      continue;
    }
    resolution = dims[2];
    found_dims = true;
    break;
  }
  if (!found_dims) {
    return AVERROR(EINVAL);
  }
  return resolution;
}

int get_tokenized_batch(THClipContext *clip_ctx, const char **labels,
                        int label_count, const char *tokenizer_path,
                        DnnContext *ctx, const c10::Device &device) {
  int **tokens_array = NULL;
  int *token_counts = NULL;
  TokenizerHandle tokenizer = NULL;
  int ret;

  if (!labels || label_count <= 0) {
    av_log(ctx, AV_LOG_ERROR, "Label file invalid.\n");
    return AVERROR(EINVAL);
  }

  if (!tokenizer_path) {
    av_log(ctx, AV_LOG_ERROR, "Tokenizer path not provided.\n");
    return AVERROR(EINVAL);
  }

  ret = ff_dnn_create_tokenizer_and_encode_batch(
      tokenizer_path, labels, label_count, &tokens_array, &token_counts, ctx);

  if (ret < 0) {
    av_log(ctx, AV_LOG_ERROR, "Failed to tokenize batch text\n");
    return ret;
  }

  // Create tensors for tokens and attention mask
  std::vector<torch::Tensor> tokens_tensors;
  std::vector<torch::Tensor> attention_tensors;

  for (int i = 0; i < label_count; i++) {
    std::vector<int64_t> current_tokens;
    std::vector<int64_t> current_attention;

    current_tokens.reserve(CLXP_EMBEDDING_DIMS);
    current_attention.reserve(CLXP_EMBEDDING_DIMS);

    for (int j = 0; j < CLXP_EMBEDDING_DIMS; j++) {
      if (j < token_counts[i]) {
        current_tokens.push_back(static_cast<int64_t>(tokens_array[i][j]));
        current_attention.push_back(1); // Set all valid token positions to 1
      } else {
        current_tokens.push_back(0);    // Padding for tokens
        current_attention.push_back(0); // Padding for attention mask
      }
    }

    tokens_tensors.push_back(torch::tensor(current_tokens, torch::kInt64));
    attention_tensors.push_back(
        torch::tensor(current_attention, torch::kInt64));
  }

  // Stack all tensors into batches
  clip_ctx->tokenized_text = new torch::Tensor(torch::stack(tokens_tensors));
  clip_ctx->attention_mask = new torch::Tensor(torch::stack(attention_tensors));

  // Move tensors to the appropriate device
  if (clip_ctx->tokenized_text->device() != device) {
    *clip_ctx->tokenized_text = clip_ctx->tokenized_text->to(device);
  }
  if (clip_ctx->attention_mask->device() != device) {
    *clip_ctx->attention_mask = clip_ctx->attention_mask->to(device);
  }

  // Free allocated memory
  ff_dnn_tokenizer_free_batch(tokens_array, label_count);
  av_freep(&token_counts);
  ff_dnn_tokenizer_free(tokenizer);

  return 0;
}

torch::Tensor calculate_clip_similarity_matrix(const torch::Tensor &image_features,
                                 const torch::Tensor &text_embedding,
                                 DnnContext *ctx) {
  try {
    return torch::matmul(image_features, text_embedding.transpose(0, 1));
  } catch (const c10::Error &e) {
    av_log(ctx, AV_LOG_ERROR, "Similarity computation failed: %s\n", e.what());
    return {};
  }
}

int init_clip_model(THModel *th_model, DNNFunctionType func_type,
                    const char **labels, int label_count,
                    const char *tokenizer_path,
                    const AVFilterContext *filter_ctx) {
  int ret = 0;
  c10::Device device = (*th_model->jit_model->parameters().begin()).device();
  th_model->clip_ctx = (THClipContext *)av_mallocz(sizeof(THClipContext));
  if (!th_model->clip_ctx) {
    av_log(th_model->ctx, AV_LOG_ERROR,
           "Failed to allocate memory for CLIP context\n");
    return AVERROR(ENOMEM);
  }
  if (func_type == DFT_ANALYTICS_CLIP) {
    try {
      // Should throw exception if not existing
      auto encode_image = th_model->jit_model->get_method("encode_image");
      th_model->clip_ctx->resolution = get_clip_input_res(th_model, device);
      if (th_model->clip_ctx->resolution <= 0) {
        av_log(th_model->ctx, AV_LOG_ERROR,
               "Failed to determine input resolution for CLIP model\n");
        return AVERROR(EINVAL);
      }
    } catch (const c10::Error &e) {
      av_log(th_model->ctx, AV_LOG_ERROR,
             "Error during CLIP model initialization: %s\n", e.what());
      return AVERROR(EINVAL);
    }
  }
  return get_tokenized_batch(th_model->clip_ctx, labels, label_count,
                             tokenizer_path, th_model->ctx, device);
}

int prepare_audio_tensor(const THModel *th_model,
                         const THRequestItem *request) {
  THInferRequest *infer_request = request->infer_request;
  LastLevelTaskItem *lltask = request->lltasks[0];
  TaskItem *task = lltask->task;
  DnnContext *ctx = th_model->ctx;
  float *audio_data = NULL;
  int nb_samples = 0;

  int target_samples = CLAP_SAMPLE_RATE * 7;

  audio_data = (float *)task->in_frame->data[0];
  nb_samples = task->in_frame->nb_samples;

  // Calculate batch size dynamically based on available samples
  int batch_size = nb_samples / target_samples;

  // Check if we have enough samples for at least one batch
  if (batch_size < 1) {
    av_log(ctx, AV_LOG_ERROR,
           "Not enough samples for processing. Have %d, need at least %d\n",
           nb_samples, target_samples);
    return AVERROR(EINVAL);
  }

  av_log(ctx, AV_LOG_INFO, "Dynamically calculated batch size: %d\n",
         batch_size);

  // Check if frame already has the target sample rate
  if (task->in_frame->sample_rate == CLAP_SAMPLE_RATE &&
      task->in_frame->format == AV_SAMPLE_FMT_FLT) {
    // No resampling needed
  }

  try {
    // Create input tensor with batch dimension {batch_size, target_samples}
    *infer_request->input_tensor =
        torch::from_blob(audio_data, {batch_size, target_samples},
                         torch::kFloat32)
            .clone();
  } catch (const c10::Error &e) {
    av_log(ctx, AV_LOG_ERROR, "Audio encoding error: %s\n", e.what());
    return AVERROR(EINVAL);
  }

  return 0;
}

int preprocess_image_tensor(const THModel *th_model,
                            torch::Tensor *input_tensor,
                            const c10::Device &device) {
  DnnContext *ctx = th_model->ctx;
  try {
    if (input_tensor->device() != device) {
      *input_tensor = input_tensor->to(device);
    }
    *input_tensor = torch::nn::functional::interpolate(
        *input_tensor,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{th_model->clip_ctx->resolution,
                                       th_model->clip_ctx->resolution})
            .mode(torch::kBicubic)
            .align_corners(false));
    return 0;
  } catch (const c10::Error &e) {
    av_log(ctx, AV_LOG_ERROR, "Image encoding error: %s\n", e.what());
    return AVERROR(EINVAL);
  }
}

void free_clip_context(THClipContext *clip_ctx) {
  if (!clip_ctx)
    return;
  if (clip_ctx->tokenized_text) {
    delete clip_ctx->tokenized_text;
    clip_ctx->tokenized_text = nullptr;
  }
  if (clip_ctx->attention_mask) {
    delete clip_ctx->attention_mask;
    clip_ctx->attention_mask = nullptr;
  }
  av_freep(&clip_ctx);
}
#endif
