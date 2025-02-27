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

/**
 * @file
 * C interface for tokenizer handling using the tokenizers_c API
 */

#include "dnn_tokenizer.h"

#include "libavformat/avio.h"
#include "libavutil/error.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include <string.h>

static int load_file_content(const char *path, char **data, size_t *data_size,
                             void *log_ctx) {
  AVIOContext *avio_ctx = NULL;
  int ret;
  int64_t size;

  ret = avio_open(&avio_ctx, path, AVIO_FLAG_READ);
  if (ret < 0) {
    if (log_ctx)
      av_log(log_ctx, AV_LOG_ERROR, "Cannot open file: %s\n", path);
    return ret;
  }

  size = avio_size(avio_ctx);
  if (size < 0) {
    if (log_ctx)
      av_log(log_ctx, AV_LOG_ERROR, "Failed to determine file size: %s\n",
             path);
    avio_closep(&avio_ctx);
    return size;
  }

  *data = av_malloc(size + 1);
  if (!*data) {
    avio_closep(&avio_ctx);
    return AVERROR(ENOMEM);
  }

  ret = avio_read(avio_ctx, (unsigned char *)*data, size);
  avio_closep(&avio_ctx);

  if (ret < 0) {
    if (log_ctx)
      av_log(log_ctx, AV_LOG_ERROR, "Failed to read file: %s\n", path);
    av_freep(data);
    return ret;
  }

  if (ret != size) {
    if (log_ctx)
      av_log(log_ctx, AV_LOG_ERROR, "Incomplete read: %s\n", path);
    av_freep(data);
    return AVERROR(EIO);
  }

  // Null-terminate the data
  (*data)[size] = '\0';
  *data_size = size;

  return 0;
}

TokenizerHandle ff_dnn_tokenizer_create(const char *path, void *log_ctx) {
  char *blob = NULL;
  size_t blob_size = 0;
  TokenizerHandle handle = NULL;
  int ret;

  if (!path) {
    if (log_ctx)
      av_log(log_ctx, AV_LOG_ERROR, "Tokenizer path is NULL\n");
    return NULL;
  }

  ret = load_file_content(path, &blob, &blob_size, log_ctx);
  if (ret < 0)
    return NULL;

  handle = tokenizers_new_from_str(blob, blob_size);
  av_freep(&blob);

  if (!handle && log_ctx)
    av_log(log_ctx, AV_LOG_ERROR, "Error creating tokenizer\n");

  return handle;
}

int ff_dnn_tokenizer_encode(TokenizerHandle tokenizer, const char *text,
                            int **token_ids, int *token_count, void *log_ctx) {
  TokenizerEncodeResult result = {0};
  int i;

  if (!tokenizer) {
    if (log_ctx)
      av_log(log_ctx, AV_LOG_ERROR, "Tokenizer is NULL\n");
    return AVERROR(EINVAL);
  }

  if (!text || !token_ids || !token_count) {
    if (log_ctx)
      av_log(log_ctx, AV_LOG_ERROR, "Invalid parameters\n");
    return AVERROR(EINVAL);
  }

  // Tokenize with special tokens
  tokenizers_encode(tokenizer, text, strlen(text), 1, &result);

  if (!result.token_ids || result.len == 0) {
    if (log_ctx)
      av_log(log_ctx, AV_LOG_WARNING, "No tokens generated for text\n");
    tokenizers_free_encode_results(&result, 1);
    *token_count = 0;
    *token_ids = NULL;
    return 0;
  }

  *token_count = result.len;

  // Allocate memory for token IDs
  *token_ids = av_malloc(result.len * sizeof(int));
  if (!*token_ids) {
    tokenizers_free_encode_results(&result, 1);
    return AVERROR(ENOMEM);
  }

  // Copy all tokens
  for (i = 0; i < result.len; i++) {
    (*token_ids)[i] = result.token_ids[i];
  }

  tokenizers_free_encode_results(&result, 1);
  return 0;
}

int ff_dnn_create_tokenizer_and_encode(const char *path, const char *text, int **token_ids,
                            int *token_count, void *log_ctx) {
    TokenizerHandle tokenizer = NULL;
    int ret;
    tokenizer = ff_dnn_tokenizer_create(path, log_ctx);
    if (!tokenizer) {
        av_log(log_ctx, AV_LOG_ERROR, "Error creating tokenizer\n");
        return AVERROR(EINVAL);
    }

    // Tokenize batch
    ret = ff_dnn_tokenizer_encode(tokenizer, text, token_ids, token_count, log_ctx);

    if (ret < 0) {
        av_log(log_ctx, AV_LOG_ERROR, "Failed to tokenize text\n");
        ff_dnn_tokenizer_free(tokenizer);
        return ret;
    }

    // Later cleanup
    ff_dnn_tokenizer_free(tokenizer);
    return 0;
}

int ff_dnn_tokenizer_encode_batch(TokenizerHandle tokenizer, const char **texts,
                                  int text_count, int ***token_ids,
                                  int **token_counts, void *log_ctx) {
  size_t *lengths = NULL;
  TokenizerEncodeResult *results = NULL;
  int i, j, ret = 0;

  if (!tokenizer) {
    if (log_ctx)
      av_log(log_ctx, AV_LOG_ERROR, "Tokenizer is NULL\n");
    return AVERROR(EINVAL);
  }

  if (!texts || text_count <= 0 || !token_ids || !token_counts) {
    if (log_ctx)
      av_log(log_ctx, AV_LOG_ERROR, "Invalid parameters\n");
    return AVERROR(EINVAL);
  }

  // Allocate arrays for the results
  *token_ids = av_calloc(text_count, sizeof(**token_ids));
  if (!*token_ids) {
    ret = AVERROR(ENOMEM);
    goto fail;
  }

  *token_counts = av_calloc(text_count, sizeof(**token_counts));
  if (!*token_counts) {
    ret = AVERROR(ENOMEM);
    goto fail;
  }

  results = av_calloc(text_count, sizeof(*results));
  if (!results) {
    ret = AVERROR(ENOMEM);
    goto fail;
  }

  lengths = av_calloc(text_count, sizeof(*lengths));
  if (!lengths) {
    ret = AVERROR(ENOMEM);
    goto fail;
  }

  // Calculate text lengths
  for (i = 0; i < text_count; i++) {
    lengths[i] = texts[i] ? strlen(texts[i]) : 0;
  }

  // Tokenize all texts in batch
  tokenizers_encode_batch(tokenizer, texts, lengths, text_count, 1, results);

  // Process results
  for (i = 0; i < text_count; i++) {
    (*token_counts)[i] = results[i].len;

    // Allocate memory for this text's tokens
    (*token_ids)[i] = av_malloc(results[i].len * sizeof(int));
    if (!(*token_ids)[i]) {
      ret = AVERROR(ENOMEM);
      goto fail;
    }

    // Copy all tokens
    for (j = 0; j < results[i].len; j++) {
      (*token_ids)[i][j] = results[i].token_ids[j];
    }
  }

  // Clean up
  tokenizers_free_encode_results(results, text_count);
  av_freep(&results);
  av_freep(&lengths);

  return 0;

fail:
  if (*token_ids) {
    for (i = 0; i < text_count; i++) {
      if ((*token_ids)[i])
        av_freep(&(*token_ids)[i]);
    }
    av_freep(token_ids);
  }
  av_freep(token_counts);
  if (results)
    tokenizers_free_encode_results(results, text_count);
  av_freep(&results);
  av_freep(&lengths);

  return ret;
}

int ff_dnn_create_tokenizer_and_encode_batch(const char *path, const char **texts, int text_count,
    int ***token_ids, int **token_counts, void *log_ctx){
    int ret;

    // Create tokenizer
    TokenizerHandle tokenizer = ff_dnn_tokenizer_create(path, log_ctx);
    if (!tokenizer) {
        av_log(log_ctx, AV_LOG_ERROR, "Error creating tokenizer\n");
        return AVERROR(EINVAL);
    }

    // Tokenize batch
    ret = ff_dnn_tokenizer_encode_batch(tokenizer, texts, text_count, 
        token_ids, token_counts, log_ctx);

    if (ret < 0) {
        av_log(log_ctx, AV_LOG_ERROR, "Failed to tokenize batch text\n");
        ff_dnn_tokenizer_free(tokenizer);
        return ret;
    }

    // Later cleanup
    ff_dnn_tokenizer_free(tokenizer);
    return 0;
}

void ff_dnn_tokenizer_free_batch(int **token_ids, int token_count) {
  int i;

  if (!token_ids)
    return;

  for (i = 0; i < token_count; i++) {
    av_freep(&token_ids[i]);
  }
  av_freep(&token_ids);
}

void ff_dnn_tokenizer_free(TokenizerHandle tokenizer) {
  if (tokenizer)
    tokenizers_free(tokenizer);
}