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

 #ifndef AVFILTER_DNN_DNN_TOKENIZER_H
 #define AVFILTER_DNN_DNN_TOKENIZER_H
 
 #include <stddef.h>
 #include <stdint.h>
 #include "tokenizers_c.h"
 
 /**
  * Create a tokenizer from a file path.
  *
  * @param path    Path to the tokenizer file.
  * @param log_ctx Context for logging.
  * @return Handle to the created tokenizer, or NULL on failure.
  */
 TokenizerHandle ff_dnn_tokenizer_create(const char *path, void *log_ctx);
 
 /**
  * Tokenize text with special tokens.
  *
  * @param tokenizer     Handle to the tokenizer.
  * @param text          Text to tokenize.
  * @param token_ids     Output parameter for token IDs (caller must free with av_free).
  * @param token_count   Output parameter for the number of tokens generated.
  * @param log_ctx       Context for logging.
  * @return 0 on success, negative error code on failure.
  */
 int ff_dnn_tokenizer_encode(TokenizerHandle tokenizer, const char *text,
                             int **token_ids, int *token_count, void *log_ctx);

 /**
  * Tokenize text with special tokens.
  *
  * @param path          Path to the tokenizer file.
  * @param text          Text to tokenize.
  * @param token_ids     Output parameter for token IDs (caller must free with av_free).
  * @param token_count   Output parameter for the number of tokens generated.
  * @param log_ctx       Context for logging.
  * @return 0 on success, negative error code on failure.
  */
 int ff_dnn_create_tokenizer_and_encode(const char *path, const char *text,
    int **token_ids, int *token_count, void *log_ctx);

 /**
  * Tokenize multiple texts in batch mode with special tokens.
  *
  * @param tokenizer     Handle to the tokenizer.
  * @param texts         Array of texts to tokenize.
  * @param text_count    Number of texts in the array.
  * @param token_ids     Output parameter for token IDs array (caller must free with ff_dnn_tokenizer_free_batch).
  * @param token_counts  Output parameter for the number of tokens for each text (caller must free with av_free).
  * @param log_ctx       Context for logging.
  * @return 0 on success, negative error code on failure.
  */
 int ff_dnn_tokenizer_encode_batch(TokenizerHandle tokenizer, const char **texts, int text_count,
                                   int ***token_ids, int **token_counts, void *log_ctx);
 

 /**
  * Tokenize multiple texts in batch mode with special tokens.
  *
  * @param path          Path to the tokenizer file.
  * @param texts         Array of texts to tokenize.
  * @param text_count    Number of texts in the array.
  * @param token_ids     Output parameter for token IDs array (caller must free with ff_dnn_tokenizer_free_batch).
  * @param token_counts  Output parameter for the number of tokens for each text (caller must free with av_free).
  * @param log_ctx       Context for logging.
  * @return 0 on success, negative error code on failure.
  */
 int ff_dnn_create_tokenizer_and_encode_batch(const char *path, const char **texts, int text_count,
    int ***token_ids, int **token_counts, void *log_ctx);

 /**
  * Free resources allocated by ff_dnn_tokenizer_encode_batch.
  *
  * @param token_ids    Array of token ID arrays to free.
  * @param token_count  Number of token arrays.
  */
 void ff_dnn_tokenizer_free_batch(int **token_ids, int token_count);
 
 /**
  * Free tokenizer resources.
  *
  * @param tokenizer Handle to the tokenizer.
  */
 void ff_dnn_tokenizer_free(TokenizerHandle tokenizer);
 
 #endif /* AVFILTER_DNN_TOKENIZER_H */