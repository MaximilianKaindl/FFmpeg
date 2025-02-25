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
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 /**
  * Load bytes from a file into memory.
  * 
  * @param path      The file path.
  * @param data      Output parameter for the loaded data.
  * @param data_size Output parameter for the data size.
  * @param log_ctx   Context for logging.
  * @return 0 on success, error code on failure.
  */
 int load_bytes_from_file(const char *path, char **data, size_t *data_size, void *log_ctx);
 
 /**
  * Create a tokenizer from a tokenizer file.
  * 
  * @param tokenizer_path Path to the tokenizer file.
  * @param log_ctx        Context for logging.
  * @return Handle to the created tokenizer, or NULL on failure.
  */
 TokenizerHandle create_tokenizer(const char *tokenizer_path, void *log_ctx);
 
 /**
  * Tokenize a text prompt with special tokens automatically handled by the tokenizer.
  * The function allocates memory for the token IDs which should be freed by the caller.
  * 
  * @param tokenizer  Pointer to the tokenizer.
  * @param prompt     The text prompt to tokenize.
  * @param token_ids  Output parameter for the dynamically allocated token ID array.
  * @param n_tokens   Output parameter for the number of tokens.
  * @param log_ctx    Context for logging.
  * @return 0 on success, error code on failure.
  */
 int tokenize_text(TokenizerHandle tokenizer, const char *prompt, int target_length,
                  int **token_ids, int *n_tokens, void *log_ctx);
 
 /**
  * Tokenize a text prompt and return attention mask along with tokens.
  * Special tokens are handled automatically by the tokenizer.
  * The function allocates memory for both token IDs and attention mask 
  * which should be freed by the caller.
  * 
  * @param tokenizer      Pointer to the tokenizer.
  * @param prompt         The text prompt to tokenize.
  * @param token_ids      Output parameter for the dynamically allocated token ID array.
  * @param attention_mask Output parameter for the dynamically allocated attention mask array.
  * @param n_tokens       Output parameter for the number of tokens.
  * @param log_ctx        Context for logging.
  * @return 0 on success, error code on failure.
  */
 int tokenize_text_with_mask(TokenizerHandle tokenizer, const char *prompt, int target_length,
                            int **token_ids, int **attention_mask, 
                            int *n_tokens, void *log_ctx);
 
 /**
  * Free tokenizer resources.
  * 
  * @param tokenizer Handle to the tokenizer.
  */
 void free_tokenizer(TokenizerHandle tokenizer);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // AVFILTER_DNN_TOKENIZER_H