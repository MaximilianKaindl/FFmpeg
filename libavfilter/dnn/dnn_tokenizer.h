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

 #ifndef AVFILTER_DNN_DNN_TOKENIZER_H
 #define AVFILTER_DNN_DNN_TOKENIZER_H
 
 #include <string>
 #include <memory>
 #include <vector>
 #include <utility> // for std::pair
 #include <stdexcept> // Added for std::runtime_error

 extern "C" {
 #include "libavutil/log.h"
 }
 
 #include <tokenizers_cpp.h>
 
 using tokenizers::Tokenizer;
 
 // Tokenizer constants
 extern const std::string START_TOKEN;
 extern const std::string END_TOKEN;
 extern const int32_t PADDING_TOKEN;
 extern const int DEFAULT_MAX_LENGTH;
 
 /**
  * Load data from a file.
  * 
  * @param path The file path.
  * @param data Output parameter to store the file content.
  * @param log_ctx Context for logging.
  * @return 0 on success, error code on failure.
  */
 int load_bytes_from_file(const std::string& path, std::string& data, void* log_ctx);
 
 /**
  * Create a tokenizer from a tokenizer file.
  * 
  * @param tokenizer_path Path to the tokenizer file.
  * @param log_ctx Context for logging.
  * @return Unique pointer to the created tokenizer, or nullptr on failure.
  */
 std::unique_ptr<Tokenizer> create_tokenizer(const std::string& tokenizer_path, void* log_ctx);
 
 /**
  * Tokenize a text prompt using raw pointer.
  * 
  * @param tokenizer Pointer to the tokenizer.
  * @param prompt The text prompt to tokenize.
  * @param max_length Maximum length of token sequence (including special tokens).
  * @param log_ctx Context for logging.
  * @return Vector of token IDs.
  */
 std::vector<int64_t> get_tokens(Tokenizer* tokenizer, 
                                const std::string& prompt, 
                                int max_length = DEFAULT_MAX_LENGTH,
                                void* log_ctx = nullptr);
 
 /**
  * Tokenize a text prompt using unique_ptr.
  * 
  * @param tokenizer Reference to unique_ptr of the tokenizer.
  * @param prompt The text prompt to tokenize.
  * @param max_length Maximum length of token sequence (including special tokens).
  * @param log_ctx Context for logging.
  * @return Vector of token IDs.
  */
 std::vector<int64_t> get_tokens(const std::unique_ptr<Tokenizer>& tokenizer, 
                                const std::string& prompt, 
                                int max_length = DEFAULT_MAX_LENGTH,
                                void* log_ctx = nullptr);
 
 /**
  * Tokenize a text prompt and return attention mask using raw pointer.
  * 
  * @param tokenizer Pointer to the tokenizer.
  * @param prompt The text prompt to tokenize.
  * @param max_length Maximum length of token sequence (including special tokens).
  * @param log_ctx Context for logging.
  * @return Pair of vectors: first is token IDs, second is attention mask (1 for tokens, 0 for padding).
  */
 std::pair<std::vector<int64_t>, std::vector<int64_t>> get_tokens_with_mask(
    Tokenizer* tokenizer,
    const std::string& prompt,
    int max_length = DEFAULT_MAX_LENGTH,
    void* log_ctx = nullptr);
 
 /**
  * Tokenize a text prompt and return attention mask using unique_ptr.
  * 
  * @param tokenizer Reference to unique_ptr of the tokenizer.
  * @param prompt The text prompt to tokenize.
  * @param max_length Maximum length of token sequence (including special tokens).
  * @param log_ctx Context for logging.
  * @return Pair of vectors: first is token IDs, second is attention mask (1 for tokens, 0 for padding).
  */
 std::pair<std::vector<int64_t>, std::vector<int64_t>> get_tokens_with_mask(
    const std::unique_ptr<Tokenizer>& tokenizer,
    const std::string& prompt,
    int max_length = DEFAULT_MAX_LENGTH,
    void* log_ctx = nullptr);
 
 #endif // AVFILTER_DNN_DNN_TOKENIZER_H