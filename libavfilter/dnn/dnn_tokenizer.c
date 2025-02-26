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

#include "libavutil/mem.h"
#include "libavutil/log.h"
#include "libavformat/avio.h"
#include "libavutil/error.h"
#include "string.h"

int load_bytes_from_file(const char *path, char **data, size_t *data_size, void *log_ctx)
{
    AVIOContext *avio_ctx = NULL;
    int ret;
    int64_t size;

    ret = avio_open(&avio_ctx, path, AVIO_FLAG_READ);
    if (ret < 0)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Cannot open file: %s\n", path);
        }
        return ret;
    }

    size = avio_size(avio_ctx);
    if (size < 0)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to determine file size: %s\n", path);
        }
        avio_closep(&avio_ctx);
        return size;
    }

    *data = av_malloc(size + 1);
    if (!*data)
    {
        avio_closep(&avio_ctx);
        return AVERROR(ENOMEM);
    }

    ret = avio_read(avio_ctx, (unsigned char *)*data, size);
    avio_closep(&avio_ctx);

    if (ret < 0)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to read file: %s\n", path);
        }
        av_freep(data);
        return ret;
    }

    if (ret != size)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Incomplete read: %s\n", path);
        }
        av_freep(data);
        return AVERROR(EIO);
    }

    // Null-terminate the data
    (*data)[size] = '\0';
    *data_size = size;

    return 0;
}

TokenizerHandle create_tokenizer(const char *tokenizer_path, void *log_ctx)
{
    char *blob = NULL;
    size_t blob_size = 0;
    int ret = load_bytes_from_file(tokenizer_path, &blob, &blob_size, log_ctx);

    if (ret < 0)
    {
        return NULL;
    }

    TokenizerHandle handle = tokenizers_new_from_str(blob, blob_size);
    av_freep(&blob);

    if (!handle && log_ctx)
    {
        av_log(log_ctx, AV_LOG_ERROR, "Error creating tokenizer\n");
    }

    return handle;
}

int tokenize_text(TokenizerHandle tokenizer, const char *prompt, int target_length,
                  int **token_ids, int *n_tokens, void *log_ctx)
{
    if (!tokenizer)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Tokenizer is null\n");
        }
        return AVERROR(EINVAL);
    }

    // Encode the prompt with special tokens
    TokenizerEncodeResult result;
    memset(&result, 0, sizeof(result));
    tokenizers_encode(tokenizer, prompt, strlen(prompt), 1, &result);

    if (!result.token_ids || result.len == 0)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_WARNING, "No tokens generated for prompt\n");
        }
        tokenizers_free_encode_results(&result, 1);
        return 0;
    }

    *n_tokens = result.len;
    // Allocate memory for the token IDs
    *token_ids = av_malloc(target_length * sizeof(int));
    if (!*token_ids)
    {
        tokenizers_free_encode_results(&result, 1);
        return AVERROR(ENOMEM);
    }

    // Copy tokens and pad with zeros (or appropriate padding token)
    int i;
    for (i = 0; i < result.len && i < target_length; i++)
    {
        (*token_ids)[i] = result.token_ids[i];
    }

    // Fill remaining positions with padding token (typically 0)
    for (; i < target_length; i++)
    {
        (*token_ids)[i] = 0; // Use appropriate padding token
    }

    // Free the tokenizer result
    tokenizers_free_encode_results(&result, 1);

    return 0;
}

/**
 * Tokenize multiple text prompts in batch mode.
 * The function allocates memory for the token IDs which should be freed by the caller.
 *
 * @param tokenizer     Pointer to the tokenizer.
 * @param prompts       Array of text prompts to tokenize.
 * @param num_prompts   Number of prompts in the array.
 * @param target_length Maximum length for each tokenized sequence.
 * @param token_ids     Output parameter for the dynamically allocated token ID arrays.
 * @param n_tokens      Output parameter for the number of tokens for each prompt.
 * @param log_ctx       Context for logging.
 * @return 0 on success, error code on failure.
 */
int tokenize_text_batch(TokenizerHandle tokenizer, const char **prompts, int num_prompts,
                        int target_length, int ***token_ids, int **n_tokens, void *log_ctx)
{
    if (!tokenizer)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Tokenizer is null\n");
        }
        return AVERROR(EINVAL);
    }

    if (num_prompts <= 0)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Invalid number of prompts\n");
        }
        return AVERROR(EINVAL);
    }

    // Allocate arrays for results
    *token_ids = av_calloc(num_prompts, sizeof(int *));
    if (!*token_ids)
    {
        return AVERROR(ENOMEM);
    }

    *n_tokens = av_calloc(num_prompts, sizeof(int));
    if (!*n_tokens)
    {
        av_freep(token_ids);
        return AVERROR(ENOMEM);
    }

    // Prepare for batch encoding
    TokenizerEncodeResult *results = av_calloc(num_prompts, sizeof(TokenizerEncodeResult));
    if (!results)
    {
        av_freep(n_tokens);
        av_freep(token_ids);
        return AVERROR(ENOMEM);
    }

    // Prepare length array
    size_t *lengths = av_calloc(num_prompts, sizeof(size_t));
    if (!lengths)
    {
        av_freep(results);
        av_freep(n_tokens);
        av_freep(token_ids);
        return AVERROR(ENOMEM);
    }

    // Fill lengths array
    for (int i = 0; i < num_prompts; i++)
    {
        lengths[i] = strlen(prompts[i]);
    }

    // Encode batch with special tokens
    tokenizers_encode_batch(tokenizer, prompts, lengths, num_prompts, 1, results);

    // Process results
    for (int i = 0; i < num_prompts; i++)
    {
        (*n_tokens)[i] = results[i].len;

        // Allocate memory for the token IDs
        (*token_ids)[i] = av_malloc(target_length * sizeof(int));
        if (!(*token_ids)[i])
        {
            // Clean up already allocated resources
            for (int j = 0; j < i; j++)
            {
                av_freep(&(*token_ids)[j]);
            }
            tokenizers_free_encode_results(results, num_prompts);
            av_freep(&lengths);
            av_freep(n_tokens);
            av_freep(token_ids);
            return AVERROR(ENOMEM);
        }

        // Copy tokens and pad with zeros (or appropriate padding token)
        int j;
        for (j = 0; j < results[i].len && j < target_length; j++)
        {
            (*token_ids)[i][j] = results[i].token_ids[j];
        }

        // Fill remaining positions with padding token (typically 0)
        for (; j < target_length; j++)
        {
            (*token_ids)[i][j] = 0; // Use appropriate padding token
        }
    }

    // Free resources
    tokenizers_free_encode_results(results, num_prompts);
    av_freep(&lengths);

    return 0;
}

/**
 * Free resources allocated by tokenize_text_batch.
 *
 * @param token_ids    Array of token ID arrays to free.
 * @param num_prompts  Number of prompts/token arrays.
 */
void free_batch_tokens(int **token_ids, int num_prompts)
{
    if (!token_ids)
        return;

    for (int i = 0; i < num_prompts; i++)
    {
        av_freep(&token_ids[i]);
    }
    av_freep(&token_ids);
}

void free_tokenizer(TokenizerHandle tokenizer)
{
    if (tokenizer)
    {
        tokenizers_free(tokenizer);
    }
}