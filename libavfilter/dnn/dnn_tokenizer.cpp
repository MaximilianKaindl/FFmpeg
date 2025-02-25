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

#include "dnn_tokenizer.h"

extern "C"
{
#include "libavutil/mem.h"
#include "libavutil/log.h"
#include "libavformat/avio.h"
#include "libavutil/error.h"
}

// Define constants
const std::string START_TOKEN = "<|startoftext|>";
const std::string END_TOKEN = "<|endoftext|>";
const int32_t PADDING_TOKEN = 0;
const int DEFAULT_MAX_LENGTH = 77;

int load_bytes_from_file(const std::string &path, std::string &data, void *log_ctx)
{
    AVIOContext *avio_ctx = NULL;
    int ret;
    int64_t size;

    ret = avio_open(&avio_ctx, path.c_str(), AVIO_FLAG_READ);
    if (ret < 0)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Cannot open file: %s\n", path.c_str());
        }
        return ret;
    }

    size = avio_size(avio_ctx);
    if (size < 0)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to determine file size: %s\n", path.c_str());
        }
        avio_closep(&avio_ctx);
        return size;
    }

    try
    {
        data.resize(size);
        ret = avio_read(avio_ctx, (unsigned char *)data.data(), size);
        avio_closep(&avio_ctx);

        if (ret < 0)
        {
            if (log_ctx)
            {
                av_log(log_ctx, AV_LOG_ERROR, "Failed to read file: %s\n", path.c_str());
            }
            return ret;
        }
        if (ret != size)
        {
            if (log_ctx)
            {
                av_log(log_ctx, AV_LOG_ERROR, "Incomplete read: %s\n", path.c_str());
            }
            return AVERROR(EIO);
        }
    }
    catch (const std::exception &e)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Exception while reading file %s: %s\n",
                   path.c_str(), e.what());
        }
        avio_closep(&avio_ctx);
        return AVERROR(ENOMEM);
    }

    return 0;
}

std::unique_ptr<Tokenizer> create_tokenizer(const std::string &tokenizer_path, void *log_ctx)
{
    std::string blob;
    int ret = load_bytes_from_file(tokenizer_path, blob, log_ctx);
    if (ret < 0)
    {
        return nullptr;
    }

    try
    {
        return Tokenizer::FromBlobJSON(blob);
    }
    catch (const std::exception &e)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Error creating tokenizer: %s\n", e.what());
        }
        return nullptr;
    }
}

// Overload that accepts unique_ptr by reference
std::vector<int64_t> get_tokens(const std::unique_ptr<Tokenizer> &tokenizer,
                                const std::string &prompt,
                                int max_length,
                                void *log_ctx)
{
    return get_tokens(tokenizer.get(), prompt, max_length, log_ctx);
}

std::vector<int64_t> get_tokens(Tokenizer *tokenizer,
                                const std::string &prompt,
                                int max_length,
                                void *log_ctx)
{
    // Create vector with correct size, filled with padding tokens
    std::vector<int64_t> padded_ids(max_length, PADDING_TOKEN);

    if (!tokenizer)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Tokenizer is null\n");
        }
        return padded_ids;
    }

    try
    {
        // Check if special tokens exist in the vocabulary
        int32_t start_token = tokenizer->TokenToId(START_TOKEN);
        int32_t end_token = tokenizer->TokenToId(END_TOKEN);
        bool has_start_token = (start_token != -1);
        bool has_end_token = (end_token != -1);

        if (!has_start_token && log_ctx)
        {
            av_log(log_ctx, AV_LOG_INFO, "Start token <%s> not found in vocabulary\n", START_TOKEN.c_str());
        }

        if (!has_end_token && log_ctx)
        {
            av_log(log_ctx, AV_LOG_INFO, "End token <%s> not found in vocabulary\n", END_TOKEN.c_str());
        }

        // Get tokens from the tokenizer
        std::vector<int> tokens = tokenizer->Encode(prompt);

        // Calculate how many tokens we can include
        int position = 0;

        // Add start token if it exists
        if (has_start_token)
        {
            padded_ids[position++] = start_token;
        }

        // Copy tokens after the start token (if present)
        size_t remaining_space = has_end_token ? max_length - position - 1 : max_length - position;
        size_t tokens_to_copy = std::min(tokens.size(), remaining_space);

        if (tokens.size() > remaining_space && log_ctx)
        {
            av_log(log_ctx, AV_LOG_WARNING,
                   "Input text is too long, truncating to %zu tokens\n", remaining_space);
        }

        for (size_t i = 0; i < tokens_to_copy; i++)
        {
            padded_ids[position++] = tokens[i];
        }

        // Add end token if it exists
        if (has_end_token)
        {
            padded_ids[position++] = end_token;
        }

        // No need to fill the rest with padding tokens as we initialized the vector with them
    }
    catch (const std::exception &e)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Token generation failed: %s\n", e.what());
        }
        // Clear and fill with padding tokens on error
        std::fill(padded_ids.begin(), padded_ids.end(), PADDING_TOKEN);
    }

    return padded_ids;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> get_tokens_with_mask(
    Tokenizer *tokenizer,
    const std::string &prompt,
    int max_length,
    void *log_ctx)
{

    // Create vectors with correct size, filled with padding tokens and zeros for mask
    std::vector<int64_t> padded_ids(max_length, PADDING_TOKEN);
    std::vector<int64_t> attention_mask(max_length, 0);

    if (!tokenizer)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Tokenizer is null\n");
        }
        return {padded_ids, attention_mask};
    }

    try
    {
        // Check if special tokens exist in the vocabulary
        int32_t start_token = tokenizer->TokenToId(START_TOKEN);
        int32_t end_token = tokenizer->TokenToId(END_TOKEN);
        bool has_start_token = (start_token != -1);
        bool has_end_token = (end_token != -1);

        if (!has_start_token && log_ctx)
        {
            av_log(log_ctx, AV_LOG_INFO, "Start token <%s> not found in vocabulary\n", START_TOKEN.c_str());
        }

        if (!has_end_token && log_ctx)
        {
            av_log(log_ctx, AV_LOG_INFO, "End token <%s> not found in vocabulary\n", END_TOKEN.c_str());
        }

        // Get tokens from the tokenizer
        std::vector<int> tokens = tokenizer->Encode(prompt);

        // Calculate how many tokens we can include
        int position = 0;

        // Add start token if it exists
        if (has_start_token)
        {
            padded_ids[position] = start_token;
            attention_mask[position] = 1;
            position++;
        }

        // Copy tokens and set attention mask
        size_t remaining_space = has_end_token ? max_length - position - 1 : max_length - position;
        size_t tokens_to_copy = std::min(tokens.size(), remaining_space);

        if (tokens.size() > remaining_space && log_ctx)
        {
            av_log(log_ctx, AV_LOG_WARNING,
                   "Input text is too long, truncating to %zu tokens\n", remaining_space);
        }

        for (size_t i = 0; i < tokens_to_copy; i++)
        {
            padded_ids[position] = tokens[i];
            attention_mask[position] = 1;
            position++;
        }

        // Add end token if it exists
        if (has_end_token)
        {
            padded_ids[position] = end_token;
            attention_mask[position] = 1;
            position++;
        }

        // No need to set remaining positions as we initialized the vectors with default values
    }
    catch (const std::exception &e)
    {
        if (log_ctx)
        {
            av_log(log_ctx, AV_LOG_ERROR, "Token generation with mask failed: %s\n", e.what());
        }
        // Clear and fill with padding tokens and zeros on error
        std::fill(padded_ids.begin(), padded_ids.end(), PADDING_TOKEN);
        std::fill(attention_mask.begin(), attention_mask.end(), 0);
    }

    return {padded_ids, attention_mask};
}

// Overload that accepts unique_ptr by reference
std::pair<std::vector<int64_t>, std::vector<int64_t>> get_tokens_with_mask(
    const std::unique_ptr<Tokenizer> &tokenizer,
    const std::string &prompt,
    int max_length,
    void *log_ctx)
{
    return get_tokens_with_mask(tokenizer.get(), prompt, max_length, log_ctx);
}