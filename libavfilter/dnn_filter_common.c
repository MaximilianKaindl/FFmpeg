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

#include "dnn_filter_common.h"
#include "libavutil/avstring.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavformat/avio.h"

#if (CONFIG_LIBTOKENIZERS == 1)
#include "tokenizers_c.h"
#endif

#define MAX_SUPPORTED_OUTPUTS_NB 4

static char **separate_output_names(const char *expr, const char *val_sep, int *separated_nb)
{
    char *val, **parsed_vals = NULL;
    int val_num = 0;
    if (!expr || !val_sep || !separated_nb) {
        return NULL;
    }

    parsed_vals = av_calloc(MAX_SUPPORTED_OUTPUTS_NB, sizeof(*parsed_vals));
    if (!parsed_vals) {
        return NULL;
    }

    do {
        val = av_get_token(&expr, val_sep);
        if(val) {
            parsed_vals[val_num] = val;
            val_num++;
        }
        if (*expr) {
            expr++;
        }
    } while(*expr);

    parsed_vals[val_num] = NULL;
    *separated_nb = val_num;

    return parsed_vals;
}

typedef struct DnnFilterBase {
    const AVClass *class;
    DnnContext dnnctx;
} DnnFilterBase;

int ff_dnn_filter_init_child_class(AVFilterContext *filter) {
    DnnFilterBase *base = filter->priv;
    ff_dnn_init_child_class(&base->dnnctx);
    return 0;
}

void *ff_dnn_filter_child_next(void *obj, void *prev)
{
    DnnFilterBase *base = obj;
    return ff_dnn_child_next(&base->dnnctx, prev);
}

static int ff_dnn_init_priv(DnnContext *ctx, DNNFunctionType func_type, AVFilterContext *filter_ctx)
{
    DNNBackendType backend = ctx->backend_type;

    if (!ctx->model_filename) {
        av_log(filter_ctx, AV_LOG_ERROR, "model file for network is not specified\n");
        return AVERROR(EINVAL);
    }

    if (backend == DNN_TH) {
        if (ctx->model_inputname)
            av_log(filter_ctx, AV_LOG_WARNING, "LibTorch backend do not require inputname, "\
                                                "inputname will be ignored.\n");
        if (ctx->model_outputnames)
            av_log(filter_ctx, AV_LOG_WARNING, "LibTorch backend do not require outputname(s), "\
                                            "all outputname(s) will be ignored.\n");

#if (CONFIG_LIBTOKENIZERS == 0)
        if ((func_type == DFT_ANALYTICS_CLIP || func_type == DFT_ANALYTICS_CLAP)) {
            av_log(ctx, AV_LOG_ERROR,
                "tokenizers-cpp is not included. CLIP/CLAP Classification requires tokenizers-cpp library. Include it with configure.\n");
            return AVERROR(EINVAL);
        }
#endif
        ctx->nb_outputs = 1;
    } else if (backend == DNN_TF) {
        if (!ctx->model_inputname) {
            av_log(filter_ctx, AV_LOG_ERROR, "input name of the model network is not specified\n");
            return AVERROR(EINVAL);
        }
        ctx->model_outputnames = separate_output_names(ctx->model_outputnames_string, "&", &ctx->nb_outputs);
        if (!ctx->model_outputnames) {
            av_log(filter_ctx, AV_LOG_ERROR, "could not parse model output names\n");
            return AVERROR(EINVAL);
        }
    }

    ctx->dnn_module = ff_get_dnn_module(ctx->backend_type, filter_ctx);
    if (!ctx->dnn_module) {
        av_log(filter_ctx, AV_LOG_ERROR, "could not create DNN module for requested backend\n");
        return AVERROR(ENOMEM);
    }
    if (!ctx->dnn_module->load_model) {
        av_log(filter_ctx, AV_LOG_ERROR, "load_model for network is not specified\n");
        return AVERROR(EINVAL);
    }

    if (ctx->backend_options) {
        void *child = NULL;

        av_log(filter_ctx, AV_LOG_WARNING,
                "backend_configs is deprecated, please set backend options directly\n");
        while (child = ff_dnn_child_next(ctx, child)) {
            if (*(const AVClass **)child == &ctx->dnn_module->clazz) {
                int ret = av_opt_set_from_string(child, ctx->backend_options,
                                                NULL, "=", "&");
                if (ret < 0) {
                    av_log(filter_ctx, AV_LOG_ERROR, "failed to parse options \"%s\"\n",
                            ctx->backend_options);
                    return ret;
                }
            }
        }
    }
    return 0;
}

int ff_dnn_init(DnnContext *ctx, DNNFunctionType func_type, AVFilterContext *filter_ctx)
{
    int ret = ff_dnn_init_priv(ctx, func_type, filter_ctx);
    if (ret < 0) {
        return ret;
    }
    ctx->model = (ctx->dnn_module->load_model)(ctx, func_type, filter_ctx);
    if (!ctx->model) {
        av_log(filter_ctx, AV_LOG_ERROR, "could not load DNN model\n");
        return AVERROR(EINVAL);
    }
    return 0;
}

int ff_dnn_init_with_tokenizer(DnnContext *ctx, DNNFunctionType func_type, char **labels, int label_count,
                            int *softmax_units, int softmax_units_count, char *tokenizer_path, AVFilterContext *filter_ctx)
{
    int ret = ff_dnn_init_priv(ctx, func_type, filter_ctx);
    if (ret < 0) {
        return ret;
    }
    ctx->model = (ctx->dnn_module->load_model_with_tokenizer)(ctx, func_type, labels, label_count, softmax_units,
                                                            softmax_units_count, tokenizer_path, filter_ctx);
    if (!ctx->model) {
        av_log(filter_ctx, AV_LOG_ERROR, "could not load DNN model\n");
        return AVERROR(EINVAL);
    }
    return 0;
}

int ff_dnn_set_frame_proc(DnnContext *ctx, FramePrePostProc pre_proc, FramePrePostProc post_proc)
{
    ctx->model->frame_pre_proc = pre_proc;
    ctx->model->frame_post_proc = post_proc;
    return 0;
}

int ff_dnn_set_detect_post_proc(DnnContext *ctx, DetectPostProc post_proc)
{
    ctx->model->detect_post_proc = post_proc;
    return 0;
}

int ff_dnn_set_classify_post_proc(DnnContext *ctx, ClassifyPostProc post_proc)
{
    ctx->model->classify_post_proc = post_proc;
    return 0;
}

int ff_dnn_get_input(DnnContext *ctx, DNNData *input)
{
    return ctx->model->get_input(ctx->model, input, ctx->model_inputname);
}

int ff_dnn_get_output(DnnContext *ctx, int input_width, int input_height, int *output_width, int *output_height)
{
    char * output_name = ctx->model_outputnames && ctx->backend_type != DNN_TH ?
                        ctx->model_outputnames[0] : NULL;
    return ctx->model->get_output(ctx->model, ctx->model_inputname, input_width, input_height,
                                (const char *)output_name, output_width, output_height);
}

int ff_dnn_execute_model(DnnContext *ctx, AVFrame *in_frame, AVFrame *out_frame)
{
    DNNExecBaseParams exec_params = {
        .input_name     = ctx->model_inputname,
        .output_names   = (const char **)ctx->model_outputnames,
        .nb_output      = ctx->nb_outputs,
        .in_frame       = in_frame,
        .out_frame      = out_frame,
    };
    return (ctx->dnn_module->execute_model)(ctx->model, &exec_params);
}

int ff_dnn_execute_model_classification(DnnContext *ctx, AVFrame *in_frame, AVFrame *out_frame, const char *target)
{
    DNNExecClassificationParams class_params = {
        {
            .input_name     = ctx->model_inputname,
            .output_names   = (const char **)ctx->model_outputnames,
            .nb_output      = ctx->nb_outputs,
            .in_frame       = in_frame,
            .out_frame      = out_frame,
        },
        .target = target,
    };
    return (ctx->dnn_module->execute_model)(ctx->model, &class_params.base);
}

int ff_dnn_execute_model_clip(DnnContext *ctx, AVFrame *in_frame, AVFrame *out_frame, const char **labels, int label_count, const char* tokenizer_path, char *target)
{
    DNNExecZeroShotClassificationParams class_params = {
        {
            .input_name     = ctx->model_inputname,
            .output_names   = (const char **)ctx->model_outputnames,
            .nb_output      = ctx->nb_outputs,
            .in_frame       = in_frame,
            .out_frame      = out_frame,
        },
        .labels = labels,
        .label_count = label_count,
        .tokenizer_path = tokenizer_path,
        .target = target,
    };
    return (ctx->dnn_module->execute_model)(ctx->model, &class_params.base);
}

int ff_dnn_execute_model_clap(DnnContext *ctx, AVFrame *in_frame, AVFrame *out_frame, const char **labels, int label_count, const char* tokenizer_path)
{
    DNNExecZeroShotClassificationParams class_params = {
        {
            .input_name     = ctx->model_inputname,
            .output_names   = (const char **)ctx->model_outputnames,
            .nb_output      = ctx->nb_outputs,
            .in_frame       = in_frame,
            .out_frame      = out_frame,
        },
        .labels = labels,
        .label_count = label_count,
        .tokenizer_path = tokenizer_path,
    };
    return (ctx->dnn_module->execute_model)(ctx->model, &class_params.base);
}

DNNAsyncStatusType ff_dnn_get_result(DnnContext *ctx, AVFrame **in_frame, AVFrame **out_frame)
{
    return (ctx->dnn_module->get_result)(ctx->model, in_frame, out_frame);
}

int ff_dnn_flush(DnnContext *ctx)
{
    return (ctx->dnn_module->flush)(ctx->model);
}

void ff_dnn_uninit(DnnContext *ctx)
{
    if (ctx->dnn_module) {
        (ctx->dnn_module->free_model)(&ctx->model);
    }
    if (ctx->model_outputnames) {
        for (int i = 0; i < ctx->nb_outputs; i++)
            av_free(ctx->model_outputnames[i]);

        av_freep(&ctx->model_outputnames);
    }
}

static int load_file_content(const char *path, char **data, size_t *data_size, void *log_ctx) {
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
            av_log(log_ctx, AV_LOG_ERROR, "Failed to determine file size: %s\n", path);
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

#if (CONFIG_LIBTOKENIZERS == 1)
TokenizerHandle ff_dnn_tokenizer_create(const char *path, void *log_ctx)
{
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

int ff_dnn_tokenizer_encode_batch(TokenizerHandle tokenizer, const char **texts, int text_count,
                                TokenizerEncodeResult **results, void *log_ctx)
{
    size_t *lengths = NULL;
    int ret = 0;

    if (!tokenizer) {
        if (log_ctx)
            av_log(log_ctx, AV_LOG_ERROR, "Tokenizer is NULL\n");
        return AVERROR(EINVAL);
    }

    if (!texts || text_count <= 0 || !results) {
        if (log_ctx)
            av_log(log_ctx, AV_LOG_ERROR, "Invalid parameters\n");
        return AVERROR(EINVAL);
    }

    *results = av_calloc(text_count, sizeof(**results));
    if (!*results) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    lengths = av_calloc(text_count, sizeof(*lengths));
    if (!lengths) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    // Calculate text lengths
    for (int i = 0; i < text_count; i++) {
        lengths[i] = texts[i] ? strlen(texts[i]) : 0;
    }

    // Tokenize all texts in batch - directly store results in the output array
    tokenizers_encode_batch(tokenizer, texts, lengths, text_count, 1, *results);

    av_freep(&lengths);
    return 0;

fail:
    av_freep(results);
    av_freep(&lengths);
    return ret;
}

int ff_dnn_create_tokenizer_and_encode_batch(const char *path, const char **texts, int text_count,
                                            TokenizerEncodeResult **results, void *log_ctx)
{
    int ret;

    // Create tokenizer
    TokenizerHandle tokenizer = ff_dnn_tokenizer_create(path, log_ctx);
    if (!tokenizer) {
        av_log(log_ctx, AV_LOG_ERROR, "Error creating tokenizer\n");
        return AVERROR(EINVAL);
    }

    // Tokenize batch
    ret = ff_dnn_tokenizer_encode_batch(tokenizer, texts, text_count, results, log_ctx);

    if (ret < 0) {
        av_log(log_ctx, AV_LOG_ERROR, "Failed to tokenize batch text\n");
    }

    // Clean up tokenizer
    ff_dnn_tokenizer_free(tokenizer);
    return ret;
}
#endif