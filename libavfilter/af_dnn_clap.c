/*
 * Copyright (c) 2024
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

/**
 * @file
 * CLAP (Contrastive Language-Audio Pretraining) filter
 */

#include "libavutil/opt.h"
#include "libavutil/avstring.h"
#include "avfilter.h"
#include "libavutil/mem.h"
#include "audio.h"
#include "formats.h"
#include "dnn_filter_common.h"
#include "libavutil/detection_bbox.h"
#include "libavutil/time.h"
#include "libavutil/file_open.h"
#include "libavutil/avstring.h"


typedef struct DNNCLAPContext
{
    const AVClass *class;
    DnnContext dnnctx;
    float confidence;
    char *labels_filename;
    char *tokenizer_path;
    char *target;
    char **labels;
    int label_count;
} DNNCLAPContext;

#define OFFSET(x) offsetof(DNNCLAPContext, dnnctx.x)
#define OFFSET2(x) offsetof(DNNCLAPContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_AUDIO_PARAM

static const AVOption dnn_clap_options[] = {
    { "dnn_backend", "DNN backend", 
        OFFSET(backend_type), AV_OPT_TYPE_INT, 
        { .i64 = DNN_TH }, INT_MIN, INT_MAX, FLAGS, .unit = "backend" },
#if (CONFIG_LIBTORCH == 1)
    { "torch", "torch backend flag", 
        0, AV_OPT_TYPE_CONST, { .i64 = DNN_TH }, 0, 0, FLAGS, .unit = "backend" },
#endif
    {"model", "path to model file", OFFSET(model_filename), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    {"confidence", "confidence threshold", OFFSET2(confidence), AV_OPT_TYPE_FLOAT, {.dbl = 0.5}, 0, 1, FLAGS},
    {"labels", "path to labels file", OFFSET2(labels_filename), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    {"tokenizer", "path to tokenizer file", OFFSET2(tokenizer_path), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    {NULL}};

AVFILTER_DNN_DEFINE_CLASS(dnn_clap, DNN_TH);

static int dnn_clap_post_proc(AVFrame *frame, DNNData *output, uint32_t bbox_index, AVFilterContext *filter_ctx)
{
    DNNCLAPContext *ctx = filter_ctx->priv;
    const int max_classes_per_box = AV_NUM_DETECTION_BBOX_CLASSIFY;
    int num_labels = ctx->label_count;
    float *probabilities = (float *)output->data;
    int num_bboxes;
    AVFrameSideData *sd;
    AVDetectionBBoxHeader *header;
    AVDetectionBBox *bbox;
    int i, j;
    int start_idx, end_idx;
    int percentage;

    // Calculate number of bounding boxes needed
    num_bboxes = (num_labels + max_classes_per_box - 1) / max_classes_per_box;

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DETECTION_BBOXES);
    if (sd != NULL)
    {
        av_log(filter_ctx, AV_LOG_ERROR, "Found existing Clip BBox. Box gets replaced ... \n");
        av_frame_remove_side_data(frame, AV_FRAME_DATA_DETECTION_BBOXES);
    }

    header = av_detection_bbox_create_side_data(frame, num_bboxes);
    if (!header)
    {
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to allocate side data for clip classification\n");
        return AVERROR(ENOMEM);
    }

    if (bbox_index == 0)
    {
        av_strlcat(header->source, ", ", sizeof(header->source));
        av_strlcat(header->source, ctx->dnnctx.model_filename, sizeof(header->source));
    }

    // Process each bbox
    for (i = 0; i < num_bboxes; i++)
    {
        bbox = av_get_detection_bbox(header, i);
        if (!bbox)
        {
            av_log(filter_ctx, AV_LOG_ERROR, "Failed to get bbox %d\n", i);
            return AVERROR(EINVAL);
        }

        // Initialize bbox
        bbox->classify_count = 0;

        start_idx = i * max_classes_per_box;
        end_idx = FFMIN(num_labels, (i + 1) * max_classes_per_box);

        // Set classifications for this bbox
        for (j = start_idx; j < end_idx && bbox->classify_count < max_classes_per_box; j++)
        {
            if (!ctx->labels[j])
            {
                av_log(filter_ctx, AV_LOG_ERROR, "Invalid label at index %d\n", j);
                continue;
            }

            percentage = (int)(probabilities[j] * 10000);
            bbox->classify_confidences[bbox->classify_count] = av_make_q(percentage, 10000);
            av_strlcpy(bbox->classify_labels[bbox->classify_count],
                       ctx->labels[j],
                       AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE);
            bbox->classify_count++;
        }
    }

    return 0;
}
static int dnn_clip_flush_frame(AVFilterLink *outlink, int64_t pts, int64_t *out_pts)
{
    DNNCLAPContext *ctx = outlink->src->priv;
    int ret;
    DNNAsyncStatusType async_state;

    ret = ff_dnn_flush(&ctx->dnnctx);
    if (ret != 0)
    {
        return -1;
    }

    do
    {
        AVFrame *in_frame = NULL;
        AVFrame *out_frame = NULL;
        async_state = ff_dnn_get_result(&ctx->dnnctx, &in_frame, &out_frame);
        if (async_state == DAST_SUCCESS)
        {
            ret = ff_filter_frame(outlink, in_frame);
            if (ret < 0)
                return ret;
            if (out_pts)
                *out_pts = in_frame->pts + pts;
        }
        av_usleep(5000);
    } while (async_state >= DAST_NOT_READY);

    return 0;
}
static int dnn_clap_activate(AVFilterContext *filter_ctx)
{
    AVFilterLink *inlink = filter_ctx->inputs[0];
    AVFilterLink *outlink = filter_ctx->outputs[0];
    DNNCLAPContext *ctx = filter_ctx->priv;
    AVFrame *in = NULL;
    int64_t pts;
    int ret, status;
    int got_frame = 0;
    int async_state;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    do
    {
        // Process all available input frames
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
        if (ret > 0)
        {
            if (ff_dnn_execute_model_clip(&ctx->dnnctx, in, NULL, ctx->labels,  ctx->label_count, ctx->tokenizer_path, NULL) != 0)
            {
                return AVERROR(EIO);
            }
        }
    } while (ret > 0);

    // Handle processed frames
    do
    {
        AVFrame *in_frame = NULL;
        AVFrame *out_frame = NULL;
        async_state = ff_dnn_get_result(&ctx->dnnctx, &in_frame, &out_frame);
        if (async_state == DAST_SUCCESS)
        {
            ret = ff_filter_frame(outlink, in_frame);
            if (ret < 0)
                return ret;
            got_frame = 1;
        }
    } while (async_state == DAST_SUCCESS);

    // Schedule next filter if frame was processed
    if (got_frame)
        return 0;

    if (ff_inlink_acknowledge_status(inlink, &status, &pts))
    {
        if (status == AVERROR_EOF)
        {
            int64_t out_pts = pts;
            ret = dnn_clip_flush_frame(outlink, pts, &out_pts);
            ff_outlink_set_status(outlink, status, out_pts);
            return ret;
        }
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return 0;
}

static int read_classify_label_file(AVFilterContext *context)
{
    int line_len;
    FILE *file;
    DNNCLAPContext *ctx = context->priv;

    file = avpriv_fopen_utf8(ctx->labels_filename, "r");
    if (!file)
    {
        av_log(context, AV_LOG_ERROR, "Failed to open file %s\n", ctx->labels_filename);
        return AVERROR(EINVAL);
    }

    while (!feof(file))
    {
        char *prompt;
        char buf[256];
        if (!fgets(buf, sizeof(buf), file))
            break;

        line_len = strlen(buf);
        while (line_len)
        {
            int i = line_len - 1;
            if (buf[i] == '\n' || buf[i] == '\r' || buf[i] == ' ')
            {
                buf[i] = '\0';
                line_len--;
            }
            else
                break;
        }

        if (line_len == 0)
            continue;

        if (line_len > AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE)
        {
            av_log(context, AV_LOG_ERROR, "Text prompt %s too long\n", buf);
            fclose(file);
            return AVERROR(EINVAL);
        }

        prompt = av_strdup(buf);
        if (!prompt)
        {
            av_log(context, AV_LOG_ERROR, "Failed to allocate memory for prompt %s\n", buf);
            fclose(file);
            return AVERROR(ENOMEM);
        }

        if (av_dynarray_add_nofree(&ctx->labels, &ctx->label_count, prompt) < 0)
        {
            av_log(context, AV_LOG_ERROR, "Failed to add prompt to array\n");
            fclose(file);
            av_freep(&prompt);
            return AVERROR(ENOMEM);
        }
    }

    fclose(file);
    return 0;
}

static void free_classify_labels(DNNCLAPContext *ctx)
{
    for (int i = 0; i < ctx->label_count; i++)
        av_freep(&ctx->labels[i]);
    ctx->label_count = 0;
    av_freep(&ctx->labels);
}

static av_cold int dnn_clap_init(AVFilterContext *context)
{
    DNNCLAPContext *ctx = context->priv;
    int ret;

    ret = ff_dnn_init(&ctx->dnnctx, DFT_ANALYTICS_CLAP, context);
    if (ret < 0)
        return ret;
    ff_dnn_set_classify_post_proc(&ctx->dnnctx, dnn_clap_post_proc);
    if (!ctx->labels_filename)
    {
        av_log(context, AV_LOG_ERROR, "Text prompts file is required for CLIP classification\n");
        return AVERROR(EINVAL);
    }
    if (!ctx->tokenizer_path)
    {
        av_log(context, AV_LOG_ERROR, "Tokenizer file is required\n");
        return AVERROR(EINVAL);
    }

    return read_classify_label_file(context);
}

static void dnn_clap_uninit(AVFilterContext *context)
{
    DNNCLAPContext *ctx = context->priv;

    ff_dnn_uninit(&ctx->dnnctx);

    free_classify_labels(ctx);
}

const FFFilter ff_af_dnn_clap = {
    .p.name = "dnn_clap",
    .p.description = NULL_IF_CONFIG_SMALL("Apply CLAP (Contrastive Language-Audio Pretraining) filter."),
    .p.priv_class = &dnn_clap_class,
    .priv_size = sizeof(DNNCLAPContext),
    .preinit       = ff_dnn_filter_init_child_class,
    .init = dnn_clap_init,
    .uninit = dnn_clap_uninit,
    .activate = dnn_clap_activate,
    FILTER_INPUTS(ff_audio_default_filterpad),
    FILTER_OUTPUTS(ff_audio_default_filterpad),
    FILTER_SAMPLEFMTS_ARRAY(ff_all_formats),
};