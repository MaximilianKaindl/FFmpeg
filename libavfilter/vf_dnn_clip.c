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
 * DNN CLIP filter - Zero-shot image classification using CLIP models
 */

#include "libavutil/file_open.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "filters.h"
#include "dnn_filter_common.h"
#include "video.h"
#include "libavutil/time.h"
#include "libavutil/avstring.h"
#include "libavutil/detection_bbox.h"
#include "libavfilter/dnn_interface.h"

typedef struct CLIPContext {
    const AVClass *clazz;
    DnnContext dnnctx;           /* Base DNN context */
    char *labels_filename;       /* Path to text prompts file */
    char *tokenizer_path;       /* Path to text prompts file */
    char **labels;              /* Array of text prompts */
    int label_count;            /* Number of text prompts */
} CLIPContext;

#define OFFSET(x) offsetof(CLIPContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM

static const AVOption dnn_clip_options[] = {
    { "dnn_backend", "DNN backend", 
        OFFSET(dnnctx.backend_type), AV_OPT_TYPE_INT, 
        { .i64 = DNN_TH }, INT_MIN, INT_MAX, FLAGS, .unit = "backend" },
#if (CONFIG_LIBTORCH == 1)
    { "torch", "torch backend flag", 
        0, AV_OPT_TYPE_CONST, { .i64 = DNN_TH }, 0, 0, FLAGS, .unit = "backend" },
#endif
    { "labels", "path to text prompts file", 
        OFFSET(labels_filename), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
    { "tokenizer", "path to text tokenizer.json file", 
        OFFSET(tokenizer_path), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
    { NULL }
};

AVFILTER_DNN_DEFINE_CLASS(dnn_clip, DNN_TH);

/**
 * Free allocated text prompts.
 */
static void free_classify_labels(CLIPContext *ctx)
{
    for (int i = 0; i < ctx->label_count; i++)
        av_freep(&ctx->labels[i]);
    ctx->label_count = 0;
    av_freep(&ctx->labels);
}

/**
 * Read text prompts from file.
 * Each line contains one prompt for zero-shot classification.
 */
static int read_classify_label_file(AVFilterContext *context)
{
    int line_len;
    FILE *file;
    CLIPContext *ctx = context->priv;

    file = avpriv_fopen_utf8(ctx->labels_filename, "r");
    if (!file) {
        av_log(context, AV_LOG_ERROR, "Failed to open file %s\n", ctx->labels_filename);
        return AVERROR(EINVAL);
    }

    while (!feof(file)) {
        char *prompt;
        char buf[256];
        if (!fgets(buf, sizeof(buf), file))
            break;

        line_len = strlen(buf);
        while (line_len) {
            int i = line_len - 1;
            if (buf[i] == '\n' || buf[i] == '\r' || buf[i] == ' ') {
                buf[i] = '\0';
                line_len--;
            } else
                break;
        }

        if (line_len == 0)
            continue;

        if (line_len >= AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE) {
            av_log(context, AV_LOG_ERROR, "Text prompt %s too long\n", buf);
            fclose(file);
            return AVERROR(EINVAL);
        }

        prompt = av_strdup(buf);
        if (!prompt) {
            av_log(context, AV_LOG_ERROR, "Failed to allocate memory for prompt %s\n", buf);
            fclose(file);
            return AVERROR(ENOMEM);
        }

        if (av_dynarray_add_nofree(&ctx->labels, &ctx->label_count, prompt) < 0) {
            av_log(context, AV_LOG_ERROR, "Failed to add prompt to array\n");
            fclose(file);
            av_freep(&prompt);
            return AVERROR(ENOMEM);
        }
    }

    fclose(file);
    return 0;
}

/**
 * Initialize the filter.
 */
static av_cold int dnn_clip_init(AVFilterContext *context)
{
    CLIPContext *ctx = context->priv;
    int ret;

    ret = ff_dnn_init(&ctx->dnnctx, DFT_ANALYTICS_ZEROSHOTCLASSIFY, context);
    if (ret < 0)
        return ret;

    if (!ctx->labels_filename) {
        av_log(context, AV_LOG_ERROR, "Text prompts file is required for CLIP classification\n");
        return AVERROR(EINVAL);
    }

    return read_classify_label_file(context);
}

/**
 * Uninitialize the filter.
 */
static av_cold void dnn_clip_uninit(AVFilterContext *context)
{
    CLIPContext *ctx = context->priv;
    ff_dnn_uninit(&ctx->dnnctx);
    free_classify_labels(ctx);
}
/**
 * Handle flushing of frames in the filter.
 * Ensures any remaining frames are processed when the stream ends.
 */
static int dnn_clip_flush_frame(AVFilterLink *outlink, int64_t pts, int64_t *out_pts)
{
    CLIPContext *ctx = outlink->src->priv;
    int ret;
    DNNAsyncStatusType async_state;

    ret = ff_dnn_flush(&ctx->dnnctx);
    if (ret != 0) {
        return -1;
    }

    do {
        AVFrame *in_frame = NULL;
        AVFrame *out_frame = NULL;
        async_state = ff_dnn_get_result(&ctx->dnnctx, &in_frame, &out_frame);
        if (async_state == DAST_SUCCESS) {
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

/**
 * Filter activation function.
 * Controls the flow of frames through the filter and manages async processing.
 */
static int dnn_clip_activate(AVFilterContext *filter_ctx)
{
    AVFilterLink *inlink = filter_ctx->inputs[0];
    AVFilterLink *outlink = filter_ctx->outputs[0];
    CLIPContext *ctx = filter_ctx->priv;
    AVFrame *in = NULL;
    int64_t pts;
    int ret, status;
    int got_frame = 0;
    int async_state;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    do {
        // Process all available input frames
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
        if (ret > 0) {
            if (ff_dnn_execute_model_clip(&ctx->dnnctx, in, NULL, ctx->labels, ctx->tokenizer_path, ctx->label_count) != 0) {
                return AVERROR(EIO);
            }
        }
    } while (ret > 0);

    // Handle processed frames
    do {
        AVFrame *in_frame = NULL;
        AVFrame *out_frame = NULL;
        async_state = ff_dnn_get_result(&ctx->dnnctx, &in_frame, &out_frame);
        if (async_state == DAST_SUCCESS) {
            ret = ff_filter_frame(outlink, in_frame);
            if (ret < 0)
                return ret;
            got_frame = 1;
        }
    } while (async_state == DAST_SUCCESS);

    // Schedule next filter if frame was processed
    if (got_frame)
        return 0;

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            int64_t out_pts = pts;
            ret = dnn_clip_flush_frame(outlink, pts, &out_pts);
            ff_outlink_set_status(outlink, status, out_pts);
            return ret;
        }
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return 0;
}

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,
    AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAYF32,
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE
};

const AVFilter ff_vf_dnn_clip = {
    .name          = "dnn_clip",
    .description   = NULL_IF_CONFIG_SMALL("Apply CLIP zero-shot classification."),
    .preinit       = ff_dnn_filter_init_child_class,
    .priv_size     = sizeof(CLIPContext),
    .init          = dnn_clip_init,
    .uninit        = dnn_clip_uninit,
    .activate      = dnn_clip_activate,  
    FILTER_INPUTS(ff_video_default_filterpad),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .priv_class    = &dnn_clip_class,
};