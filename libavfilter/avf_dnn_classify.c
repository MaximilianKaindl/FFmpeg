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
 * DNN classification filter supporting both video (standard and CLIP)
 * and audio (CLAP) classification
 */

#include "audio.h"
#include "avfilter.h"
#include "dnn/dnn_labels.h"
#include "dnn_filter_common.h"
#include "filters.h"
#include "formats.h"
#include "libavutil/avstring.h"
#include "libavutil/detection_bbox.h"
#include "libavutil/file_open.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/time.h"
#include "video.h"

#define TYPE_ALL 2  // Number of media types (video and audio)

typedef struct DnnClassifyContext {
    const AVClass *class;
    DnnContext dnnctx;
    float confidence;
    char *labels_filename;
    char *target;
    enum AVMediaType type;  // AVMEDIA_TYPE_VIDEO or AVMEDIA_TYPE_AUDIO

    // Standard classification
    LabelContext *label_classification_ctx;

    CategoryClassifcationContext *category_classification_ctx;

    char *categories_filename;
    char *tokenizer_path;

    // Audio-specific parameters
    int is_audio;               // New flag to indicate if input will be audio
} DnnClassifyContext;

#define OFFSET(x) offsetof(DnnClassifyContext, dnnctx.x)
#define OFFSET2(x) offsetof(DnnClassifyContext, x)
#if (CONFIG_LIBTORCH == 1)
#define OFFSET3(x) offsetof(DnnClassifyContext, dnnctx.torch_option.x)
#endif
#define FLAGS                                               \
    AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM | \
        AV_OPT_FLAG_AUDIO_PARAM

        static const AVOption dnn_classify_options[] = {
            {"dnn_backend", "DNN backend", OFFSET(backend_type), AV_OPT_TYPE_INT,
             {.i64 = DNN_OV}, INT_MIN, INT_MAX, FLAGS, .unit = "backend"},
        #if (CONFIG_LIBOPENVINO == 1)
            {"openvino", "openvino backend flag", 0, AV_OPT_TYPE_CONST,
             {.i64 = DNN_OV}, 0, 0, FLAGS, .unit = "backend"},
        #endif
        #if (CONFIG_LIBTORCH == 1)
            {"torch", "torch backend flag", 0, AV_OPT_TYPE_CONST,
             {.i64 = DNN_TH}, 0, 0, FLAGS, .unit = "backend"},
             {"logit_scale", "logit scale for similarity calculation",
                OFFSET3(logit_scale), AV_OPT_TYPE_FLOAT, {.dbl = -1.0}, -1.0, 100.0, FLAGS},
            {"temperature", "softmax temperature", OFFSET3(temperature),
                AV_OPT_TYPE_FLOAT, {.dbl = 1.0}, 1, 100.0, FLAGS},
            {"forward_order", "Order of forward output (0: media text, 1: text media) (CLIP/CLAP only)", OFFSET3(forward_order),
                    AV_OPT_TYPE_BOOL, {.i64 = -1}, -1, 1, FLAGS},
            {"normalize", "Normalize the input tensor (CLIP/CLAP only)", OFFSET3(normalize),
                    AV_OPT_TYPE_BOOL, {.i64 = -1}, -1, 1, FLAGS},
            {"sample_rate_clap", "audio processing model expected sample rate", OFFSET3(sample_rate),
                AV_OPT_TYPE_INT64, {.i64 = 44100}, 1600, 192000, FLAGS},
            {"sample_duration", "audio processing model expected sample duration", OFFSET3(sample_duration),
                AV_OPT_TYPE_INT64, {.i64 = 7}, 1, 100, FLAGS},
            {"token_dimension", "dimension of token vector", OFFSET3(token_dimension),
                AV_OPT_TYPE_INT64, {.i64 = 77}, 1, 10000, FLAGS},
        #endif
            {"confidence", "threshold of confidence", OFFSET2(confidence),
             AV_OPT_TYPE_FLOAT, {.dbl = 0.5}, 0, 1, FLAGS},
            {"labels", "path to labels file", OFFSET2(labels_filename),
             AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
            {"target", "which one to be classified", OFFSET2(target),
             AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
            {"categories", "path to categories file (CLIP/CLAP only)",
             OFFSET2(categories_filename), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
            {"tokenizer", "path to text tokenizer.json file (CLIP/CLAP only)",
             OFFSET2(tokenizer_path), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
            {"is_audio", "audio processing mode", OFFSET2(is_audio),
                AV_OPT_TYPE_BOOL, {.i64 = 0}, 0, 1, FLAGS},
            {NULL}
        };

AVFILTER_DNN_DEFINE_CLASS(dnn_classify, DNN_OV);

static AVDetectionBBox *find_or_create_detection_bbox(
    AVFrame *frame, uint32_t bbox_index, AVFilterContext *filter_ctx,
    DnnClassifyContext *ctx)
{
    AVFrameSideData *sd;
    AVDetectionBBoxHeader *header;
    AVDetectionBBox *bbox;

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DETECTION_BBOXES);
    if (!sd) {
        header = av_detection_bbox_create_side_data(frame, 1);
        if (!header) {
            av_log(filter_ctx, AV_LOG_ERROR,
                   "Cannot get side data in labels processing\n");
            return NULL;
        }
    } else {
        header = (AVDetectionBBoxHeader *)sd->data;
    }

    if (bbox_index == 0) {
        av_strlcat(header->source, ", ", sizeof(header->source));
        av_strlcat(header->source, ctx->dnnctx.model_filename,
                   sizeof(header->source));
    }

    // Get bbox for current index
    bbox = av_get_detection_bbox(header, bbox_index);
    if (!bbox) {
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to get bbox %d\n", bbox_index);
        return NULL;
    }

    return bbox;
}

// Processing functions for standard classification (video only)
static int post_proc_standard(AVFrame *frame, DNNData *output,
                              uint32_t bbox_index, AVFilterContext *filter_ctx)
{
    DnnClassifyContext *ctx = filter_ctx->priv;
    float conf_threshold = ctx->confidence;
    AVDetectionBBoxHeader *header;
    AVDetectionBBox *bbox;
    float *classifications;
    uint32_t label_id;
    float confidence;
    AVFrameSideData *sd;
    int output_size = output->dims[3] * output->dims[2] * output->dims[1];

    if (output_size <= 0) {
        return -1;
    }

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DETECTION_BBOXES);
    if (!sd) {
        av_log(filter_ctx, AV_LOG_ERROR,
               "Cannot get side data in post_proc_standard\n");
        return -1;
    }
    header = (AVDetectionBBoxHeader *)sd->data;

    if (bbox_index == 0) {
        av_strlcat(header->source, ", ", sizeof(header->source));
        av_strlcat(header->source, ctx->dnnctx.model_filename,
                   sizeof(header->source));
    }

    classifications = output->data;
    label_id = 0;
    confidence = classifications[0];
    for (int i = 1; i < output_size; i++) {
        if (classifications[i] > confidence) {
            label_id = i;
            confidence = classifications[i];
        }
    }

    if (confidence < conf_threshold) {
        return 0;
    }

    bbox = av_get_detection_bbox(header, bbox_index);
    bbox->classify_confidences[bbox->classify_count] =
        av_make_q((int)(confidence * 10000), 10000);

    if (ctx->label_classification_ctx->labels &&
        label_id < ctx->label_classification_ctx->label_count) {
        av_strlcpy(bbox->classify_labels[bbox->classify_count],
                   ctx->label_classification_ctx->labels[label_id],
                   sizeof(bbox->classify_labels[bbox->classify_count]));
    } else {
        snprintf(bbox->classify_labels[bbox->classify_count],
                 sizeof(bbox->classify_labels[bbox->classify_count]), "%d",
                 label_id);
    }
    bbox->classify_count++;

    return 0;
}

static int post_proc_clxp_labels(AVFrame *frame, DNNData *output,
                                 uint32_t bbox_index,
                                 AVFilterContext *filter_ctx)
{
    DnnClassifyContext *ctx = filter_ctx->priv;
    const int max_classes_per_box = AV_NUM_DETECTION_BBOX_CLASSIFY;
    float *probabilities = (float *)output->data;
    int num_labels = ctx->label_classification_ctx->label_count;
    AVDetectionBBox *bbox;
    float confidence_threshold = ctx->confidence;
    int ret;

    // Get or create detection bbox
    bbox = find_or_create_detection_bbox(frame, bbox_index, filter_ctx, ctx);
    if (!bbox) {
        return AVERROR(EINVAL);
    }

    ret = av_detection_bbox_fill_with_best_labels(ctx->label_classification_ctx->labels,
                                     probabilities, num_labels, bbox,
                                     max_classes_per_box, confidence_threshold);
    if (ret < 0) {
        av_log(filter_ctx, AV_LOG_ERROR,
               "Failed to fill bbox with best labels\n");
        return ret;
    }
    return 0;
}

static int post_proc_clxp_categories(AVFrame *frame, DNNData *output,
                                     uint32_t bbox_index,
                                     AVFilterContext *filter_ctx)
{
    DnnClassifyContext *ctx = filter_ctx->priv;
    CategoryClassifcationContext *cat_class_ctx =
        ctx->category_classification_ctx;
    CategoryContext *best_category;
    AVDetectionBBox *bbox;
    float *probabilities = output->data;
    int ret, prob_offset = 0;
    char **ctx_labels;
    float *ctx_probabilities;

    // Get or create detection bbox
    bbox = find_or_create_detection_bbox(frame, bbox_index, filter_ctx, ctx);
    if (!bbox) {
        return AVERROR(EINVAL);
    }

    // Allocate temporary arrays
    ctx_labels = av_malloc_array(cat_class_ctx->num_contexts, sizeof(char *));
    if (!ctx_labels) {
        return AVERROR(ENOMEM);
    }

    for (int i = 0; i < cat_class_ctx->num_contexts; i++) {
        ctx_labels[i] = av_mallocz(AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE);
        if (!ctx_labels[i]) {
            // Clean up previously allocated memory
            for (int j = 0; j < i; j++) {
                av_freep(&ctx_labels[j]);
            }
            av_freep(&ctx_labels);
            return AVERROR(ENOMEM);
        }
    }

    ctx_probabilities =
        av_malloc_array(cat_class_ctx->num_contexts, sizeof(float));
    if (!ctx_probabilities) {
        // Clean up
        for (int i = 0; i < cat_class_ctx->num_contexts; i++) {
            av_freep(&ctx_labels[i]);
        }
        av_freep(&ctx_labels);
        return AVERROR(ENOMEM);
    }

    // Process each context
    for (int ctx_idx = 0; ctx_idx < cat_class_ctx->num_contexts; ctx_idx++) {
        CategoriesContext *categories_ctx =
            cat_class_ctx->category_units[ctx_idx];
        if (!categories_ctx) {
            av_log(filter_ctx, AV_LOG_ERROR,
                   "Missing classification data at context %d\n", ctx_idx);
            continue;
        }

        // Find best category
        best_category =
            get_best_category(categories_ctx, probabilities + prob_offset);
        if (!best_category || !best_category->name) {
            av_log(filter_ctx, AV_LOG_ERROR,
                   "Invalid best category at context %d\n", ctx_idx);
            continue;
        }

        // Copy category name instead of assigning pointer
        av_strlcpy(ctx_labels[ctx_idx], best_category->name,
                   AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE);
        ctx_probabilities[ctx_idx] = best_category->total_probability;

        prob_offset += categories_ctx->label_count;
    }

    // Fill bbox with best labels
    ret = av_detection_bbox_fill_with_best_labels(
        ctx_labels, ctx_probabilities, cat_class_ctx->num_contexts, bbox,
        AV_NUM_DETECTION_BBOX_CLASSIFY, ctx->confidence);

    // Clean up
    for (int i = 0; i < cat_class_ctx->num_contexts; i++) {
        av_freep(&ctx_labels[i]);
    }
    av_freep(&ctx_labels);
    av_freep(&ctx_probabilities);

    return ret;
}

static int dnn_classify_post_proc(AVFrame *frame, DNNData *output,
                                  uint32_t bbox_index,
                                  AVFilterContext *filter_ctx)
{
    DnnClassifyContext *ctx = filter_ctx->priv;

    if (!frame || !output || !output->data) {
        av_log(filter_ctx, AV_LOG_ERROR, "Invalid input to post processing\n");
        return AVERROR(EINVAL);
    }

    // Choose post-processing based on backend and context
    if (ctx->dnnctx.backend_type == DNN_TH) {
        if (ctx->category_classification_ctx) {
            return post_proc_clxp_categories(frame, output, bbox_index,
                                             filter_ctx);
        } else if (ctx->label_classification_ctx) {
            return post_proc_clxp_labels(frame, output, bbox_index, filter_ctx);
        }
        av_log(filter_ctx, AV_LOG_ERROR,
               "No valid classification context available\n");
        return AVERROR(EINVAL);
    } else {
        // Standard classification (video only)
        return post_proc_standard(frame, output, bbox_index, filter_ctx);
    }
}

static void free_contexts(DnnClassifyContext *ctx)
{
    if (!ctx)
        return;
    if (ctx->category_classification_ctx) {
        free_category_classfication_context(ctx->category_classification_ctx);
        av_freep(&ctx->category_classification_ctx);
        av_freep(&ctx->label_classification_ctx);
        ctx->category_classification_ctx = NULL;
        ctx->label_classification_ctx = NULL;
    } else if (ctx->label_classification_ctx) {
        free_label_context(ctx->label_classification_ctx);
        ctx->label_classification_ctx = NULL;
    }
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *context = inlink->dst;
    DnnClassifyContext *ctx = context->priv;
    AVFilterLink *outlink = context->outputs[0];
    int ret;
    DNNFunctionType goal_mode;

    // Set media type based on is_audio flag
    if (ctx->is_audio) {
        ctx->type = AVMEDIA_TYPE_AUDIO;
    } else {
        ctx->type = inlink->type;
    }

    // Set the output link type to match the input link type
    outlink->type = inlink->type;
    outlink->w = inlink->w;
    outlink->h = inlink->h;
    outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
    outlink->time_base = inlink->time_base;

    int64_t sample_rate = ctx->dnnctx.torch_option.sample_rate;

    // Validate media type
    if (ctx->type != AVMEDIA_TYPE_AUDIO && ctx->type != AVMEDIA_TYPE_VIDEO) {
        av_log(context, AV_LOG_ERROR,
               "Invalid media type. Only audio or video is supported\n");
        return AVERROR(EINVAL);
    }

    // Set type-specific parameters and check compatibility
    if (ctx->type == AVMEDIA_TYPE_AUDIO) {
        // Audio-specific settings
        goal_mode = DFT_ANALYTICS_CLAP;

        // Check backend compatibility
        if (ctx->dnnctx.backend_type != DNN_TH) {
            av_log(context, AV_LOG_ERROR,
                   "Audio classification requires Torch backend\n");
            return AVERROR(EINVAL);
        }

        // Check sample rate
        if (inlink->sample_rate != sample_rate) {
            av_log(context, AV_LOG_ERROR,
                   "Invalid sample rate. CLAP requires 44100 Hz\n");
            return AVERROR(EINVAL);
        }

        // Copy audio properties to output
        outlink->sample_rate = inlink->sample_rate;
        outlink->ch_layout = inlink->ch_layout;
    } else {
        // Video mode
        goal_mode = (ctx->dnnctx.backend_type == DNN_TH)
                        ? DFT_ANALYTICS_CLIP
                        : DFT_ANALYTICS_CLASSIFY;
    }
    // Initialize label and category contexts based on provided files
    if (ctx->dnnctx.backend_type == DNN_TH) {
        if (ctx->labels_filename) {
            ctx->label_classification_ctx = av_calloc(1, sizeof(LabelContext));
            if (!ctx->label_classification_ctx)
                return AVERROR(ENOMEM);

            ret = read_label_file(context, ctx->label_classification_ctx,
                                  ctx->labels_filename,
                                  AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE);
            if (ret < 0) {
                av_log(context, AV_LOG_ERROR, "Failed to read labels file\n");
                return ret;
            }
            // Initialize DNN with tokenizer for CLIP/CLAP models
            ret = ff_dnn_init_with_tokenizer(
                &ctx->dnnctx, goal_mode, ctx->label_classification_ctx->labels,
                ctx->label_classification_ctx->label_count, NULL, 0, ctx->tokenizer_path,
                context);
            if (ret < 0) {
                free_contexts(ctx);
                return ret;
            }
        } else if (ctx->categories_filename) {
            ctx->category_classification_ctx =
                av_calloc(1, sizeof(CategoryClassifcationContext));
            if (!ctx->category_classification_ctx)
                return AVERROR(ENOMEM);

            ret = read_categories_file(context, ctx->category_classification_ctx,
                                     ctx->categories_filename,
                                     AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE);
            if (ret < 0) {
                av_log(context, AV_LOG_ERROR, "Failed to read categories file\n");
                free_contexts(ctx);
                return ret;
            }

            ret = combine_all_category_labels(&ctx->label_classification_ctx,
                                              ctx->category_classification_ctx);
            if (ret < 0) {
                av_log(context, AV_LOG_ERROR, "Failed to combine labels\n");
                free_contexts(ctx);
                return ret;
            }
            // Get label counts for categories
            int total_labels;
            int *label_counts = NULL;

            total_labels = get_category_label_counts(ctx->category_classification_ctx,
                                            &label_counts);
            if (total_labels <= 0) {
                av_log(context, AV_LOG_ERROR, "Failed to get category label counts or no labels found\n");
                free_contexts(ctx);
                return ret;
            }

            // Initialize DNN with tokenizer for CLIP/CLAP models
            ret = ff_dnn_init_with_tokenizer(
                &ctx->dnnctx, goal_mode, ctx->label_classification_ctx->labels,
                ctx->label_classification_ctx->label_count,
                label_counts, total_labels, ctx->tokenizer_path,
                context);
            if (ret < 0) {
                av_freep(&label_counts);
                free_contexts(ctx);
                return ret;
            }
            av_freep(&label_counts);
        }
    } else if (ctx->dnnctx.backend_type == DNN_OV) {
        // Initialize standard DNN for OpenVINO
        ret = ff_dnn_init(&ctx->dnnctx, goal_mode, context);
        if (ret < 0)
            return ret;

        // Read labels file
        ctx->label_classification_ctx = av_calloc(1, sizeof(LabelContext));
        if (!ctx->label_classification_ctx)
            return AVERROR(ENOMEM);

        ret = read_label_file(context, ctx->label_classification_ctx,
                              ctx->labels_filename,
                              AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE);
        if (ret < 0) {
            av_log(context, AV_LOG_ERROR, "Failed to read labels file\n");
            free_contexts(ctx);
            return ret;
        }
    }

    // Set the post-processing callback
    ff_dnn_set_classify_post_proc(&ctx->dnnctx, dnn_classify_post_proc);
    return 0;
}

static av_cold int dnn_classify_init(AVFilterContext *context)
{
    DnnClassifyContext *ctx = context->priv;
    int ret;

    // Create a static pad with the appropriate media type
    AVFilterPad pad = {
        .name = av_strdup("default"),
        .type = ctx->is_audio ? AVMEDIA_TYPE_AUDIO : AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
    };

    ret = ff_append_inpad(context, &pad);
    if (ret < 0)
        return ret;

    // Create a matching output pad
    AVFilterPad outpad = {
        .name = av_strdup("default"),
        .type = ctx->is_audio ? AVMEDIA_TYPE_AUDIO : AVMEDIA_TYPE_VIDEO,
    };

    ret = ff_append_outpad(context, &outpad);
    if (ret < 0)
        return ret;

    // Check backend and file parameters (parameter validation only)
    if (ctx->dnnctx.backend_type == DNN_TH) {
        // Check CLIP/CLAP specific parameters
        if (ctx->labels_filename && ctx->categories_filename) {
            av_log(context, AV_LOG_ERROR,
                   "Labels and categories file cannot be used together\n");
            return AVERROR(EINVAL);
        }

        if (!ctx->labels_filename && !ctx->categories_filename) {
            av_log(
                context, AV_LOG_ERROR,
                "Labels or categories file is required for classification\n");
            return AVERROR(EINVAL);
        }

        if (!ctx->tokenizer_path) {
            av_log(context, AV_LOG_ERROR,
                   "Tokenizer file is required for CLIP/CLAP classification\n");
            return AVERROR(EINVAL);
        }
    } else if (ctx->dnnctx.backend_type == DNN_OV) {
        // Check OpenVINO specific parameters
        if (!ctx->labels_filename) {
            av_log(context, AV_LOG_ERROR,
                   "Labels file is required for classification\n");
            return AVERROR(EINVAL);
        }

        if (ctx->categories_filename) {
            av_log(context, AV_LOG_ERROR,
                   "Categories file is only supported for CLIP/CLAP models\n");
            return AVERROR(EINVAL);
        }

        // Audio classification is not supported with OpenVINO backend
        if (ctx->is_audio) {
            av_log(context, AV_LOG_ERROR,
                   "Audio classification requires Torch backend\n");
            return AVERROR(EINVAL);
        }
    }
    return 0;
}

static int query_formats(AVFilterContext *ctx)
{
    DnnClassifyContext *classify_ctx = ctx->priv;

    int ret;
    // Get the type from the first input pad
    enum AVMediaType type = ctx->inputs[0]->type;

    if (type == AVMEDIA_TYPE_VIDEO) {
        static const enum AVPixelFormat pix_fmts[] = {
            AV_PIX_FMT_RGB24,   AV_PIX_FMT_BGR24,   AV_PIX_FMT_GRAY8,
            AV_PIX_FMT_GRAYF32, AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
            AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
            AV_PIX_FMT_NV12,    AV_PIX_FMT_NONE};

        ret = ff_set_common_formats(ctx, ff_make_format_list(pix_fmts));
        if (ret < 0)
            return ret;
    } else if (type == AVMEDIA_TYPE_AUDIO) {
        static const enum AVSampleFormat sample_fmts[] = {AV_SAMPLE_FMT_FLT,
                                                          AV_SAMPLE_FMT_NONE};

        ret = ff_set_common_formats(ctx, ff_make_format_list(sample_fmts));
        if (ret < 0)
            return ret;
        #if (CONFIG_LIBTORCH == 1)
        ret = ff_set_common_samplerates(
            ctx, ff_make_format_list((const int[]){classify_ctx->dnnctx.torch_option.sample_rate, -1}));
        if (ret < 0)
            return ret;
        #endif
        ret = ff_set_common_channel_layouts(ctx, ff_all_channel_layouts());
        if (ret < 0)
            return ret;
    } else {
        av_log(ctx, AV_LOG_ERROR, "Unsupported media type: %d\n", type);
        return AVERROR(EINVAL);
    }

    return 0;
}
static int dnn_classify_flush_frame(AVFilterLink *outlink, int64_t pts,
                                    int64_t *out_pts)
{
    AVFilterContext *context = outlink->src;
    DnnClassifyContext *ctx = context->priv;
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

static int process_video_frame(AVFilterContext *context, AVFrame *frame)
{
    DnnClassifyContext *ctx = context->priv;
    int ret;

    if (ctx->dnnctx.backend_type == DNN_TH) {
        ret = ff_dnn_execute_model_clip(
            &ctx->dnnctx, frame, NULL, ctx->label_classification_ctx->labels,
            ctx->label_classification_ctx->label_count, ctx->tokenizer_path,
            ctx->target);
    } else {
        ret = ff_dnn_execute_model_classification(&ctx->dnnctx, frame, NULL,
                                                  ctx->target);
    }

    if (ret != 0) {
        av_frame_free(&frame);
        return AVERROR(EIO);
    }

    return 0;
}

static int process_audio_frame(AVFilterContext *context, AVFrame *frame)
{
    DnnClassifyContext *ctx = context->priv;
    int ret;

    int64_t samples_per_frame = ctx->dnnctx.torch_option.sample_rate * ctx->dnnctx.torch_option.sample_duration;

    if (frame->nb_samples < samples_per_frame) {
        av_log(context, AV_LOG_WARNING,
               "Audio frame too short for CLAP analysis (needs %d samples, got "
               "%d)\n",
               samples_per_frame, frame->nb_samples);
    }

    ret = ff_dnn_execute_model_clap(
        &ctx->dnnctx, frame, NULL, ctx->label_classification_ctx->labels,
        ctx->label_classification_ctx->label_count, ctx->tokenizer_path);

    if (ret != 0) {
        av_frame_free(&frame);
        return AVERROR(EIO);
    }

    return 0;
}

static int dnn_classify_activate(AVFilterContext *context)
{
    DnnClassifyContext *ctx = context->priv;
    AVFilterLink *inlink = context->inputs[0];
    AVFilterLink *outlink = context->outputs[0];
    int ret, status;
    int64_t pts;
    AVFrame *in = NULL;
    int got_frame = 0;
    DNNAsyncStatusType async_state;

    // Check for EOF or other status
    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            int64_t out_pts = pts;
            ret = dnn_classify_flush_frame(outlink, pts, &out_pts);
            ff_outlink_set_status(outlink, status, out_pts);
            return ret;
        }
    }

    // Process frames
    ret = ff_inlink_consume_frame(inlink, &in);
    if (ret < 0)
        return ret;
    if (ret > 0) {
        // Process frame based on media type
        if (ctx->type == AVMEDIA_TYPE_VIDEO) {
            ret = process_video_frame(context, in);
        } else {
            ret = process_audio_frame(context, in);
        }

        if (ret < 0) {
            av_frame_free(&in);
            return ret;
        }
    }

    // Get processed results
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

    if (got_frame)
        return 0;

    // Request more frames if needed
    if (ff_outlink_frame_wanted(outlink))
        ff_inlink_request_frame(inlink);

    return FFERROR_NOT_READY;
}

static av_cold void dnn_classify_uninit(AVFilterContext *context)
{
    DnnClassifyContext *ctx = context->priv;
    ff_dnn_uninit(&ctx->dnnctx);
    free_contexts(ctx);
}

const FFFilter ff_avf_dnn_classify = {
    .p.name = "dnn_classify",
    .p.description = NULL_IF_CONFIG_SMALL("Apply DNN classification filter to the input."),
    .p.priv_class = &dnn_classify_class,
    .p.flags = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
    .priv_size = sizeof(DnnClassifyContext),
    .preinit = ff_dnn_filter_init_child_class,
    .init = dnn_classify_init,
    .uninit = dnn_classify_uninit,
    .activate = dnn_classify_activate,
    FILTER_QUERY_FUNC2(query_formats),
};
