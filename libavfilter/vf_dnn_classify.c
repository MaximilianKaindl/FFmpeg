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
 * DNN classification filter supporting both standard classification and CLIP zero-shot classification
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
#include "dnn/dnn_labels.h"

typedef struct DnnClassifyContext {
    const AVClass *class;
    DnnContext dnnctx;
    float confidence;
    char *labels_filename;
    char *target;

    // Standard classification
    LabelContext *label_classification_ctx;

    // CLIP-specific fields
    // classify in categories
    CategoryClassifcationContext *category_classification_ctx;

    // Parameters to change Results of the simularity calculation
    float logit_scale;
    float temperature;

    char *categories_filename;
    char *tokenizer_path;
} DnnClassifyContext;

#define OFFSET(x) offsetof(DnnClassifyContext, dnnctx.x)
#define OFFSET2(x) offsetof(DnnClassifyContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM
static const AVOption dnn_classify_options[] = {
    {
        "dnn_backend", "DNN backend", OFFSET(backend_type), AV_OPT_TYPE_INT, {.i64 = DNN_OV}, INT_MIN, INT_MAX, FLAGS,
        .unit = "backend"
    },
#if (CONFIG_LIBOPENVINO == 1)
    { "openvino",    "openvino backend flag",      0,                        AV_OPT_TYPE_CONST,     { .i64 = DNN_OV },    0, 0, FLAGS, .unit = "backend" },
#endif
#if (CONFIG_LIBTORCH == 1)
    {"torch", "torch backend flag", 0, AV_OPT_TYPE_CONST, {.i64 = DNN_TH}, 0, 0, FLAGS, .unit = "backend"},
#endif
    {"confidence", "threshold of confidence", OFFSET2(confidence), AV_OPT_TYPE_FLOAT, {.dbl = 0.5}, 0, 1, FLAGS},
    {"labels", "path to labels file", OFFSET2(labels_filename), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    {"target", "which one to be classified", OFFSET2(target), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    {
        "categories", "path to categories file (CLIP only)", OFFSET2(categories_filename), AV_OPT_TYPE_STRING,
        {.str = NULL}, 0, 0, FLAGS
    },
    {"logit_scale", "logit scale for CLIP", OFFSET2(logit_scale), AV_OPT_TYPE_FLOAT, {.dbl = 4.6052}, 0, 100.0, FLAGS},
    {
        "temperature", "softmax temperature for CLIP", OFFSET2(temperature), AV_OPT_TYPE_FLOAT, {.dbl = 1.0}, 0, 100.0,
        FLAGS
    },
    {
        "tokenizer", "path to text tokenizer.json file (CLIP only)", OFFSET2(tokenizer_path), AV_OPT_TYPE_STRING,
        {.str = NULL}, 0, 0, FLAGS
    },
    {NULL}
};

AVFILTER_DNN_DEFINE_CLASS(dnn_classify, DNN_OV);


static int dnn_classify_set_prob_and_label_of_bbox(AVDetectionBBox *bbox, char *label, int index, float probability) {
    // Validate parameters
    if (!bbox || !label) {
        av_log(NULL, AV_LOG_ERROR, "Invalid parameters in set_prob_and_label_of_bbox\n");
        return AVERROR(EINVAL);
    }

    // Check index bounds
    if (index < 0 || index >= AV_NUM_DETECTION_BBOX_CLASSIFY) {
        av_log(NULL, AV_LOG_ERROR, "Invalid index %d in set_prob_and_label_of_bbox\n", index);
        return AVERROR(EINVAL);
    }

    // Set probability
    bbox->classify_confidences[index] = av_make_q((int) (probability * 10000), 10000);

    // Copy label with size checking
    if (av_strlcpy(bbox->classify_labels[index], label,
                   AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE) >= AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE) {
        av_log(NULL, AV_LOG_WARNING, "Label truncated in set_prob_and_label_of_bbox\n");
    }

    av_log(NULL, AV_LOG_DEBUG, "Set bbox label: %s with probability: %f at index: %d\n",
           bbox->classify_labels[index], probability, index);

    return 0;
}

static int softmax(float *input, size_t input_len, float logit_scale, float temperature, AVFilterContext *ctx) {
    float sum, offset, m;

    if (!input || input_len == 0) {
        av_log(ctx, AV_LOG_ERROR, "Invalid input to softmax\n");
        return AVERROR(EINVAL);
    }

    if (temperature <= 0.0f) {
        temperature = 1.0f;
    }

    // Apply logit scale
    for (size_t i = 0; i < input_len; i++) {
        input[i] *= logit_scale;
    }

    m = input[0];
    for (size_t i = 1; i < input_len; i++) {
        if (input[i] > m) {
            m = input[i];
        }
    }

    sum = 0.0f;
    for (size_t i = 0; i < input_len; i++) {
        sum += expf((input[i] - m) / temperature);
    }

    if (sum == 0.0f) {
        av_log(ctx, AV_LOG_ERROR, "Division by zero in softmax\n");
        return AVERROR(EINVAL);
    }

    offset = m + temperature * logf(sum);
    for (size_t i = 0; i < input_len; i++) {
        input[i] = expf((input[i] - offset) / temperature);
    }

    return 0;
}

static int post_proc_standard(AVFrame *frame, DNNData *output, uint32_t bbox_index, AVFilterContext *filter_ctx) {
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
        av_log(filter_ctx, AV_LOG_ERROR, "Cannot get side data in dnn_classify_post_proc_standard\n");
        return -1;
    }
    header = (AVDetectionBBoxHeader *) sd->data;

    if (bbox_index == 0) {
        av_strlcat(header->source, ", ", sizeof(header->source));
        av_strlcat(header->source, ctx->dnnctx.model_filename, sizeof(header->source));
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
    bbox->classify_confidences[bbox->classify_count] = av_make_q((int) (confidence * 10000), 10000);

    if (ctx->label_classification_ctx->labels && label_id < ctx->label_classification_ctx->label_count) {
        av_strlcpy(bbox->classify_labels[bbox->classify_count], ctx->label_classification_ctx->labels[label_id],
                   sizeof(bbox->classify_labels[bbox->classify_count]));
    } else {
        snprintf(bbox->classify_labels[bbox->classify_count], sizeof(bbox->classify_labels[bbox->classify_count]), "%d",
                 label_id);
    }

    bbox->classify_count++;

    return 0;
}


static AVDetectionBBox *
find_or_create_detection_bbox(AVFrame *frame, uint32_t bbox_index, AVFilterContext *filter_ctx) {
    DnnClassifyContext *ctx = filter_ctx->priv;
    AVFrameSideData *sd;
    AVDetectionBBoxHeader *header;
    AVDetectionBBox *bbox;

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DETECTION_BBOXES);
    if (!sd) {
        header = av_detection_bbox_create_side_data(frame, 1);
        if (!header) {
            av_log(filter_ctx, AV_LOG_ERROR, "Cannot get side data in CLIP labels processing\n");
            return AVERROR(EINVAL);
        }
    } else {
        header = (AVDetectionBBoxHeader *) sd->data;
    }

    if (bbox_index == 0) {
        av_strlcat(header->source, ", ", sizeof(header->source));
        av_strlcat(header->source, ctx->dnnctx.model_filename, sizeof(header->source));
    }
    // Get bbox for current index
    bbox = av_get_detection_bbox(header, bbox_index);
    if (!bbox) {
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to get bbox %d\n", bbox_index);
        return AVERROR(EINVAL);
    }

    return bbox;
}

static int fill_bbox_with_best_labels(DnnClassifyContext *ctx, char **labels, float *probabilities,
                                      int num_labels, AVDetectionBBox *bbox,
                                      int max_classes_per_box, float confidence_threshold) {
    int i, j, minpos, ret;
    float min;

    if (!labels || !probabilities || !bbox) {
        return AVERROR(EINVAL);
    }

    for (i = 0; i < num_labels; i++) {
        if (probabilities[i] >= confidence_threshold) {
            if (bbox->classify_count >= max_classes_per_box) {
                // Find lowest probability classification
                min = av_q2d(bbox->classify_confidences[0]);
                minpos = 0;
                for (j = 1; j < bbox->classify_count; j++) {
                    float prob = av_q2d(bbox->classify_confidences[j]);
                    if (prob < min) {
                        min = prob;
                        minpos = j;
                    }
                }

                if (probabilities[i] > min) {
                    ret = dnn_classify_set_prob_and_label_of_bbox(bbox, labels[i], minpos, probabilities[i]);
                    if (ret < 0)
                        return ret;
                }
            } else {
                ret = dnn_classify_set_prob_and_label_of_bbox(bbox, labels[i], bbox->classify_count, probabilities[i]);
                if (ret < 0)
                    return ret;
                bbox->classify_count++;
            }
        }
    }
    return 0;
}

static int post_proc_clip_labels(AVFrame *frame, DNNData *output, uint32_t bbox_index, AVFilterContext *filter_ctx) {
    DnnClassifyContext *ctx = filter_ctx->priv;
    const int max_classes_per_box = AV_NUM_DETECTION_BBOX_CLASSIFY;
    float *probabilities = (float *) output->data;
    int num_labels = ctx->label_classification_ctx->label_count;
    AVDetectionBBox *bbox;
    float confidence_threshold = ctx->confidence;
    int ret, new = 0;

    // Apply softmax to probabilities
    if (softmax(probabilities, num_labels, ctx->logit_scale, ctx->temperature, filter_ctx) < 0) {
        return AVERROR(EINVAL);
    }

    // Get or create detection bbox
    bbox = find_or_create_detection_bbox(frame, bbox_index, filter_ctx);
    if (!bbox) {
        return AVERROR(EINVAL);
    }

    ret = fill_bbox_with_best_labels(ctx, ctx->label_classification_ctx->labels, probabilities, num_labels, bbox,
                                     max_classes_per_box, confidence_threshold);
    if (ret < 0) {
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to fill bbox with best labels\n");
        return ret;
    }
    return 0;
}

static int softmax_over_all_categories(DnnClassifyContext *ctx, AVFilterContext *filter_ctx, float *probabilities) {
    int prob_offset = 0;
    CategoryClassifcationContext *cat_class_ctx = ctx->category_classification_ctx;

    for (int c = 0; c < cat_class_ctx->num_contexts; c++) {
        CategoriesContext *categories_ctx = cat_class_ctx->category_units[c];
        if (!categories_ctx) {
            av_log(filter_ctx, AV_LOG_ERROR, "Missing classification data at context %d\n", c);
            continue;
        }
        // Apply softmax only to the labels within this category
        if (softmax(probabilities + prob_offset,
                    categories_ctx->label_count,
                    ctx->logit_scale,
                    ctx->temperature,
                    filter_ctx) < 0) {
            return AVERROR(EINVAL);
        }
        prob_offset += categories_ctx->label_count;
    }
    return 0;
}

static CategoryContext *get_best_category(CategoriesContext *categories_ctx, float *probabilities) {
    CategoryContext *best_category;
    float best_probability = -1.0f;
    int prob_offset = 0;
    // Calculate total probability for each category
    for (int cat_idx = 0; cat_idx < categories_ctx->category_count; cat_idx++) {
        CategoryContext *category = &categories_ctx->categories[cat_idx];
        // Sum probabilities for all labels in this category
        category->total_probability = 0.0f;
        for (int label_idx = 0; label_idx < category->label_count; label_idx++) {
            category->total_probability += probabilities[prob_offset + label_idx];
        }
        if (category->total_probability > best_probability) {
            best_probability = category->total_probability;
            best_category = category;
        }
        prob_offset += category->label_count;
    }
    return best_category;
}

static int post_proc_clip_categories(AVFrame *frame, DNNData *output, uint32_t bbox_index,
                                     AVFilterContext *filter_ctx) {
    DnnClassifyContext *ctx = filter_ctx->priv;
    CategoryClassifcationContext *cat_class_ctx = ctx->category_classification_ctx;
    CategoryContext *best_category;
    float *probabilities = output->data;
    int ret, prob_offset = 0;
    char **ctx_labels;
    float *ctx_probabilities;

    // Validate input data
    if (!probabilities || !cat_class_ctx) {
        av_log(filter_ctx, AV_LOG_ERROR, "Invalid input data\n");
        return AVERROR(EINVAL);
    }

    // Apply softmax transformation
    ret = softmax_over_all_categories(ctx, filter_ctx, probabilities);
    if (ret < 0) {
        return ret;
    }

    // Get or create detection bbox
    AVDetectionBBox *bbox = find_or_create_detection_bbox(frame, bbox_index, filter_ctx);
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

    ctx_probabilities = av_malloc_array(cat_class_ctx->num_contexts, sizeof(float));
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
        CategoriesContext *categories_ctx = cat_class_ctx->category_units[ctx_idx];
        if (!categories_ctx) {
            av_log(filter_ctx, AV_LOG_ERROR, "Missing classification data at context %d\n", ctx_idx);
            continue;
        }

        // Find best category
        best_category = get_best_category(categories_ctx, probabilities + prob_offset);
        if (!best_category || !best_category->name) {
            av_log(filter_ctx, AV_LOG_ERROR, "Invalid best category at context %d\n", ctx_idx);
            continue;
        }

        // Copy category name instead of assigning pointer
        av_strlcpy(ctx_labels[ctx_idx], best_category->name, AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE);
        ctx_probabilities[ctx_idx] = best_category->total_probability;

        prob_offset += categories_ctx->label_count;
    }

    // Fill bbox with best labels
    ret = fill_bbox_with_best_labels(ctx, ctx_labels, ctx_probabilities,
                                     cat_class_ctx->num_contexts, bbox,
                                     AV_NUM_DETECTION_BBOX_CLASSIFY,
                                     ctx->confidence);

    // Clean up
    for (int i = 0; i < cat_class_ctx->num_contexts; i++) {
        av_freep(&ctx_labels[i]);
    }
    av_freep(&ctx_labels);
    av_freep(&ctx_probabilities);

    return ret;
}

static int dnn_classify_post_proc(AVFrame *frame, DNNData *output, uint32_t bbox_index, AVFilterContext *filter_ctx) {
    DnnClassifyContext *ctx = filter_ctx->priv;
    if (!frame || !output || !output->data) {
        av_log(filter_ctx, AV_LOG_ERROR, "Invalid input to CLIP post processing\n");
        return AVERROR(EINVAL);
    }

    // Choose post-processing based on backend
    if (ctx->dnnctx.backend_type == DNN_TH) {
        if (ctx->label_classification_ctx) {
            return post_proc_clip_labels(frame, output, bbox_index, filter_ctx);
        } else if (ctx->category_classification_ctx) {
            return post_proc_clip_categories(frame, output, bbox_index, filter_ctx);
        }
        av_log(filter_ctx, AV_LOG_ERROR, "No valid CLIP classification context available\n");
        return AVERROR(EINVAL);
    } else {
        return post_proc_standard(frame, output, bbox_index, filter_ctx);
    }
}

static void free_contexts(DnnClassifyContext *ctx) {
    if (!ctx)
        return;

    if (ctx->label_classification_ctx) {
        free_label_context(ctx->label_classification_ctx);
        ctx->label_classification_ctx = NULL;
    }

    if (ctx->category_classification_ctx) {
        free_category_classfication_context(ctx->category_classification_ctx);
        av_freep(&ctx->category_classification_ctx);
        ctx->category_classification_ctx = NULL;
    }
}

static av_cold int dnn_classify_init(AVFilterContext *context) {
    DnnClassifyContext *ctx = context->priv;
    int ret;

    DNNFunctionType goal_mode = DFT_ANALYTICS_CLASSIFY;
    if (ctx->dnnctx.backend_type == DNN_TH) {
        goal_mode = DFT_ANALYTICS_CLIP;
    }

    ret = ff_dnn_init(&ctx->dnnctx, goal_mode, context);
    if (ret < 0)
        return ret;
    ff_dnn_set_classify_post_proc(&ctx->dnnctx, dnn_classify_post_proc);

    if (ctx->labels_filename && ctx->categories_filename) {
        av_log(context, AV_LOG_ERROR, "Labels and categories file cannot be used together\n");
        return AVERROR(EINVAL);
    }

    if (ctx->labels_filename) {
        ctx->label_classification_ctx = av_calloc(1, sizeof(LabelContext));
        if (!ctx->label_classification_ctx)
            return AVERROR(ENOMEM);
        ret = read_label_file(context, ctx->label_classification_ctx, ctx->labels_filename,
                              AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE);
        if (ret < 0) {
            av_log(context, AV_LOG_ERROR, "Failed to read labels file\n");
            return ret;
        }
    } else if (ctx->categories_filename) {
        ctx->category_classification_ctx = av_calloc(1, sizeof(CategoryClassifcationContext));
        if (!ctx->category_classification_ctx)
            return AVERROR(ENOMEM);

        ret = read_categories_file(context, ctx->category_classification_ctx, ctx->categories_filename,
                                   AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE);
        if (ret < 0) {
            av_log(context, AV_LOG_ERROR, "Failed to read categories file\n");
            free_contexts(ctx);
            return ret;
        }
    }

    // For CLIP models, require tokenizer
    if (ctx->dnnctx.backend_type == DNN_TH && !ctx->tokenizer_path) {
        av_log(context, AV_LOG_ERROR, "Tokenizer file is required for CLIP classification\n");
        return AVERROR(EINVAL);
    }

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

static int dnn_classify_flush_frame(AVFilterLink *outlink, int64_t pts, int64_t *out_pts) {
    DnnClassifyContext *ctx = outlink->src->priv;
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

static int execute_clip_model_for_all_categories(DnnClassifyContext *ctx, AVFrame *frame) {
    char **combined_labels = NULL;
    int combined_idx = 0;
    int ret;
    CategoryClassifcationContext *cat_class_ctx = ctx->category_classification_ctx;

    // Allocate array for all labels
    combined_labels = av_calloc(cat_class_ctx->total_labels, sizeof(char *));
    if (!combined_labels) {
        return AVERROR(ENOMEM);
    }
    for (int c = 0; c < cat_class_ctx->num_contexts; c++) {
        CategoriesContext *current_ctx = cat_class_ctx->category_units[c];
        for (int i = 0; i < current_ctx->category_count; i++) {
            CategoryContext *category = &current_ctx->categories[i];
            for (int j = 0; j < category->labels->label_count; j++) {
                combined_labels[combined_idx] = category->labels->labels[j];
                combined_idx++;
            }
        }
    }
    // Execute model with ALL labels combined
    ret = ff_dnn_execute_model_clip(&ctx->dnnctx, frame, NULL,
                                    combined_labels,
                                    cat_class_ctx->total_labels,
                                    ctx->tokenizer_path,
                                    ctx->target
    );

    av_freep(&combined_labels);
    return ret;
}

static int dnn_classify_activate(AVFilterContext *filter_ctx) {
    AVFilterLink *inlink = filter_ctx->inputs[0];
    AVFilterLink *outlink = filter_ctx->outputs[0];
    DnnClassifyContext *ctx = filter_ctx->priv;
    AVFrame *in = NULL;
    int64_t pts;
    int ret, status;
    int got_frame = 0;
    int async_state;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    do {
        // drain all input frames
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
        if (ret > 0) {
            if (ctx->dnnctx.backend_type == DNN_TH) {
                // CLIP processing
                if (ctx->label_classification_ctx) {
                    ret = ff_dnn_execute_model_clip(&ctx->dnnctx, in, NULL,
                                                    ctx->label_classification_ctx->labels,
                                                    ctx->label_classification_ctx->label_count,
                                                    ctx->tokenizer_path,
                                                    ctx->target
                    );
                } else if (ctx->category_classification_ctx) {
                    ret = execute_clip_model_for_all_categories(ctx, in);
                }
            } else {
                // Standard classification
                ret = ff_dnn_execute_model_classification(&ctx->dnnctx, in, NULL, ctx->target);
            }

            if (ret != 0) {
                av_frame_free(&in);
                return AVERROR(EIO);
            }
        }
    } while (ret > 0);

    // drain all processed frames
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

    // if frame got, schedule to next filter
    if (got_frame)
        return 0;

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            int64_t out_pts = pts;
            ret = dnn_classify_flush_frame(outlink, pts, &out_pts);
            ff_outlink_set_status(outlink, status, out_pts);
            return ret;
        }
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return 0;
}

static av_cold void dnn_classify_uninit(AVFilterContext *context) {
    DnnClassifyContext *ctx = context->priv;
    ff_dnn_uninit(&ctx->dnnctx);
    free_contexts(ctx);
}

const FFFilter ff_vf_dnn_classify = {
    .p.name = "dnn_classify",
    .p.description = NULL_IF_CONFIG_SMALL("Apply DNN classify filter to the input."),
    .p.priv_class = &dnn_classify_class,
    .priv_size = sizeof(DnnClassifyContext),
    .preinit = ff_dnn_filter_init_child_class,
    .init = dnn_classify_init,
    .uninit = dnn_classify_uninit,
    FILTER_INPUTS(ff_video_default_filterpad),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .activate = dnn_classify_activate,
};
