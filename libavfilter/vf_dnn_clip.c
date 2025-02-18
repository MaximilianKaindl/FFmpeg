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
#include "libavutil/clip_bbox.h"
#include "libavutil/avstring.h"

/*
    Labels that are being used to classify the image
*/
typedef struct ClassifyLabelContext {
    char **labels;
    int label_count;
} ClassifyLabelContext;

/*
    Header (Attribute) that is being described by all its labels. 
    (name) in labels file

    e.g. 
    (Comic)
    a drawn image
    a fictional character
    ...
*/
typedef struct ClassifyCategory {
    char *name;
    ClassifyLabelContext *labels;
    int label_count;
    float total_probability;
} ClassifyCategory;

/*
    Single unit that is being classified 
    [name] in categories file

    e.g.
    [RecordingSystem]
    (Professional)
    a photo with high level of detail 
    ...
*/
typedef struct ClassifyContext {
    char *name;
    ClassifyCategory *categories;
    int category_count;
    int label_count;
    int max_categories;
} ClassifyContext;

typedef struct DNNCLIPContext {
    const AVClass *clazz;
    DnnContext dnnctx;

    // used to store labels and categories if labels_filename is specified
    ClassifyLabelContext *label_ctx;

    // classify in categories
    // used to store labels and categories if categories_filename is specified
    ClassifyContext **classifcation_ctx;
    int num_contexts;
    int max_contexts;
    int total_labels;
    int total_categories;

    // Parameters to change Results of the simularity calculation
    float logit_scale;
    float temperature;

    char *labels_filename;
    char *categories_filename;
    char *tokenizer_path;
} DNNCLIPContext;

#define OFFSET(x) offsetof(DNNCLIPContext, dnnctx.x)
#define OFFSET2(x) offsetof(DNNCLIPContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM

static const AVOption dnn_clip_options[] = {
    { "dnn_backend", "DNN backend", 
        OFFSET(backend_type), AV_OPT_TYPE_INT, 
        { .i64 = DNN_TH }, INT_MIN, INT_MAX, FLAGS, .unit = "backend" },
#if (CONFIG_LIBTORCH == 1)
    { "torch", "torch backend flag", 
        0, AV_OPT_TYPE_CONST, { .i64 = DNN_TH }, 0, 0, FLAGS, .unit = "backend" },
#endif
    { "labels", "path to text prompts file", 
        OFFSET2(labels_filename), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
    {"categories", "path to categories and prompts file", 
        OFFSET2(categories_filename), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    { "logit_scale", "logit scale to scale logits", 
        OFFSET2(logit_scale), AV_OPT_TYPE_FLOAT, { .dbl = 4.6052 }, 0, 100.0, FLAGS },
    { "temperature", "softmax temperature", 
        OFFSET2(temperature), AV_OPT_TYPE_FLOAT, { .dbl = 1.0 }, 0, 100.0, FLAGS },
    { "tokenizer", "path to text tokenizer.json file", 
        OFFSET2(tokenizer_path), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
    { NULL }
};

AVFILTER_DNN_DEFINE_CLASS(dnn_clip, DNN_TH);

static int softmax(float *input, size_t input_len, float logit_scale, float temperature, AVFilterContext *ctx) 
{
    // Input validation
    if (!input) {
        av_log(ctx, AV_LOG_ERROR, "Invalid input pointer to softmax\n");
        return AVERROR(EINVAL);
    }

    if (input_len == 0) {
        av_log(ctx, AV_LOG_ERROR, "Invalid input length for softmax\n");
        return AVERROR(EINVAL);
    }

    // Ensure valid temperature
    if (temperature <= 0.0f) {
        temperature = 1.0f;
    }
    
    // Apply logit scale to inputs (in-place)
    for (size_t i = 0; i < input_len; i++) {
        input[i] *= logit_scale;
    }

    // Find maximum for numerical stability
    float m = input[0];
    for (size_t i = 1; i < input_len; i++) {
        if (input[i] > m) {
            m = input[i];
        }
    }

    // Calculate sum with temperature
    float sum = 0.0f;
    for (size_t i = 0; i < input_len; i++) {
        sum += expf((input[i] - m) / temperature);
    }

    if (sum == 0.0f) {
        av_log(ctx, AV_LOG_ERROR, "Division by zero in softmax\n");
        return AVERROR(EINVAL);
    }

    // Apply softmax with temperature
    float offset = m + temperature * logf(sum);
    for (size_t i = 0; i < input_len; i++) {
        input[i] = expf((input[i] - offset) / temperature);
    }

    return 0;
}

static int dnn_clip_post_proc_lables(AVFrame *frame, DNNData *output, uint32_t bbox_index, AVFilterContext *filter_ctx)
{
    DNNCLIPContext *ctx = filter_ctx->priv;
    const int max_classes_per_box = AV_CLIP_BBOX_CLASSES_MAX_COUNT;
    int num_labels = ctx->label_ctx->label_count;
    float *probabilities = (float *)output->data;
    softmax(probabilities,num_labels, ctx->logit_scale, ctx->temperature,filter_ctx);
    int num_bboxes;
    AVFrameSideData *sd;
    AVClipBBoxHeader *header;
    AVClipBBox *bbox;
    int i, j;
    int start_idx, end_idx;
    int percentage;

    // Calculate number of bounding boxes needed
    num_bboxes = (num_labels + max_classes_per_box - 1) / max_classes_per_box;

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_CLIP_BBOXES);
    if (sd != NULL) {
        av_log(filter_ctx, AV_LOG_ERROR, "Found existing Clip BBox. Box gets replaced ... \n");
        av_frame_remove_side_data(frame, AV_FRAME_DATA_CLIP_BBOXES);
    }

    header = av_clip_bbox_create_side_data(frame, num_bboxes);
    if (!header) {
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to allocate side data for clip classification\n");
        return AVERROR(ENOMEM);
    }

    if (bbox_index == 0) {
        av_strlcat(header->source, ", ", sizeof(header->source));
        av_strlcat(header->source, ctx->dnnctx.model_filename, sizeof(header->source));
    }

    //Process each bbox
    for (i = 0; i < num_bboxes; i++) {
        bbox = av_get_clip_bbox(header, i);
        if (!bbox) {
            av_log(filter_ctx, AV_LOG_ERROR, "Failed to get bbox %d\n", i);
            return AVERROR(EINVAL);
        }

        // Initialize bbox
        bbox->classify_count = 0;

        start_idx = i * max_classes_per_box;
        end_idx = FFMIN(num_labels, (i + 1) * max_classes_per_box);

        // Set classifications for this bbox
        for (j = start_idx; j < end_idx && bbox->classify_count < max_classes_per_box; j++) {
            if (!ctx->label_ctx->labels[j]) {
                av_log(filter_ctx, AV_LOG_ERROR, "Invalid label at index %d\n", j);
                continue;
            }

            percentage = (int)(probabilities[j] * 10000);
            bbox->classify_confidences[bbox->classify_count] = av_make_q(percentage, 10000);
            av_strlcpy(bbox->classify_labels[bbox->classify_count],
                       ctx->label_ctx->labels[j],
                       AV_CLIP_BBOX_LABEL_NAME_MAX_SIZE);
            bbox->classify_count++;
        }
    }

    return 0;
}

static int dnn_clip_post_proc_categories(AVFrame *frame, DNNData *output, uint32_t bbox_index, AVFilterContext *filter_ctx) 
{
    DNNCLIPContext *ctx = filter_ctx->priv;
    float *probabilities = (float *)output->data;
    AVFrameSideData *sd;
    AVClipBBoxHeader *header;
    AVClipBBox *bbox;
    int ret = 0;
    const int max_classes_per_box = AV_CLIP_BBOX_CLASSES_MAX_COUNT;
    
    // Create side data for this classification context
    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_CLIP_BBOXES);
    if (sd == NULL) {
        // Calculate total number of bboxes needed for all categories across all contexts
        int total_categories = 0;
        for(int c = 0; c < ctx->num_contexts; c++) {
            ClassifyContext *current_ctx = ctx->classifcation_ctx[c];
            if (current_ctx) {
                total_categories += current_ctx->category_count;
            }
        }
        
        // Calculate number of bboxes needed to hold all categories
        int num_bboxes = (total_categories + max_classes_per_box - 1) / max_classes_per_box;
        
        header = av_clip_bbox_create_side_data(frame, num_bboxes);
        if (!header) {
            av_log(filter_ctx, AV_LOG_ERROR, "Failed to allocate side data for clip classification\n");
            return AVERROR(ENOMEM);
        }
    } else {
        header = (AVClipBBoxHeader *)sd->data;
    }

    if (bbox_index == 0) {
        av_strlcat(header->source, ", ", sizeof(header->source));
        av_strlcat(header->source, ctx->dnnctx.model_filename, sizeof(header->source));
    }

    // Track the current bbox and class count within the current bbox
    int current_bbox_idx = 0;
    int current_class_count = 0;
    bbox = av_get_clip_bbox(header, current_bbox_idx);
    if (!bbox) {
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to get initial bbox\n");
        return AVERROR(EINVAL);
    }
    bbox->classify_count = 0;

    for(int c = 0; c < ctx->num_contexts; c++) {
        ClassifyContext *current_ctx = ctx->classifcation_ctx[c];
        if (!current_ctx) {
            av_log(filter_ctx, AV_LOG_ERROR, "Missing classification data at context %d\n", c);
            continue;
        }
        ClassifyCategory *best_category;
        float best_probability = -1.0f;
        int prob_offset = 0;
        // Calculate total probability for each category
        for (int cat_idx = 0; cat_idx < current_ctx->category_count; cat_idx++) {
            ClassifyCategory *category = &current_ctx->categories[cat_idx];

            // Apply softmax only to the labels within this category
            if (softmax(probabilities + prob_offset, 
                current_ctx->label_count,
                ctx->logit_scale, 
                ctx->temperature,
                filter_ctx) < 0) {
                return AVERROR(EINVAL);
            }

            // Sum probabilities for all labels in this category
            category->total_probability = 0.0f;
            for (int label_idx = 0; label_idx < category->label_count; label_idx++) {
                av_log(filter_ctx, AV_LOG_INFO, "prob = %f\n", probabilities[prob_offset + label_idx]);
                category->total_probability += probabilities[prob_offset + label_idx];
            }
            if(category->total_probability > best_probability) {
                best_probability = category->total_probability;
                best_category = category;
            }
            prob_offset += category->label_count;
        }
        
        // Check if we need to move to a new bbox
        if (current_class_count >= max_classes_per_box) {
            current_bbox_idx++;
            current_class_count = 0;
            bbox = av_get_clip_bbox(header, current_bbox_idx);
            if (!bbox) {
                av_log(filter_ctx, AV_LOG_ERROR, "Failed to get bbox %d\n", current_bbox_idx);
                return AVERROR(EINVAL);
            }
            bbox->classify_count = 0;
        }

        // Add category to current bbox
        int percentage = (int)(best_category->total_probability * 10000);
        bbox->classify_confidences[current_class_count] = av_make_q(percentage, 10000);
        av_strlcpy(bbox->classify_labels[current_class_count],
                    best_category->name,
                    AV_CLIP_BBOX_LABEL_NAME_MAX_SIZE);
        
        bbox->classify_count++;
        current_class_count++;
    }

    return ret;
}

static int dnn_clip_post_proc(AVFrame *frame, DNNData *output, uint32_t bbox_index, AVFilterContext *filter_ctx)
{
    DNNCLIPContext *ctx = filter_ctx->priv;

    if (!frame || !output || !output->data) {
        av_log(filter_ctx, AV_LOG_ERROR, "Invalid input to post processing\n");
        return AVERROR(EINVAL);
    }

    if (ctx->label_ctx) {
        return dnn_clip_post_proc_lables(frame, output, bbox_index, filter_ctx);
    }
    if (ctx->classifcation_ctx) {
        return dnn_clip_post_proc_categories(frame, output, bbox_index, filter_ctx);
    }
    av_log(filter_ctx, AV_LOG_ERROR, "No valid classification context available\n");
    return AVERROR(EINVAL);
}

static void free_label_context(ClassifyLabelContext *label_ctx)
{
    if (!label_ctx)
        return;

    if (label_ctx->labels) {
        for (int i = 0; i < label_ctx->label_count; i++) {
            av_freep(&label_ctx->labels[i]);
        }
        av_freep(&label_ctx->labels);
    }
    label_ctx->label_count = 0;
}

static void free_category(ClassifyCategory *category) 
{
    if (!category)
        return;

    av_freep(&category->name);
    if (category->labels) {
        free_label_context(category->labels);
        av_freep(&category->labels);
    }
}

static void free_classification_context(ClassifyContext *ctx) {
    if (!ctx)
        return;

    if (ctx->categories) {
        for (int i = 0; i < ctx->category_count; i++) {
            free_category(&ctx->categories[i]);
        }
        av_freep(&ctx->categories);
    }
    av_freep(&ctx->name);
    ctx->category_count = 0;
    ctx->max_categories = 0;
}

static void free_clip_contexts(DNNCLIPContext *ctx) {
    if (ctx->label_ctx) {
        free_label_context(ctx->label_ctx);
        av_freep(&ctx->label_ctx);
    }

    if (ctx->classifcation_ctx)
    {
        for (int i = 0; i < ctx->num_contexts; i++) {
            if (ctx->classifcation_ctx[i]) {
                free_classification_context(ctx->classifcation_ctx[i]);
                av_freep(&ctx->classifcation_ctx[i]);
            }
        }
        av_freep(&ctx->classifcation_ctx);
    }
    ctx->num_contexts = 0;
    ctx->max_contexts = 0;
}

static int read_classify_label_file(AVFilterContext *context)
{
    int line_len;
    FILE *file;
    DNNCLIPContext *ctx = context->priv;

    // Initialize the label context
    ctx->label_ctx = av_calloc(1, sizeof(ClassifyLabelContext));
    ctx->label_ctx->labels = NULL;
    ctx->label_ctx->label_count = 0;

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

        if (line_len > AV_CLIP_BBOX_LABEL_NAME_MAX_SIZE) {
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

        if (av_dynarray_add_nofree(&ctx->label_ctx->labels, &ctx->label_ctx->label_count, prompt) < 0) {
            av_log(context, AV_LOG_ERROR, "Failed to add prompt to array\n");
            fclose(file);
            av_freep(&prompt);
            return AVERROR(ENOMEM);
        }
    }

    fclose(file);
    return 0;
}

static int read_categories_file(AVFilterContext *context)
{
    DNNCLIPContext *ctx = context->priv;
    FILE *file;
    char buf[256];
    int ret = 0;
    ClassifyContext *current_ctx = NULL;
    ClassifyCategory *current_category = NULL;

    file = avpriv_fopen_utf8(ctx->categories_filename, "r");
    if (!file)
    {
        av_log(context, AV_LOG_ERROR, "Failed to open categories file %s\n", ctx->categories_filename);
        return AVERROR(EINVAL);
    }

    // Initialize contexts array
    ctx->max_contexts = 10; // Initial size
    ctx->num_contexts = 0;
    ctx->classifcation_ctx = av_calloc(ctx->max_contexts, sizeof(ClassifyContext *));
    if (!ctx->classifcation_ctx)
    {
        fclose(file);
        return AVERROR(ENOMEM);
    }

    while (fgets(buf, sizeof(buf), file))
    {
        char *line = buf;
        int line_len = strlen(line);

        // Trim whitespace and newlines
        while (line_len > 0 && (line[line_len - 1] == '\n' ||
                                line[line_len - 1] == '\r' ||
                                line[line_len - 1] == ' '))
        {
            line[--line_len] = '\0';
        }

        if (line_len == 0)
            continue;

        // Check for context marker [ContextName]
        if (line[0] == '[' && line[line_len - 1] == ']')
        {
            if (current_ctx != NULL)
            {
                // Store the previous context
                if (ctx->num_contexts >= ctx->max_contexts)
                {
                    int new_size = ctx->max_contexts * 2;
                    ClassifyContext **new_contexts = av_realloc_array(
                        ctx->classifcation_ctx,
                        new_size,
                        sizeof(ClassifyContext *));
                    if (!new_contexts)
                    {
                        ret = AVERROR(ENOMEM);
                        goto end;
                    }
                    ctx->classifcation_ctx = new_contexts;
                    ctx->max_contexts = new_size;
                }
                ctx->classifcation_ctx[ctx->num_contexts++] = current_ctx;
            }

            // Create new classification context
            current_ctx = av_calloc(1, sizeof(ClassifyContext));
            if (!current_ctx)
            {
                ret = AVERROR(ENOMEM);
                goto end;
            }

            // Extract context name
            line[line_len - 1] = '\0';
            current_ctx->max_categories = 10; // Default max categories
            current_ctx->name = av_strdup(line + 1);
            if (!current_ctx->name)
            {
                av_freep(&current_ctx);
                ret = AVERROR(ENOMEM);
                goto end;
            }

            current_ctx->category_count = 0;
            current_ctx->categories = av_calloc(current_ctx->max_categories, sizeof(ClassifyCategory));
            if (!current_ctx->categories)
            {
                av_freep(&current_ctx->name);
                av_freep(&current_ctx);
                ret = AVERROR(ENOMEM);
                goto end;
            }

            current_category = NULL;
        }
        // Check for category marker (CategoryName)
        else if (line[0] == '(' && line[line_len - 1] == ')')
        {
            if (!current_ctx)
            {
                av_log(context, AV_LOG_ERROR, "Category found without context\n");
                ret = AVERROR(EINVAL);
                goto end;
            }

            // Check if we need to resize categories array
            if (current_ctx->category_count >= current_ctx->max_categories)
            {
                int new_size = current_ctx->max_categories * 2;
                ClassifyCategory *new_categories = av_realloc_array(current_ctx->categories,
                                                                          new_size, sizeof(ClassifyCategory));
                if (!new_categories)
                {
                    ret = AVERROR(ENOMEM);
                    goto end;
                }
                current_ctx->categories = new_categories;
                current_ctx->max_categories = new_size;
            }

            // Extract category name
            line[line_len - 1] = '\0';
            current_category = &current_ctx->categories[current_ctx->category_count++];
            ctx->total_categories++;
            current_category->label_count = 0;
            memset(current_category, 0, sizeof(ClassifyCategory));

            current_category->name = av_strdup(line + 1);
            if (!current_category->name)
            {
                ret = AVERROR(ENOMEM);
                goto end;
            }

            current_category->labels = av_calloc(1, sizeof(ClassifyLabelContext));
            if (!current_category->labels)
            {
                av_freep(&current_category->name);
                ret = AVERROR(ENOMEM);
                goto end;
            }
            current_category->total_probability = 0.0f;
        }
        // Must be a label
        else if (line[0] != '\0' && current_category)
        {
            char *label = av_strdup(line);
            if (!label)
            {
                ret = AVERROR(ENOMEM);
                goto end;
            }

            char **new_labels = av_realloc_array(current_category->labels->labels,
                                                 current_category->labels->label_count + 1,
                                                 sizeof(char *));
            if (!new_labels)
            {
                av_freep(&label);
                ret = AVERROR(ENOMEM);
                goto end;
            }

            current_category->labels->labels = new_labels;
            current_category->label_count++;
            current_category->labels->labels[current_category->labels->label_count++] = label;
            current_ctx->label_count++;
            ctx->total_labels++;
        }
    }

    // Store the last context if it exists
    if (current_ctx)
    {
        if (ctx->num_contexts >= ctx->max_contexts)
        {
            int new_size = ctx->max_contexts * 2;
            ClassifyContext **new_contexts = av_realloc_array(
                ctx->classifcation_ctx,
                new_size,
                sizeof(ClassifyContext *));
            if (!new_contexts)
            {
                ret = AVERROR(ENOMEM);
                goto end;
            }
            ctx->classifcation_ctx = new_contexts;
            ctx->max_contexts = new_size;
        }
        ctx->classifcation_ctx[ctx->num_contexts++] = current_ctx;
    }

end:
    if (ret < 0)
    {
        // Clean up current context if it wasn't added to the array
        if (current_ctx)
        {
            free_classification_context(current_ctx);
            av_freep(&current_ctx);
        }

        // Clean up all stored contexts
        for (int i = 0; i < ctx->num_contexts; i++)
        {
            if (ctx->classifcation_ctx[i])
            {
                free_classification_context(ctx->classifcation_ctx[i]);
                av_freep(&ctx->classifcation_ctx[i]);
            }
        }
        av_freep(&ctx->classifcation_ctx);
        ctx->num_contexts = 0;
        ctx->max_contexts = 0;
    }

    fclose(file);
    return ret;
}

static av_cold int dnn_clip_init(AVFilterContext *context)
{
    DNNCLIPContext *ctx = context->priv;
    int ret;

    ret = ff_dnn_init(&ctx->dnnctx, DFT_ANALYTICS_CLIP, context);
    if (ret < 0)
        return ret;
    ff_dnn_set_classify_post_proc(&ctx->dnnctx, dnn_clip_post_proc);

    if (ctx->labels_filename && ctx->categories_filename) {
        av_log(context, AV_LOG_ERROR, "Labels and categories file cannot be used together\n");
        return AVERROR(EINVAL);
    }

    if (ctx->labels_filename) {
        ret = read_classify_label_file(context);
        if (ret < 0) {
            av_log(context, AV_LOG_ERROR, "Failed to read labels file\n");
            return ret;
        }
    }
    else if (ctx->categories_filename) {
        ret = read_categories_file(context);
        if (ret < 0) {
            av_log(context, AV_LOG_ERROR, "Failed to read categories file\n");
            return ret;
        }
    }
    else {
        av_log(context, AV_LOG_ERROR, "Labels or categories file is required for CLIP classification\n");
        return AVERROR(EINVAL);
    }

    if (!ctx->tokenizer_path) {
        av_log(context, AV_LOG_ERROR, "Tokenizer file is required for CLIP classification\n");
        return AVERROR(EINVAL);
    }
    return 0;
}

static av_cold void dnn_clip_uninit(AVFilterContext *context)
{
    DNNCLIPContext *ctx = context->priv;
    ff_dnn_uninit(&ctx->dnnctx);
    free_clip_contexts(ctx);
}

static int dnn_clip_flush_frame(AVFilterLink *outlink, int64_t pts, int64_t *out_pts)
{
    DNNCLIPContext *ctx = outlink->src->priv;
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

static int execute_clip_model_for_all_categories(DNNCLIPContext *ctx, AVFrame *frame) {
    char **combined_labels = NULL;
    int combined_idx = 0;
    int ret;

    // Allocate array for all labels
    combined_labels = av_calloc(ctx->total_labels, sizeof(char *));
    if (!combined_labels) {
        return AVERROR(ENOMEM);
    }
    for(int c = 0; c < ctx->num_contexts; c++) {
        ClassifyContext *current_ctx = ctx->classifcation_ctx[c];
        for (int i = 0; i < current_ctx->category_count; i++) {
            ClassifyCategory *category = &current_ctx->categories[i];
            for (int j = 0; j < category->labels->label_count; j++) {
                combined_labels[combined_idx] = category->labels->labels[j];
                combined_idx++;
            }
        }    
    }
    // Execute model with ALL labels combined
    ret = ff_dnn_execute_model_clip(&ctx->dnnctx, frame, NULL,
        combined_labels,
        ctx->tokenizer_path,
        ctx->total_labels);

    av_freep(&combined_labels);
    return ret;
}

static int dnn_clip_activate(AVFilterContext *filter_ctx)
{
    AVFilterLink *inlink = filter_ctx->inputs[0];
    AVFilterLink *outlink = filter_ctx->outputs[0];
    DNNCLIPContext *ctx = filter_ctx->priv;
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
            if (ctx->label_ctx) {
                // Simple label context case
                ret = ff_dnn_execute_model_clip(&ctx->dnnctx, in, NULL,
                                                ctx->label_ctx->labels,
                                                ctx->tokenizer_path,
                                                ctx->label_ctx->label_count);
                if (ret != 0) {
                    av_frame_free(&in);
                    return AVERROR(EIO);
                }
            } else if (ctx->classifcation_ctx){
                // Process all classification context
                ret = execute_clip_model_for_all_categories(ctx, in);
                if (ret != 0)
                {
                    av_frame_free(&in);
                    return ret;
                }
            } else {
                av_log(filter_ctx, AV_LOG_ERROR,
                       "No label context or classification contexts available\n");
                av_frame_free(&in);
                return AVERROR(EINVAL);
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
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUVJ420P,
    AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE
};

const FFFilter ff_vf_dnn_clip = {
    .p.name          = "dnn_clip",
    .p.description   = NULL_IF_CONFIG_SMALL("Apply CLIP zero-shot classification."),
    .p.priv_class    = &dnn_clip_class,
    .preinit       = ff_dnn_filter_init_child_class,
    .priv_size     = sizeof(DNNCLIPContext),
    .init          = dnn_clip_init,
    .uninit        = dnn_clip_uninit,
    .activate      = dnn_clip_activate,  
    FILTER_INPUTS(ff_video_default_filterpad),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
};