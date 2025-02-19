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
 
/*
    Labels that are being used to classify the image
*/
 typedef struct LabelContext {
     char **labels;
     int label_count;
 } LabelContext;
 
 /*
    Header (Attribute) that is being described by all its labels. 
    (name) in labels file

    e.g. 
    (Comic)
    a drawn image
    a fictional character
    ...

    Can also be used to substitute the labeltext with the header text
    so that the header is displayed instead of the label with the probability
*/
 typedef struct CategoryContext {
     char *name;
     LabelContext *labels;
     int label_count;
     float total_probability;
 } CategoryContext;
 
 /*
    Single unit that is being classified 
    [name] in categories file

    e.g.
    [RecordingSystem]
    (Professional)
    a photo with high level of detail 
    ...
    (HomeRecording)
    a photo with low level of detail
    ...

    empowers to do multiple classification "runs"
*/
 typedef struct CategoriesContext {
     char *name;
     CategoryContext *categories;
     int category_count;
     int label_count;
     int max_categories;
 } CategoriesContext;

 typedef struct CategoryClassifcationContext {
    CategoriesContext **category_units;
    int num_contexts;
    int max_contexts;
    int total_labels;
    int total_categories;
 } CategoryClassifcationContext;
 
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
    { "dnn_backend", "DNN backend",                OFFSET(backend_type),     AV_OPT_TYPE_INT,       { .i64 = DNN_OV },    INT_MIN, INT_MAX, FLAGS, .unit = "backend" },
 #if (CONFIG_LIBOPENVINO == 1)
    { "openvino",    "openvino backend flag",      0,                        AV_OPT_TYPE_CONST,     { .i64 = DNN_OV },    0, 0, FLAGS, .unit = "backend" },
 #endif
 #if (CONFIG_LIBTORCH == 1)
     { "torch", "torch backend flag", 0, AV_OPT_TYPE_CONST, { .i64 = DNN_TH }, 0, 0, FLAGS, .unit = "backend" },
 #endif
    { "confidence",  "threshold of confidence",    OFFSET2(confidence),      AV_OPT_TYPE_FLOAT,     { .dbl = 0.5 },  0, 1, FLAGS},
    { "labels",      "path to labels file",        OFFSET2(labels_filename), AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },
    { "target",      "which one to be classified", OFFSET2(target),          AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },
 // CLIP-specific options
     { "categories", "path to categories file (CLIP only)", OFFSET2(categories_filename), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
     { "logit_scale", "logit scale for CLIP", OFFSET2(logit_scale), AV_OPT_TYPE_FLOAT, { .dbl = 4.6052 }, 0, 100.0, FLAGS },
     { "temperature", "softmax temperature for CLIP", OFFSET2(temperature), AV_OPT_TYPE_FLOAT, { .dbl = 1.0 }, 0, 100.0, FLAGS },
     { "tokenizer", "path to text tokenizer.json file (CLIP only)", OFFSET2(tokenizer_path), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
     { NULL }
 };
 
 AVFILTER_DNN_DEFINE_CLASS(dnn_classify, DNN_OV);
 

static int dnn_classify_set_prob_and_label_of_bbox(AVDetectionBBox **bbox, char *label, int index, float probability){
    (*bbox)->classify_confidences[index] = av_make_q((int)(probability * 10000), 10000);
    av_strlcpy((*bbox)->classify_labels[index],
                label,
                AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE);
    return 0;
} 
static int dnn_classify_post_proc_standard(AVFrame *frame, DNNData *output, uint32_t bbox_index, AVFilterContext *filter_ctx)
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
        av_log(filter_ctx, AV_LOG_ERROR, "Cannot get side data in dnn_classify_post_proc_standard\n");
         return -1;
     }
     header = (AVDetectionBBoxHeader *)sd->data;
 
     if (bbox_index == 0) {
         av_strlcat(header->source, ", ", sizeof(header->source));
         av_strlcat(header->source, ctx->dnnctx.model_filename, sizeof(header->source));
     }
 
     classifications = output->data;
     label_id = 0;
     confidence= classifications[0];
     for (int i = 1; i < output_size; i++) {
         if (classifications[i] > confidence) {
             label_id = i;
             confidence= classifications[i];
         }
     }
 
     if (confidence < conf_threshold) {
         return 0;
     }
 
     bbox = av_get_detection_bbox(header, bbox_index);
     bbox->classify_confidences[bbox->classify_count] = av_make_q((int)(confidence * 10000), 10000);
 
     if (ctx->label_classification_ctx->labels && label_id < ctx->label_classification_ctx->label_count) {
         av_strlcpy(bbox->classify_labels[bbox->classify_count], ctx->label_classification_ctx->labels[label_id], sizeof(bbox->classify_labels[bbox->classify_count]));
     } else {
         snprintf(bbox->classify_labels[bbox->classify_count], sizeof(bbox->classify_labels[bbox->classify_count]), "%d", label_id);
     }
 
     bbox->classify_count++;
 
     return 0;
 }
 
static int dnn_classify_softmax(float *input, size_t input_len, float logit_scale, float temperature, AVFilterContext *ctx) 
{
    float sum,offset,m;

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

static AVDetectionBBoxHeader *dnn_classify_find_or_create_detection_bbox(AVFrame *frame, uint32_t *bbox_index, int *new, AVFilterContext *filter_ctx){
    DnnClassifyContext *ctx = filter_ctx->priv;
    AVFrameSideData *sd;
    AVDetectionBBoxHeader *header;

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DETECTION_BBOXES);
    if (!sd) {
        header = av_detection_bbox_create_side_data(frame, 1);
        if (!header) {
            av_log(filter_ctx, AV_LOG_ERROR, "Cannot get side data in CLIP labels processing\n");
            return AVERROR(EINVAL);
        }
        //ensure if created index is 0
        *bbox_index = 0;
        *new = 1;
    }
    else{
        header = (AVDetectionBBoxHeader *)sd->data;
    }

    if (*bbox_index == 0) {
        av_strlcat(header->source, ", ", sizeof(header->source));
        av_strlcat(header->source, ctx->dnnctx.model_filename, sizeof(header->source));
    }
    return header;
}

static int dnn_classify_fill_bbox_with_best_labels(DnnClassifyContext *ctx, char **labels, float** probabilities, int num_labels, AVDetectionBBox *bbox, int max_classes_per_box, float confidence_threshold){
    int i, j, minpos;
    float min;

    for (i = 0; i < num_labels; i++) {
        if ((*probabilities)[i] >= confidence_threshold) {
            if (bbox->classify_count >= max_classes_per_box) {
                // Find the classification with lowest probability to potentially substitute
                min = (*probabilities)[0];
                minpos = 0;
                for (j = 1; j < bbox->classify_count; j++) {
                    if (av_q2d(bbox->classify_confidences[j]) < min) {
                        min = av_q2d(bbox->classify_confidences[j]);
                        minpos = j;
                    }
                }
                
                // If current probability is higher than the minimum, substitute it
                if ((*probabilities)[i] > min) {
                    dnn_classify_set_prob_and_label_of_bbox(&bbox, 
                        ctx->label_classification_ctx->labels[i], minpos,
                        (*probabilities)[i]);
                    // Note: classify_count doesn't change as we're substituting
                }
            } else {
                // Add new classification
                dnn_classify_set_prob_and_label_of_bbox(&bbox, 
                    ctx->label_classification_ctx->labels[i], i,
                    (*probabilities)[i]);
                bbox->classify_count++;
            }
        }
    }
    return 0;
}

static int dnn_classify_post_proc_clip_labels(AVFrame *frame, DNNData *output, uint32_t bbox_index, AVFilterContext *filter_ctx)
{
    DnnClassifyContext *ctx = filter_ctx->priv;
    const int max_classes_per_box = AV_NUM_DETECTION_BBOX_CLASSIFY;
    float *probabilities = (float *)output->data;
    int num_labels = ctx->label_classification_ctx->label_count;
    AVDetectionBBoxHeader *header;
    AVDetectionBBox *bbox;
    float confidence_threshold = ctx->confidence;
    int ret, new = 0;

    // Apply softmax to probabilities
    if (dnn_classify_softmax(probabilities, num_labels, ctx->logit_scale, ctx->temperature, filter_ctx) < 0) {
        return AVERROR(EINVAL);
    }

    // Get or create detection bbox header
    header = dnn_classify_find_or_create_detection_bbox(frame, &bbox_index, &new, filter_ctx);
    if (!header) {
        return AVERROR(EINVAL);
    }

    // Get bbox for current index
    bbox = av_get_detection_bbox(header, bbox_index);
    if (!bbox) {
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to get bbox %d\n", bbox_index);
        return AVERROR(EINVAL);
    }

    // Initialize classify count if newly created
    if(new == 1){
        bbox->classify_count = 0;
    }

    ret = dnn_classify_fill_bbox_with_best_labels(ctx, ctx->label_classification_ctx->labels, &probabilities, num_labels, bbox, max_classes_per_box, confidence_threshold);
    if(ret < 0){
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to fill bbox with best labels\n");
        return ret;
    }
    return 0;
}

static int dnn_classify_apply_softmax_over_all_categories(DnnClassifyContext *ctx , AVFilterContext *filter_ctx, float* probabilities){
    int prob_offset = 0;
    CategoryClassifcationContext *cat_class_ctx = ctx->category_classification_ctx;

    for(int c = 0; c < cat_class_ctx->num_contexts; c++) {
        CategoriesContext *categories_ctx = cat_class_ctx->category_units[c];
        if (!categories_ctx) {
            av_log(filter_ctx, AV_LOG_ERROR, "Missing classification data at context %d\n", c);
            continue;
        }
        // Apply softmax only to the labels within this category
        if (dnn_classify_softmax(probabilities + prob_offset,
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

static CategoryContext *dnn_classify_get_best_category(CategoriesContext *categories_ctx, float** probabilities){
    CategoryContext *best_category;
    float best_probability = -1.0f;
    int prob_offset = 0;
    // Calculate total probability for each category
    for (int cat_idx = 0; cat_idx < categories_ctx->category_count; cat_idx++) {
        CategoryContext *category = &categories_ctx->categories[cat_idx];
        // Sum probabilities for all labels in this category
        category->total_probability = 0.0f;
        for (int label_idx = 0; label_idx < category->label_count; label_idx++) {
            category->total_probability += (*probabilities)[prob_offset + label_idx];
        }
        if(category->total_probability > best_probability) {
            best_probability = category->total_probability;
            best_category = category;
        }
        prob_offset += category->label_count;
    }
    return best_category;
}

static int dnn_classify_post_proc_clip_categories(AVFrame *frame, DNNData *output, uint32_t bbox_index, AVFilterContext *filter_ctx)
{
    DnnClassifyContext *ctx = filter_ctx->priv;
    CategoryClassifcationContext *cat_class_ctx;
    AVDetectionBBoxHeader *header;
    AVDetectionBBox *bbox;
    CategoryContext *best_category;
    float *probabilities;
    int ret, new = 0, prob_offset = 0;
    char *ctx_labels;
    float *ctx_probabilities;

    // Input validation
    if (!frame || !output || !output->data || !ctx->category_classification_ctx) {
        av_log(filter_ctx, AV_LOG_ERROR, "Invalid input to CLIP categories post-processing\n");
        return AVERROR(EINVAL);
    }

    cat_class_ctx = ctx->category_classification_ctx;
    probabilities = output->data;

    // Apply softmax transformation to probabilities
    ret = dnn_classify_apply_softmax_over_all_categories(ctx, filter_ctx, probabilities);
    if (ret < 0) {
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to apply softmax transformation\n");
        return ret;
    }

    // Get or create detection bbox header
    header = dnn_classify_find_or_create_detection_bbox(frame, &bbox_index, &new, filter_ctx);
    if (!header) {
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to get or create detection bbox header\n");
        return AVERROR(EINVAL);
    }

    // Get bbox for current index
    bbox = av_get_detection_bbox(header, bbox_index);
    if (!bbox) {
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to get bbox at index %d\n", bbox_index);
        return AVERROR(EINVAL);
    }

    // Initialize classify count if newly created
    if(new == 1){
        bbox->classify_count = 0;
    }

    // Get each best Category of each context
    ctx_labels = (char *)av_mallocz(cat_class_ctx->num_contexts * sizeof(char));
    ctx_probabilities = (float *)av_mallocz(cat_class_ctx->num_contexts * sizeof(float));

    for (int ctx_idx = 0; ctx_idx < cat_class_ctx->num_contexts; ctx_idx++) {
        CategoriesContext *categories_ctx = cat_class_ctx->category_units[ctx_idx];
        if (!categories_ctx) {
            av_log(filter_ctx, AV_LOG_ERROR, "Missing classification data at context %d\n", ctx_idx);
            continue;
        }

        // Find best category within this context
        best_category = dnn_classify_get_best_category(categories_ctx, 
                                                                      (&ctx_probabilities + prob_offset));
        if (!best_category) {
            av_log(filter_ctx, AV_LOG_ERROR, "Failed to determine best category for context %d\n", ctx_idx);
            return AVERROR(EINVAL);
        }
        ctx_labels[ctx_idx] = *best_category->name;
        ctx_probabilities[ctx_idx] = best_category->total_probability;
        // Update probability offset for next context
        prob_offset += categories_ctx->label_count;
    }
    ret = dnn_classify_fill_bbox_with_best_labels(ctx, &ctx_labels, &ctx_probabilities, cat_class_ctx->num_contexts, bbox, AV_NUM_DETECTION_BBOX_CLASSIFY, ctx->confidence); 
    if (ret < 0) {
        av_log(filter_ctx, AV_LOG_ERROR, "Failed to fill bbox with best labels\n");
        return ret;
    }   

    av_freep(&ctx_labels);
    av_freep(&ctx_probabilities);

    return 0;
}
 
 static int dnn_classify_post_proc(AVFrame *frame, DNNData *output, uint32_t bbox_index, AVFilterContext *filter_ctx)
 {
     DnnClassifyContext *ctx = filter_ctx->priv;
     if (!frame || !output || !output->data) {
        av_log(filter_ctx, AV_LOG_ERROR, "Invalid input to CLIP post processing\n");
        return AVERROR(EINVAL);
    }

     // Choose post-processing based on backend
     if (ctx->dnnctx.backend_type == DNN_TH) {
         if (ctx->label_classification_ctx) {
             return dnn_classify_post_proc_clip_labels(frame, output, bbox_index, filter_ctx);
         } else if (ctx->category_classification_ctx) {
             return dnn_classify_post_proc_clip_categories(frame, output, bbox_index, filter_ctx);
         }
         av_log(filter_ctx, AV_LOG_ERROR, "No valid CLIP classification context available\n");
         return AVERROR(EINVAL);
     } else {
         return dnn_classify_post_proc_standard(frame, output, bbox_index, filter_ctx);
     }
 }
 
 static void free_label_context(LabelContext *label_classification_ctx)
 {
     if (!label_classification_ctx)
         return;
 
     if (label_classification_ctx->labels) {
         for (int i = 0; i < label_classification_ctx->label_count; i++) {
             av_freep(&label_classification_ctx->labels[i]);
         }
         av_freep(&label_classification_ctx->labels);
     }
     label_classification_ctx->label_count = 0;
     av_freep(label_classification_ctx);
 }
 
 static void free_category_context(CategoryContext *category)
{
    if (!category)
        return;

    if (category->name) {
        av_freep(&category->name);
    }
    
    if (category->labels) {
        free_label_context(category->labels);
        category->labels = NULL;  
    }
}

static void free_categories_context(CategoriesContext *ctx)
{
    if (!ctx)
        return;

    if (ctx->categories) {
        for (int i = 0; i < ctx->category_count; i++) {
            free_category_context(&ctx->categories[i]);
        }
        // Now free the array of categories
        av_freep(&ctx->categories);
    }
    
    if (ctx->name) {
        av_freep(&ctx->name);
    }
    
    ctx->category_count = 0;
    ctx->max_categories = 0;
    ctx->label_count = 0;
}

static void free_contexts(DnnClassifyContext *ctx)
{
    if (!ctx)
        return;

    if (ctx->label_classification_ctx) {
        free_label_context(ctx->label_classification_ctx);
        ctx->label_classification_ctx = NULL;
    }

    if (ctx->category_classification_ctx) {
        CategoryClassifcationContext *category_classification_ctx = ctx->category_classification_ctx;
        if (category_classification_ctx) {
            if (category_classification_ctx->category_units) {
                for (int i = 0; i < category_classification_ctx->num_contexts; i++) {
                    if (category_classification_ctx->category_units[i]) {
                        free_categories_context(category_classification_ctx->category_units[i]);
                        av_freep(&category_classification_ctx->category_units[i]);
                    }
                }
                av_freep(&category_classification_ctx->category_units);
            }
            category_classification_ctx->num_contexts = 0;
            category_classification_ctx->max_contexts = 0;
            av_freep(&ctx->category_classification_ctx);
        }
    }
}

static int read_classify_label_file(AVFilterContext *context)
{
    int line_len;
    FILE *file;
    DnnClassifyContext *ctx = context->priv;

    file = avpriv_fopen_utf8(ctx->labels_filename, "r");
    if (!file){
        av_log(context, AV_LOG_ERROR, "failed to open file %s\n", ctx->labels_filename);
        return AVERROR(EINVAL);
    }

    while (!feof(file)) {
        char *label;
        char buf[256];
        if (!fgets(buf, 256, file)) {
            break;
        }

        line_len = strlen(buf);
        while (line_len) {
            int i = line_len - 1;
            if (buf[i] == '\n' || buf[i] == '\r' || buf[i] == ' ') {
                buf[i] = '\0';
                line_len--;
            } else {
                break;
            }
        }

        if (line_len == 0)  // empty line
            continue;

        if (line_len >= AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE) {
            av_log(context, AV_LOG_ERROR, "label %s too long\n", buf);
            fclose(file);
            return AVERROR(EINVAL);
        }

        label = av_strdup(buf);
        if (!label) {
            av_log(context, AV_LOG_ERROR, "failed to allocate memory for label %s\n", buf);
            fclose(file);
            return AVERROR(ENOMEM);
        }

        if (av_dynarray_add_nofree(&ctx->label_classification_ctx->labels, &ctx->label_classification_ctx->label_count, label) < 0) {
            av_log(context, AV_LOG_ERROR, "failed to do av_dynarray_add\n");
            fclose(file);
            av_freep(&label);
            return AVERROR(ENOMEM);
        }
    }

    fclose(file);
    return 0;
}

static int read_categories_file(AVFilterContext *context)
{
    DnnClassifyContext *ctx = context->priv;
    FILE *file;
    char buf[256];
    int ret = 0;
    CategoriesContext *current_ctx = NULL;
    CategoryContext *current_category = NULL;
    CategoryClassifcationContext *cat_class_ctx = ctx->category_classification_ctx;

    file = avpriv_fopen_utf8(ctx->categories_filename, "r");
    if (!file) {
        av_log(context, AV_LOG_ERROR, "Failed to open categories file %s\n", ctx->categories_filename);
        return AVERROR(EINVAL);
    }

    // Initialize contexts array
    cat_class_ctx->max_contexts = 10;
    cat_class_ctx->num_contexts = 0;
    cat_class_ctx->category_units = av_calloc(cat_class_ctx->max_contexts, sizeof(CategoriesContext *));
    if (!cat_class_ctx->category_units) {
        fclose(file);
        return AVERROR(ENOMEM);
    }

    while (fgets(buf, sizeof(buf), file)) {
        char *line = buf;
        int line_len = strlen(line);

        // Trim whitespace and newlines
        while (line_len > 0 && (line[line_len - 1] == '\n' ||
                               line[line_len - 1] == '\r' ||
                               line[line_len - 1] == ' ')) {
            line[--line_len] = '\0';
        }

        if (line_len == 0)
            continue;

        if (line_len >= AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE) {
            av_log(context, AV_LOG_ERROR, "Label %s too long\n", buf);
            ret = AVERROR(ENOMEM);
            goto end;
        }

        // Check for context marker [ContextName]
        if (line[0] == '[' && line[line_len - 1] == ']') {
            if (current_ctx != NULL) {
                // Store previous context
                if (cat_class_ctx->num_contexts >= cat_class_ctx->max_contexts) {
                    int new_size = cat_class_ctx->max_contexts * 2;
                    CategoriesContext **new_contexts = av_realloc_array(
                        cat_class_ctx->category_units,
                        new_size,
                        sizeof(CategoriesContext *));
                    if (!new_contexts) {
                        ret = AVERROR(ENOMEM);
                        goto end;
                    }
                    cat_class_ctx->category_units = new_contexts;
                    cat_class_ctx->max_contexts = new_size;
                }
                cat_class_ctx->category_units[cat_class_ctx->num_contexts++] = current_ctx;
            }

            // Create new context
            current_ctx = av_calloc(1, sizeof(CategoriesContext));
            if (!current_ctx) {
                ret = AVERROR(ENOMEM);
                goto end;
            }

            // Extract context name
            line[line_len - 1] = '\0';
            current_ctx->name = av_strdup(line + 1);
            if (!current_ctx->name) {
                av_freep(&current_ctx);
                ret = AVERROR(ENOMEM);
                goto end;
            }

            current_ctx->category_count = 0;
            current_ctx->max_categories = 10;
            current_ctx->categories = av_calloc(current_ctx->max_categories, sizeof(CategoryContext));
            if (!current_ctx->categories) {
                av_freep(&current_ctx->name);
                av_freep(&current_ctx);
                ret = AVERROR(ENOMEM);
                goto end;
            }

            current_category = NULL;
        }
        // Check for category marker (CategoryName)
        else if (line[0] == '(' && line[line_len - 1] == ')') {
            if (!current_ctx) {
                av_log(context, AV_LOG_ERROR, "Category found without context\n");
                ret = AVERROR(EINVAL);
                goto end;
            }

            if (current_ctx->category_count >= current_ctx->max_categories) {
                int new_size = current_ctx->max_categories * 2;
                CategoryContext *new_categories = av_realloc_array(
                    current_ctx->categories,
                    new_size,
                    sizeof(CategoryContext));
                if (!new_categories) {
                    ret = AVERROR(ENOMEM);
                    goto end;
                }
                current_ctx->categories = new_categories;
                current_ctx->max_categories = new_size;
            }

            line[line_len - 1] = '\0';
            current_category = &current_ctx->categories[current_ctx->category_count++];
            cat_class_ctx->total_categories++;

            current_category->name = av_strdup(line + 1);
            if (!current_category->name) {
                ret = AVERROR(ENOMEM);
                goto end;
            }

            current_category->labels = av_calloc(1, sizeof(LabelContext));
            if (!current_category->labels) {
                av_freep(&current_category->name);
                ret = AVERROR(ENOMEM);
                goto end;
            }
            current_category->label_count = 0;
            current_category->total_probability = 0.0f;
        }
        // Must be a label
        else if (line[0] != '\0' && current_category) {
            char *label = av_strdup(line);
            if (!label) {
                ret = AVERROR(ENOMEM);
                goto end;
            }

            if (av_dynarray_add_nofree(&current_category->labels->labels,
                                      &current_category->labels->label_count,
                                      label) < 0) {
                av_freep(&label);
                ret = AVERROR(ENOMEM);
                goto end;
            }
            
            current_category->label_count++;
            current_ctx->label_count++;
            cat_class_ctx->total_labels++;
        }
    }

    // Store the last context
    if (current_ctx) {
        if (cat_class_ctx->num_contexts >= cat_class_ctx->max_contexts) {
            int new_size = cat_class_ctx->max_contexts * 2;
            CategoriesContext **new_contexts = av_realloc_array(
                cat_class_ctx->category_units,
                new_size,
                sizeof(CategoriesContext *));
            if (!new_contexts) {
                ret = AVERROR(ENOMEM);
                goto end;
            }
            cat_class_ctx->category_units = new_contexts;
            cat_class_ctx->max_contexts = new_size;
        }
        cat_class_ctx->category_units[cat_class_ctx->num_contexts++] = current_ctx;
    }

end:
    if (ret < 0)
    {
        // Clean up current context if it wasn't added to the array
        if (current_ctx)
        {
            free_categories_context(current_ctx);
        }

        free_contexts(ctx);
    }

    fclose(file);
    return ret;
}

static av_cold int dnn_classify_init(AVFilterContext *context)
{
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
        ret = read_classify_label_file(context);
        if (ret < 0) {
            av_log(context, AV_LOG_ERROR, "Failed to read labels file\n");
            return ret;
        }
    }
    else if (ctx->categories_filename) {
        ctx->category_classification_ctx = av_calloc(1, sizeof(CategoryClassifcationContext));
        if (!ctx->category_classification_ctx)
            return AVERROR(ENOMEM);

        ret = read_categories_file(context);
        if (ret < 0) {
            av_log(context, AV_LOG_ERROR, "Failed to read categories file\n");
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

static int dnn_classify_flush_frame(AVFilterLink *outlink, int64_t pts, int64_t *out_pts)
{
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
    for(int c = 0; c < cat_class_ctx->num_contexts; c++) {
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
        ctx->tokenizer_path,
        cat_class_ctx->total_labels);

    av_freep(&combined_labels);
    return ret;
}

static int dnn_classify_activate(AVFilterContext *filter_ctx)
{
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
                        ctx->tokenizer_path,
                        ctx->label_classification_ctx->label_count);
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

static av_cold void dnn_classify_uninit(AVFilterContext *context)
{
    DnnClassifyContext *ctx = context->priv;
    ff_dnn_uninit(&ctx->dnnctx);
    free_contexts(ctx);
}

const FFFilter ff_vf_dnn_classify = {
    .p.name        = "dnn_classify",
    .p.description = NULL_IF_CONFIG_SMALL("Apply DNN classify filter to the input."),
    .p.priv_class  = &dnn_classify_class,
    .priv_size     = sizeof(DnnClassifyContext),
    .preinit       = ff_dnn_filter_init_child_class,
    .init          = dnn_classify_init,
    .uninit        = dnn_classify_uninit,
    FILTER_INPUTS(ff_video_default_filterpad),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .activate      = dnn_classify_activate,
};
