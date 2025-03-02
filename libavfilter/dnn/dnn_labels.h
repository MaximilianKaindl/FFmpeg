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
#ifndef AVFILTER_DNN_DNN_LABELS_H
#define AVFILTER_DNN_DNN_LABELS_H

#include "libavutil/file_open.h"
#include "libavutil/mem.h"
#include "libavfilter/filters.h"
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

void free_label_context(LabelContext *label_classification_ctx);
void free_category_context(CategoryContext *category);
void free_categories_context(CategoriesContext *ctx);
void free_category_classfication_context(CategoryClassifcationContext *category_classification_ctx);
int read_label_file(AVFilterContext *context, LabelContext *label_classification_ctx, char* labels_filename, int max_line_length);
int read_categories_file(AVFilterContext *context, CategoryClassifcationContext *cat_class_ctx, char* categories_filename, int max_line_length);
int combine_all_category_labels(LabelContext **label_ctx, CategoryClassifcationContext *cat_class_ctx);
int get_category_label_counts(CategoryClassifcationContext *cat_ctx, int **label_counts);
CategoryContext *get_best_category(CategoriesContext *categories_ctx, float *probabilities);
#endif
