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

#include "dnn_labels.h" 

void free_label_context(LabelContext *ctx)
{
    if (!ctx)
        return;

    if (ctx->labels) {
        for (int i = 0; i < ctx->label_count; i++) {
            av_freep(&ctx->labels[i]);
        }
        av_freep(&ctx->labels);
    }
    ctx->label_count = 0;
}

void free_category_context(CategoryContext *category)
{
   if (!category)
       return;

   if (category->name) {
       av_freep(&category->name);
   }
   
   if (category->labels) {
       free_label_context(category->labels);
       av_freep(&category->labels);  
   }
}

void free_categories_context(CategoriesContext *ctx)
{
   if (!ctx)
       return;

   if (ctx->categories) {
       for (int i = 0; i < ctx->category_count; i++) {
           free_category_context(&ctx->categories[i]);
       }
       // Now free the array of categories
       av_freep(&ctx->categories);
       ctx->categories = NULL;
   }
   
   if (ctx->name) {
       av_freep(&ctx->name);
       ctx->name = NULL;
   }
   
   ctx->category_count = 0;
   ctx->max_categories = 0;
   ctx->label_count = 0;
}

void free_category_classfication_context(CategoryClassifcationContext *category_classification_ctx){
    if (category_classification_ctx) {
        if (category_classification_ctx->category_units) {
            for (int i = 0; i < category_classification_ctx->num_contexts; i++) {
                if (category_classification_ctx->category_units[i]) {
                    free_categories_context(category_classification_ctx->category_units[i]);
                    av_freep(&category_classification_ctx->category_units[i]);
                }
            }
            av_freep(&category_classification_ctx->category_units);
            category_classification_ctx->category_units = NULL;
        }
        category_classification_ctx->num_contexts = 0;
        category_classification_ctx->max_contexts = 0;
    }
}

int read_label_file(AVFilterContext *context, LabelContext *label_classification_ctx, char* labels_filename, int max_line_length)
{
   int line_len;
   FILE *file;

   file = avpriv_fopen_utf8(labels_filename, "r");
   if (!file){
       av_log(context, AV_LOG_ERROR, "failed to open file %s\n", labels_filename);
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

       if (line_len >= max_line_length) {
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

       if (av_dynarray_add_nofree(&label_classification_ctx->labels, &label_classification_ctx->label_count, label) < 0) {
           av_log(context, AV_LOG_ERROR, "failed to do av_dynarray_add\n");
           fclose(file);
           av_freep(&label);
           return AVERROR(ENOMEM);
       }
   }

   fclose(file);
   return 0;
}

int read_categories_file(AVFilterContext *context, CategoryClassifcationContext *cat_class_ctx, char* categories_filename, int max_line_length)
{
   FILE *file;
   char buf[256];
   int ret = 0;
   CategoriesContext *current_ctx = NULL;
   CategoryContext *current_category = NULL;

   file = avpriv_fopen_utf8(categories_filename, "r");
   if (!file) {
       av_log(context, AV_LOG_ERROR, "Failed to open categories file %s\n", categories_filename);
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

       if (line_len >= max_line_length) {
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
   if (ret < 0) {
       // Clean up current context if it wasn't added to the array
       if (current_ctx) {
           free_categories_context(current_ctx);
       }
   }

   fclose(file);
   return ret;
}

int combine_all_category_labels(LabelContext **label_ctx, CategoryClassifcationContext *cat_class_ctx)
{
    char **combined_labels = NULL;
    int combined_idx = 0;
    
    *label_ctx = av_calloc(1, sizeof(LabelContext));
    if (!(*label_ctx))
        return AVERROR(ENOMEM);
    
    (*label_ctx)->label_count = cat_class_ctx->total_labels;
    (*label_ctx)->labels = av_calloc(cat_class_ctx->total_labels, sizeof(char *));
    if (!(*label_ctx)->labels) {
        av_freep(label_ctx);
        return AVERROR(ENOMEM);
    }

    combined_labels = (*label_ctx)->labels;
    
    // Combine all labels from all categories
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
    
    return 0;
}