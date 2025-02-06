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
 * Average classification probabilities video filter.
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

typedef struct ClassProb {
    char    label[AV_CLIP_BBOX_LABEL_NAME_MAX_SIZE];
    int64_t count;
    double  sum;
} ClassProb;

typedef struct AvgClassContext {
    const AVClass *clazz;
    int nb_classes;
    ClassProb *class_probs;
    char *output_file;
} AvgClassContext;

#define OFFSET(x) offsetof(AvgClassContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption avgclass_options[] = {
    { "output_file", "path to output file for averages",
        OFFSET(output_file), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(avgclass);

static av_cold int init(AVFilterContext *ctx)
{
    AvgClassContext *s = ctx->priv;
    s->nb_classes = 0;
    s->class_probs = NULL;
    return 0;
}

static void write_averages_to_file(AVFilterContext *ctx)
{
    AvgClassContext *s = ctx->priv;
    FILE *f;

    if (!s->output_file) {
        av_log(ctx, AV_LOG_WARNING, "No output file specified, skipping average output\n");
        return;
    }

    f = avpriv_fopen_utf8(s->output_file, "w");
    if (!f) {
        av_log(ctx, AV_LOG_ERROR, "Could not open output file %s\n", s->output_file);
        return;
    }

    for (int i = 0; i < s->nb_classes; i++) {
        double avg = s->class_probs[i].count > 0 ?
                    s->class_probs[i].sum / s->class_probs[i].count : 0.0;
        fprintf(f, "%s: %.4f\n", s->class_probs[i].label, avg);
        av_log(ctx, AV_LOG_INFO, "Classification Label: %s: Average probability %.4f\n", 
               s->class_probs[i].label, avg);
    }

    fclose(f);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    AvgClassContext *s = ctx->priv;
    write_averages_to_file(ctx);
    av_freep(&s->class_probs);
}

static ClassProb *find_or_create_class(AvgClassContext *s, const char *label)
{
    int i;
    ClassProb *new_probs;

    for (i = 0; i < s->nb_classes; i++) {
        if (!strcmp(s->class_probs[i].label, label))
            return &s->class_probs[i];
    }

    new_probs = av_realloc_array(s->class_probs, s->nb_classes + 1, sizeof(*s->class_probs));
    if (!new_probs)
        return NULL;
    s->class_probs = new_probs;

    av_strlcpy(s->class_probs[s->nb_classes].label, label, sizeof(s->class_probs[0].label));
    s->class_probs[s->nb_classes].count = 0;
    s->class_probs[s->nb_classes].sum = 0.0;

    return &s->class_probs[s->nb_classes++];
}

static int process_frame(AVFilterContext *ctx, AVFrame *frame)
{
    AvgClassContext *s = ctx->priv;
    AVFrameSideData *sd;
    const AVClipBBoxHeader *header;
    const AVClipBBox *bbox;
    int i, j;
    double prob;
    ClassProb *class_prob;

    
    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_CLIP_BBOXES);
    if (!sd || sd->size < sizeof(AVClipBBoxHeader)) {
        av_log(ctx, AV_LOG_ERROR, "Invalid clip bbox side data\n"); 
        return 0;
    }   

    header = (const AVClipBBoxHeader *)sd->data;

    if (!header || sd->size < sizeof(AVClipBBoxHeader)) {
        av_log(ctx, AV_LOG_ERROR, "Invalid clip bbox header\n");
        return 0;
    }

    if (header->nb_bboxes <= 0 || header->nb_bboxes > AV_CLIP_BBOX_LABEL_NAME_MAX_SIZE) {
        av_log(ctx, AV_LOG_ERROR, "Invalid number or no bboxes\n");
        return 0;
    }
    
    for (i = 0; i < header->nb_bboxes; i++) {
        bbox = av_get_clip_bbox(header, i);
        if (!bbox) {
            av_log(ctx, AV_LOG_ERROR, "Failed to get bbox at index %d\n", i);
            continue;
        }

        if (bbox->classify_count <= 0) {
            continue;
        }

        // Validate classification arrays
        if (!bbox->classify_labels || !bbox->classify_confidences) {
            av_log(ctx, AV_LOG_ERROR, "Missing classification data at bbox %d\n", i);
            continue;
        }

        for (j = 0; j < bbox->classify_count; j++) {
            // Check confidence values before division
            if (bbox->classify_confidences[j].den <= 0) {
                av_log(ctx, AV_LOG_DEBUG, "Invalid confidence at bbox %d class %d: num=%d den=%d\n",
                       i, j, bbox->classify_confidences[j].num, bbox->classify_confidences[j].den);
                continue;
            }

            if (!bbox->classify_labels[j]) {
                av_log(ctx, AV_LOG_ERROR, "NULL label at bbox %d class %d\n", i, j);
                continue;
            }
            if (bbox->classify_confidences[j].num == 0) {
                prob = 0.0;
            } else {
                prob = (double)bbox->classify_confidences[j].num / bbox->classify_confidences[j].den;
                // Sanity check on probability value
                if (prob < 0.0 || prob > 1.0) {
                    av_log(ctx, AV_LOG_WARNING, "Probability out of range [0,1] at bbox %d class %d: %f\n",
                           i, j, prob);
                    continue;
                }
                av_log(ctx, AV_LOG_DEBUG, "Label: %s, Confidence: %.6f\n", bbox->classify_labels[j], prob);

            }
            if (prob < 0.0 || prob > 1.0) {
                av_log(ctx, AV_LOG_WARNING, "Probability out of range [0,1] at bbox %d class %d: %f\n",
                        i, j, prob);
                continue;
            }

            class_prob = find_or_create_class(s, bbox->classify_labels[j]);
            if (!class_prob) {
                return AVERROR(ENOMEM);
            }

            class_prob->sum += prob;
            class_prob->count++;
        }
    }
    return 0;
}

static int avgclass_activate(AVFilterContext *filter_ctx)
{
    AVFilterLink *inlink = filter_ctx->inputs[0];
    AVFilterLink *outlink = filter_ctx->outputs[0];
    AVFrame *in = NULL;
    int ret, status;
    int64_t pts;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    ret = ff_inlink_consume_frame(inlink, &in);
    if (ret < 0)
        return ret;
    if (ret > 0) {
        ret = process_frame(filter_ctx, in);
        if (ret < 0) {
            av_frame_free(&in);
            return ret;
        }
        ret = ff_filter_frame(outlink, in);
        if (ret < 0)
            return ret;
    }

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            ff_outlink_set_status(outlink, status, pts);
            return 0;
        }
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return 0;
}

const FFFilter ff_vf_avgclass = {
    .p.name          = "avgclass",
    .p.description   = NULL_IF_CONFIG_SMALL("Average classification probabilities."),
    .p.priv_class    = &avgclass_class,
    .priv_size     = sizeof(AvgClassContext),
    .init          = init,
    .uninit        = uninit,
    .activate      = avgclass_activate,
    FILTER_INPUTS(ff_video_default_filterpad),
    FILTER_OUTPUTS(ff_video_default_filterpad),
};