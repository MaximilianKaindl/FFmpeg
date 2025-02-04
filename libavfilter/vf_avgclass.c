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
#include "libavutil/detection_bbox.h"
#include "libavutil/avstring.h"

typedef struct ClassProb {
    char    label[AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE];
    int64_t count;
    double  sum;
} ClassProb;

typedef struct AvgClassContext {
    const AVClass *clazz;
    int nb_classes;
    ClassProb *class_probs;
} AvgClassContext;

#define OFFSET(x) offsetof(AvgClassContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption avgclass_options[] = {
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

static av_cold void uninit(AVFilterContext *ctx)
{
    AvgClassContext *s = ctx->priv;

    for (int i = 0; i < s->nb_classes; i++) {
        double avg = s->class_probs[i].count > 0 ?
                    s->class_probs[i].sum / s->class_probs[i].count : 0.0;

        av_log(ctx, AV_LOG_INFO, "Classification Label:  %s: Average probability %.4f\n", s->class_probs[i].label, avg);
    }

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

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    AvgClassContext *s = ctx->priv;
    AVFrameSideData *sd;
    const AVDetectionBBoxHeader *header;
    const AVDetectionBBox *bbox;
    int i, j;
    ClassProb *class_prob;

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DETECTION_BBOXES);
    if (sd) {
        header = (const AVDetectionBBoxHeader *)sd->data;
        for (i = 0; i < header->nb_bboxes; i++) {
            bbox = av_get_detection_bbox(header, i);
            for (j = 0; j < bbox->classify_count; j++) {
                double prob = (double)bbox->classify_confidences[j].num /
                            bbox->classify_confidences[j].den;

                class_prob = find_or_create_class(s, bbox->classify_labels[j]);
                if (!class_prob) {
                    av_frame_free(&frame);
                    return AVERROR(ENOMEM);
                }

                class_prob->sum += prob;
                class_prob->count++;
            }
        }
    }
    return ff_filter_frame(ctx->outputs[0], frame);
}

static const AVFilterPad avgclass_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
};

static const AVFilterPad avgclass_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
};

const FFFilter ff_vf_avgclass = {
    .p.name          = "avgclass",
    .p.description   = NULL_IF_CONFIG_SMALL("Average classification probabilities."),
    .p.priv_class    = &avgclass_class,
    .priv_size     = sizeof(AvgClassContext),
    .init          = init,
    .uninit        = uninit,
    FILTER_INPUTS(avgclass_inputs),
    FILTER_OUTPUTS(avgclass_outputs),
};