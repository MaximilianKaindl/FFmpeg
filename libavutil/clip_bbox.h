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

#ifndef AVUTIL_CLIP_BBOX_H
#define AVUTIL_CLIP_BBOX_H

#include "rational.h"
#include "avassert.h"
#include "frame.h"
 

typedef struct AVClipBBox {
#define AV_CLIP_BBOX_LABEL_NAME_MAX_SIZE 256
#define AV_CLIP_BBOX_CLASSES_MAX_COUNT 4
    uint32_t classify_count;
    char classify_labels[AV_CLIP_BBOX_CLASSES_MAX_COUNT][AV_CLIP_BBOX_LABEL_NAME_MAX_SIZE];
    AVRational classify_confidences[AV_CLIP_BBOX_CLASSES_MAX_COUNT]; 
} AVClipBBox;

typedef struct AVClipBBoxHeader {
    char source[128];
    uint32_t nb_bboxes;
    size_t bboxes_offset;
    size_t bbox_size;
} AVClipBBoxHeader;

static av_always_inline AVClipBBox *
av_get_clip_bbox(const AVClipBBoxHeader *header, unsigned int idx) {
    av_assert0(header->nb_bboxes < AV_CLIP_BBOX_CLASSES_MAX_COUNT + 1);
    av_assert0(idx < header->nb_bboxes);
    return (AVClipBBox *)((uint8_t *)header + header->bboxes_offset +
                          idx * header->bbox_size);
}

AVClipBBoxHeader *av_clip_bbox_alloc(uint32_t nb_bboxes, size_t *out_size);
AVClipBBoxHeader *av_clip_bbox_create_side_data(AVFrame *frame, uint32_t nb_bboxes);

#endif