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
 * Average classification probabilities filter for any media type.
 */

 #include "libavutil/file_open.h"
 #include "libavutil/mem.h"
 #include "libavutil/opt.h"
 #include "filters.h"
 #include "dnn_filter_common.h"
 #include "video.h"
 #include "audio.h"
 #include "avfilter.h"
 #include "libavutil/time.h"
 #include "libavutil/detection_bbox.h"
 #include "libavutil/avstring.h"
 #include "formats.h"
 
 #define TYPE_ALL 2  // video and audio types
 
 typedef struct ClassProb {
     char    label[AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE];
     int64_t count;
     double  sum;
 } ClassProb;
 
 typedef struct StreamContext {
     int nb_classes;
     ClassProb *class_probs;
 } StreamContext;
 
 typedef struct AvgClassContext {
     const AVClass *class;
     unsigned nb_streams[TYPE_ALL]; /**< number of streams of each type */
     char *output_file;
     StreamContext *stream_ctx;     /**< per-stream context */
     int unsafe;                    /**< flag for unsafe mode */
 } AvgClassContext;
 
 #define OFFSET(x) offsetof(AvgClassContext, x)
 #define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_FILTERING_PARAM
 
 static const AVOption avgclass_options[] = {
     { "output_file", "path to output file for averages",
       OFFSET(output_file), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS },
     { "v", "specify the number of video streams",
       OFFSET(nb_streams[AVMEDIA_TYPE_VIDEO]),
       AV_OPT_TYPE_INT, { .i64 = 1 }, 0, INT_MAX, FLAGS },
     { "a", "specify the number of audio streams",
       OFFSET(nb_streams[AVMEDIA_TYPE_AUDIO]),
       AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, FLAGS },
     { "unsafe", "enable unsafe mode",
       OFFSET(unsafe),
       AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
     { NULL }
 };
 
 AVFILTER_DEFINE_CLASS(avgclass);
 
 static ClassProb *find_or_create_class(StreamContext *stream_ctx, const char *label)
 {
     int i;
     ClassProb *new_probs;
 
     for (i = 0; i < stream_ctx->nb_classes; i++) {
         if (!strcmp(stream_ctx->class_probs[i].label, label))
             return &stream_ctx->class_probs[i];
     }
 
     new_probs = av_realloc_array(stream_ctx->class_probs, 
                                  stream_ctx->nb_classes + 1, 
                                  sizeof(*stream_ctx->class_probs));
     if (!new_probs)
         return NULL;
     stream_ctx->class_probs = new_probs;
 
     av_strlcpy(stream_ctx->class_probs[stream_ctx->nb_classes].label, 
               label, 
               sizeof(stream_ctx->class_probs[0].label));
     stream_ctx->class_probs[stream_ctx->nb_classes].count = 0;
     stream_ctx->class_probs[stream_ctx->nb_classes].sum = 0.0;
 
     return &stream_ctx->class_probs[stream_ctx->nb_classes++];
 }
 
 static void write_averages_to_file(AVFilterContext *ctx)
 {
     AvgClassContext *s = ctx->priv;
     FILE *f;
     int stream_idx, i;
 
     if (!s->output_file) {
         av_log(ctx, AV_LOG_WARNING, "No output file specified, skipping average output\n");
         return;
     }
 
     f = avpriv_fopen_utf8(s->output_file, "w");
     if (!f) {
         av_log(ctx, AV_LOG_ERROR, "Could not open output file %s\n", s->output_file);
         return;
     }
 
     for (stream_idx = 0; stream_idx < ctx->nb_inputs; stream_idx++) {
         StreamContext *stream_ctx = &s->stream_ctx[stream_idx];
         
         fprintf(f, "Stream #%d:\n", stream_idx);
         for (i = 0; i < stream_ctx->nb_classes; i++) {
             double avg = stream_ctx->class_probs[i].count > 0 ?
                         stream_ctx->class_probs[i].sum / stream_ctx->class_probs[i].count : 0.0;
             fprintf(f, "  %s:%.4f:%ld\n", 
                    stream_ctx->class_probs[i].label, 
                    avg, 
                    stream_ctx->class_probs[i].count);
             av_log(ctx, AV_LOG_INFO, "Stream #%d, Label: %s: Average probability %.4f, Appeared %ld times\n", 
                    stream_idx,
                    stream_ctx->class_probs[i].label, 
                    avg, 
                    stream_ctx->class_probs[i].count);
         }
     }
 
     fclose(f);
 }
 
 static int process_frame(AVFilterContext *ctx, int stream_idx, AVFrame *frame)
 {
     AvgClassContext *s = ctx->priv;
     StreamContext *stream_ctx = &s->stream_ctx[stream_idx];
     AVFrameSideData *sd;
     const AVDetectionBBoxHeader *header;
     const AVDetectionBBox *bbox;
     int i, j;
     double prob;
     ClassProb *class_prob;
 
     sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DETECTION_BBOXES);
     if (!sd || sd->size < sizeof(AVDetectionBBoxHeader)) {
         av_log(ctx, AV_LOG_DEBUG, "No bbox side data in frame for stream %d\n", stream_idx); 
         return 0;
     }   
 
     header = (const AVDetectionBBoxHeader *)sd->data;
 
     if (!header || sd->size < sizeof(AVDetectionBBoxHeader)) {
         av_log(ctx, AV_LOG_ERROR, "Invalid bbox header\n");
         return 0;
     }
 
     if (header->nb_bboxes <= 0 || header->nb_bboxes > 10000) {
         av_log(ctx, AV_LOG_ERROR, "Invalid number or no bboxes\n");
         return 0;
     }
     
     for (i = 0; i < header->nb_bboxes; i++) {
         bbox = av_get_detection_bbox(header, i);
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
                 av_log(ctx, AV_LOG_DEBUG, "Stream #%d, Label: %s, Confidence: %.6f\n", 
                       stream_idx, bbox->classify_labels[j], prob);
             }
 
             class_prob = find_or_create_class(stream_ctx, bbox->classify_labels[j]);
             if (!class_prob) {
                 return AVERROR(ENOMEM);
             }
 
             class_prob->sum += prob;
             class_prob->count++;
         }
     }
     return 0;
 }
 
 static int query_formats(const AVFilterContext *ctx,
                          AVFilterFormatsConfig **cfg_in,
                          AVFilterFormatsConfig **cfg_out)
 {
     const AvgClassContext *s = ctx->priv;
     AVFilterFormats *formats;
     AVFilterChannelLayouts *layouts = NULL;
     AVFilterFormats *rates = NULL;
     unsigned type, nb_str, idx0 = 0, idx, str;
     int ret;
 
     for (type = 0; type < TYPE_ALL; type++) {
         nb_str = s->nb_streams[type];
         for (str = 0; str < nb_str; str++) {
             idx = idx0;
 
             /* Set the output formats */
             formats = ff_all_formats(type);
             if ((ret = ff_formats_ref(formats, &cfg_out[idx]->formats)) < 0)
                 return ret;
 
             if (type == AVMEDIA_TYPE_AUDIO) {
                 rates = ff_all_samplerates();
                 if ((ret = ff_formats_ref(rates, &cfg_out[idx]->samplerates)) < 0)
                     return ret;
                 layouts = ff_all_channel_layouts();
                 if ((ret = ff_channel_layouts_ref(layouts, &cfg_out[idx]->channel_layouts)) < 0)
                     return ret;
             }
 
             /* Set the same formats for each corresponding input */
             if ((ret = ff_formats_ref(formats, &cfg_in[idx]->formats)) < 0)
                 return ret;
                 
             if (type == AVMEDIA_TYPE_AUDIO) {
                 if ((ret = ff_formats_ref(rates, &cfg_in[idx]->samplerates)) < 0 ||
                     (ret = ff_channel_layouts_ref(layouts, &cfg_in[idx]->channel_layouts)) < 0)
                     return ret;
             }
 
             idx0++;
         }
     }
     return 0;
 }
 
 static int config_output(AVFilterLink *outlink)
 {
     FilterLink *outl     = ff_filter_link(outlink);
     AVFilterContext *ctx = outlink->src;
     unsigned out_no = FF_OUTLINK_IDX(outlink);
     AVFilterLink *inlink = ctx->inputs[out_no];
     FilterLink *inl = ff_filter_link(inlink);
 
     outlink->time_base           = inlink->time_base;
     outlink->w                   = inlink->w;
     outlink->h                   = inlink->h;
     outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
     outlink->format              = inlink->format;
     outl->frame_rate             = inl->frame_rate;
     
     return 0;
 }
 
 static AVFrame *get_video_buffer(AVFilterLink *inlink, int w, int h)
 {
     AVFilterContext *ctx = inlink->dst;
     unsigned in_no = FF_INLINK_IDX(inlink);
     AVFilterLink *outlink = ctx->outputs[in_no];
 
     return ff_get_video_buffer(outlink, w, h);
 }
 
 static AVFrame *get_audio_buffer(AVFilterLink *inlink, int nb_samples)
 {
     AVFilterContext *ctx = inlink->dst;
     unsigned in_no = FF_INLINK_IDX(inlink);
     AVFilterLink *outlink = ctx->outputs[in_no];
 
     return ff_get_audio_buffer(outlink, nb_samples);
 }
 
 static av_cold int init(AVFilterContext *ctx)
 {
     AvgClassContext *s = ctx->priv;
     unsigned type, str;
     int ret;
 
     /* create input pads */
     for (type = 0; type < TYPE_ALL; type++) {
         for (str = 0; str < s->nb_streams[type]; str++) {
             AVFilterPad pad = {
                 .type             = type,
             };
             if (type == AVMEDIA_TYPE_VIDEO)
                 pad.get_buffer.video = get_video_buffer;
             else
                 pad.get_buffer.audio = get_audio_buffer;
             pad.name = av_asprintf("%c%d", "va"[type], str);
             if ((ret = ff_append_inpad_free_name(ctx, &pad)) < 0)
                 return ret;
         }
     }
     
     /* create output pads */
     for (type = 0; type < TYPE_ALL; type++) {
         for (str = 0; str < s->nb_streams[type]; str++) {
             AVFilterPad pad = {
                 .type          = type,
                 .config_props  = config_output,
             };
             pad.name = av_asprintf("out:%c%d", "va"[type], str);
             if ((ret = ff_append_outpad_free_name(ctx, &pad)) < 0)
                 return ret;
         }
     }
 
     /* allocate per-stream contexts */
     s->stream_ctx = av_calloc(ctx->nb_inputs, sizeof(*s->stream_ctx));
     if (!s->stream_ctx)
         return AVERROR(ENOMEM);
 
     return 0;
 }
 
 static av_cold void uninit(AVFilterContext *ctx)
 {
     AvgClassContext *s = ctx->priv;
     int i;
     
     write_averages_to_file(ctx);
     
     for (i = 0; i < ctx->nb_inputs; i++) {
         av_freep(&s->stream_ctx[i].class_probs);
     }
     av_freep(&s->stream_ctx);
 }
 
 static int avgclass_activate(AVFilterContext *ctx)
 {
     int ret, status;
     int64_t pts;
     AVFrame *in = NULL;
     unsigned i;
 
     /* Handle EOF on inputs */
     for (i = 0; i < ctx->nb_inputs; i++) {
         AVFilterLink *inlink = ctx->inputs[i];
         AVFilterLink *outlink = ctx->outputs[i];
         
         if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
             if (status == AVERROR_EOF) {
                 ff_outlink_set_status(outlink, status, pts);
                 continue;
             }
         }
         
         /* Process frames */
         ret = ff_inlink_consume_frame(inlink, &in);
         if (ret < 0)
             return ret;
         if (ret > 0) {
             /* Process the frame for classification data */
             ret = process_frame(ctx, i, in);
             if (ret < 0) {
                 av_frame_free(&in);
                 return ret;
             }
             
             /* Forward the frame to the corresponding output */
             ret = ff_filter_frame(outlink, in);
             if (ret < 0)
                 return ret;
         }
         
         /* Request more frames if needed */
         if (ff_outlink_frame_wanted(outlink) && !ff_inlink_check_available_samples(inlink, 1))
             ff_inlink_request_frame(inlink);
     }
 
     return FFERROR_NOT_READY;
 }
 
 static int process_command(AVFilterContext *ctx, const char *cmd, const char *args,
                           char *res, int res_len, int flags)
 {
     if (!strcmp(cmd, "writeinfo")) {
         write_averages_to_file(ctx);
         return 0;
     }
     
     return AVERROR(ENOSYS);
 }
 
 const FFFilter ff_avf_avgclass = {
     .p.name          = "avgclass",
     .p.description   = NULL_IF_CONFIG_SMALL("Average classification probabilities for audio and video streams."),
     .p.priv_class    = &avgclass_class,
     .p.inputs        = NULL,
     .p.outputs       = NULL,
     .p.flags         = AVFILTER_FLAG_DYNAMIC_INPUTS | AVFILTER_FLAG_DYNAMIC_OUTPUTS,
     .priv_size       = sizeof(AvgClassContext),
     .init            = init,
     .uninit          = uninit,
     .activate        = avgclass_activate,
     .process_command = process_command,
     FILTER_QUERY_FUNC2(query_formats),
 };