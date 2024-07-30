// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===


#ifndef __FF_VIDEO_DECODE_H
#define __FF_VIDEO_DECODE_H

#include <iostream>
#include "ff_avframe_convert.h"

#if LIBAVCODEC_VERSION_MAJOR > 58
    static int avcodec_decode_video2(AVCodecContext* dec_ctx, AVFrame *frame, int *got_picture, AVPacket* pkt)
    {
        int ret;
        *got_picture = 0;
        ret = avcodec_send_packet(dec_ctx, pkt);
        if (ret == AVERROR_EOF) {
            ret = 0;
        }
        else if (ret < 0) {
            char err[256] = {0};
            av_strerror(ret, err, sizeof(err));
            fprintf(stderr, "Error sending a packet for decoding, %s\n", err);
            return -1;
        }
        while (ret >= 0) {
            ret = avcodec_receive_frame(dec_ctx, frame);
            if (ret == AVERROR(EAGAIN)) {
                ret = 0;
                break;
            }else if (ret == AVERROR_EOF) {
                printf("File end!\n");
                avcodec_flush_buffers(dec_ctx);
                ret = 0;
                break;
            }
            else if (ret < 0) {
                fprintf(stderr, "Error during decoding\n");
                break;
            }
            *got_picture += 1;
            break;
        }
        if (*got_picture > 1) {
            printf("got picture %d\n", *got_picture);
        }
        return ret;
    }
#endif
class VideoDec_FFMPEG
{
public:
    VideoDec_FFMPEG();
    ~VideoDec_FFMPEG();

    int openDec(const char* filename,int codec_name_flag,
                const char *coder_name,int output_format_mode = 100,
                int sophon_idx = 0, int pcie_no_copyback = 0);
    void closeDec();

    AVCodecParameters* getCodecPar();
    int grabFrame(AVFrame *frame);
    int flushFrame(AVFrame *frame);
    int isClosed();

private:
    AVFormatContext   *ifmt_ctx;
    AVCodec           *decoder;
    AVCodecContext    *video_dec_ctx ;
    AVCodecParameters *video_dec_par;

    int width;
    int height;
    int pix_fmt;

    int video_stream_idx;
    AVPacket pkt;
    int refcount;
    int is_closed;

    AVCodec* findBmDecoder(AVCodecID dec_id, const char *name = "h264_bm",
                           int codec_name_flag = 0,
                           enum AVMediaType type = AVMEDIA_TYPE_VIDEO);


    int openCodecContext(int *stream_idx, AVCodecContext **dec_ctx,
                         AVFormatContext *fmt_ctx, enum AVMediaType type,
                         int codec_name_flag, const char *coder_name,
                         int output_format_mode = 100,
                         int sophon_idx = 0,int pcie_no_copyback = 0);

};


#endif /*__FF_VIDEO_DECODE_H*/

