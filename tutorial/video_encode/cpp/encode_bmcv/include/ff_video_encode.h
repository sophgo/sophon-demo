// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===


#ifndef __FF_VIDEO_ENCODE_
#define __FF_VIDEO_ENCODE_

#include "ff_avframe_convert.h"

#define STEP_ALIGNMENT 32

enum OUTPUT_TYPE{
        RTSP_STREAM=0,
        RTMP_STREAM,
        BASE_STREAM,
        VIDEO_LOCAL_FILE
};

#if LIBAVCODEC_VERSION_MAJOR > 58
    static int avcodec_encode_video2(AVCodecContext *avctx, AVPacket *avpkt, const AVFrame *frame, int *got_packet_ptr) {
        int ret = avcodec_send_frame(avctx, frame);
        *got_packet_ptr = 0;
        if (ret < 0) {
            return ret;
        }

        ret = avcodec_receive_packet(avctx, avpkt);
        if(ret == 0){
            *got_packet_ptr = 1;
        } else if (ret == AVERROR(EAGAIN)){
            printf("Need more frame for one packet");
            ret = 0;
        } else if (ret == AVERROR_EOF) {
            printf("File end");
            ret = 0;
        } else if (ret < 0){
            printf("Error during encoding");
        }
        
        return ret;
    }
#endif

class VideoEnc_FFMPEG
{
public:
    VideoEnc_FFMPEG();
    ~VideoEnc_FFMPEG();

    int  openEnc(const char* output_filename, const char* codec_name,
                    int is_by_filename, int framerate,int width, int height,
                    int inputformat,int bitrate,int sophon_idx = 0);

    void closeEnc();
    int  writeFrame(AVFrame * inputPicture);
    int  writeAvFrame(AVFrame * inputPicture);
    int  flush_encoder();
    int  isClosed();

    AVFrame         * frameWrite;
private:
    AVFormatContext * pFormatCtx;
    AVOutputFormat  * pOutfmtormat;
    AVCodecContext  * enc_ctx;

    AVStream        * out_stream;
    uint8_t         * aligned_input;
    int               dec_pix_format;
    int               enc_pix_format;
    int               enc_frame_width;
    int               enc_frame_height;
    int               frame_idx;
    int               is_closed;
};

#endif
