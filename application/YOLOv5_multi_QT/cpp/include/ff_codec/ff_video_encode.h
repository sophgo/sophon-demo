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

extern "C" {
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
#include "libavformat/avformat.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include <stdio.h>
#include <unistd.h>
}

#define STEP_ALIGNMENT 32


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
