#include "ff_video_decode.h"
#include <iostream>

VideoDec_FFMPEG::VideoDec_FFMPEG()
{
    ifmt_ctx = NULL;
    video_dec_ctx = NULL;
    video_dec_par = NULL;
    decoder = NULL;

    width   = 0;
    height  = 0;
    pix_fmt = 0;

    video_stream_idx = -1;
    refcount = 1;

    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;

    is_closed = 1;
}
VideoDec_FFMPEG::~VideoDec_FFMPEG()
{
    printf("#VideoDec_FFMPEG exit \n");
}

AVCodecParameters* VideoDec_FFMPEG::getCodecPar()
{
    return video_dec_par;
}

int VideoDec_FFMPEG::openDec(const char* filename,int codec_name_flag,
                                const char *coder_name,int output_format_mode,
                                int sophon_idx, int pcie_no_copyback)
{
    int ret = 0;
    AVDictionary *dict = NULL;
    av_dict_set(&dict, "rtsp_flags", "prefer_tcp", 0);

    ret = avformat_open_input(&ifmt_ctx, filename, NULL, &dict);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open input file\n");
        return ret;
    }

    ret = avformat_find_stream_info(ifmt_ctx, NULL);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot find stream information\n");
        return ret;
    }

    ret = openCodecContext(&video_stream_idx, &video_dec_ctx, ifmt_ctx, AVMEDIA_TYPE_VIDEO ,
                            codec_name_flag, coder_name,output_format_mode,sophon_idx,pcie_no_copyback);
    if (ret >= 0) {
        width   = video_dec_ctx->width;
        height  = video_dec_ctx->height;
        pix_fmt = video_dec_ctx->pix_fmt;
        is_closed = 0;
    }
    av_log(video_dec_ctx, AV_LOG_INFO,
           "openDec video_stream_idx = %d, pix_fmt = %d\n",
           video_stream_idx, pix_fmt);
	av_dict_free(&dict);
    return ret;
}

void VideoDec_FFMPEG::closeDec()
{
    if (video_dec_ctx) {
        avcodec_free_context(&video_dec_ctx);
        video_dec_ctx = NULL;
    }
    if (ifmt_ctx) {
        avformat_close_input(&ifmt_ctx);
        ifmt_ctx = NULL;
    }

    is_closed = 1;
}

int VideoDec_FFMPEG::isClosed()
{
    if (is_closed)
        return 1;
    else
        return 0;
}

AVCodec* VideoDec_FFMPEG::findBmDecoder(AVCodecID dec_id, const char *name, int codec_name_flag, enum AVMediaType type){
    /* find video decoder for the stream */
    AVCodec *codec = NULL;
    if(codec_name_flag && type==AVMEDIA_TYPE_VIDEO){
        const AVCodecDescriptor *desc;
        const char *codec_string = "decoder";

        codec = avcodec_find_decoder_by_name(name);
        if (!codec && (desc = avcodec_descriptor_get_by_name(name))) {
            codec = avcodec_find_decoder(desc->id);
        }

        if (!codec) {
            av_log(NULL, AV_LOG_FATAL, "Unknown %s '%s'\n", codec_string, name);
            exit(1);
        }
        if (codec->type != type) {
            av_log(NULL, AV_LOG_FATAL, "Invalid %s type '%s'\n", codec_string, name);
            exit(1);
        }
    } else {
        codec = avcodec_find_decoder(dec_id);
    }


    if (!codec) {
        fprintf(stderr, "Failed to find %s codec\n",av_get_media_type_string(type));
        exit(1);
    }
    return codec;
}

int VideoDec_FFMPEG::openCodecContext(int *stream_idx,AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx,
                                      enum AVMediaType type,int codec_name_flag, const char *coder_name,
                                      int output_format_mode, int sophon_idx, int pcie_no_copyback)
{
    int ret, stream_index;
    AVStream *st;
    AVCodec *dec = NULL;
    AVDictionary *opts = NULL;

    ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Could not find %s stream \n",av_get_media_type_string(type));
        return ret;
    }

    stream_index = ret;
    st = fmt_ctx->streams[stream_index];

    /* find decoder for the stream */
    if(codec_name_flag && coder_name){
        decoder = findBmDecoder((AVCodecID)0,coder_name,codec_name_flag,AVMEDIA_TYPE_VIDEO);
    }else{
        decoder = findBmDecoder(st->codecpar->codec_id);
    }
    if (!decoder) {
        av_log(NULL, AV_LOG_FATAL,"Failed to find %s codec\n",
               av_get_media_type_string(type));
        return AVERROR(EINVAL);
    }

    /* Allocate a codec context for the decoder */
    *dec_ctx = avcodec_alloc_context3(decoder);// have fix_fmt
    if (!*dec_ctx) {
        av_log(NULL, AV_LOG_FATAL, "Failed to allocate the %s codec context\n",
               av_get_media_type_string(type));
        return AVERROR(ENOMEM);
    }
    /* Copy codec parameters from input stream to output codec context */

    ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar);
    if (ret < 0) {
        av_log(NULL, AV_LOG_FATAL, "Failed to copy %s codec parameters to decoder context\n",
               av_get_media_type_string(type));
        return ret;
    }
    video_dec_par = st->codecpar;
    /* Init the decoders, with or without reference counting */
    av_dict_set(&opts, "refcounted_frames", refcount ? "1" : "0", 0);

    av_dict_set_int(&opts, "zero_copy",pcie_no_copyback, 0);
    av_dict_set_int(&opts, "sophon_idx", sophon_idx, 0);

    if(output_format_mode == 101)
        av_dict_set_int(&opts, "output_format", 101, 18);

    av_dict_set_int(&opts, "extra_frame_buffer_num", 2, 0);


    ret = avcodec_open2(*dec_ctx, dec, &opts);
    if (ret < 0) {
        av_log(NULL, AV_LOG_FATAL, "Failed to open %s codec\n",
               av_get_media_type_string(type));
        return ret;
    }
    *stream_idx = stream_index;

    av_dict_free(&opts);

    return 0;
}

int VideoDec_FFMPEG::grabFrame(AVFrame *frame)
{
    int ret = 0;
    int got_frame = 0;

    while (1) {
        av_packet_unref(&pkt);
        ret = av_read_frame(ifmt_ctx, &pkt);
        if (ret < 0) {
            return NULL; // TODO
        }

        if (pkt.stream_index != video_stream_idx) {
            continue;
        }

        if (!frame) {
            av_log(video_dec_ctx, AV_LOG_ERROR, "Could not allocate frame\n");
            return NULL;
        }

        ret = avcodec_decode_video2(video_dec_ctx, frame, &got_frame, &pkt);
        if (ret < 0) {
            av_log(video_dec_ctx, AV_LOG_ERROR, "Error decoding video frame (%d)\n", ret);
            continue; // TODO
        }

        if (!got_frame) {
            continue;
        }

        width   = video_dec_ctx->width;
        height  = video_dec_ctx->height;
        pix_fmt = video_dec_ctx->pix_fmt;
        if (frame->width != width || frame->height != height || frame->format != pix_fmt) {
            av_log(video_dec_ctx, AV_LOG_ERROR,
                   "Error: Width, height and pixel format have to be "
                   "constant in a rawvideo file, but the width, height or "
                   "pixel format of the input video changed:\n"
                   "old: width = %d, height = %d, format = %s\n"
                   "new: width = %d, height = %d, format = %s\n",
                   width, height, av_get_pix_fmt_name((AVPixelFormat)pix_fmt),
                   frame->width, frame->height,
                   av_get_pix_fmt_name((AVPixelFormat)frame->format));
            continue;
        }

        break;
    }
    return got_frame;
}

int VideoDec_FFMPEG::flushFrame(AVFrame *frame){
    int ret = 0;
    int got_frame = 0;
    av_packet_unref(&pkt);
    pkt.data = NULL;
    pkt.size = 0;
    ret = avcodec_decode_video2(video_dec_ctx, frame, &got_frame, &pkt);
    if (ret < 0) {
        av_log(video_dec_ctx, AV_LOG_ERROR, "Error fflush video frame, ret=%d\n", ret);
        return ret;
    }
    if (!got_frame)
        ret = -1;
    else
        ret = 1;

    return ret;
}
