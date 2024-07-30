// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===


#include "ff_video_encode.h"

VideoEnc_FFMPEG::VideoEnc_FFMPEG()
{
    pFormatCtx    = NULL;
    pOutfmtormat  = NULL;
    out_stream    = NULL;
    aligned_input = NULL;
    enc_frame_width    = 0;
    enc_frame_height   = 0;
    frame_idx      = 0;
    enc_pix_format = 0;
    dec_pix_format = 0;
    is_closed = 1;
    frameWrite = av_frame_alloc();
}

VideoEnc_FFMPEG::~VideoEnc_FFMPEG()
{
    printf("#######VideoEnc_FFMPEG exit \n");
}


bool string_start_with(const std::string &s, const std::string &head) {
    return s.compare(0, head.size(), head) == 0;
}

int get_output_type(const std::string &output_path)
{
    if(string_start_with(output_path, "rtsp://"))
        return RTSP_STREAM;
    if(string_start_with(output_path, "rtmp://"))
        return RTMP_STREAM;
    if(string_start_with(output_path, "tcp://") || string_start_with(output_path, "udp://"))
        return BASE_STREAM;
    return VIDEO_LOCAL_FILE;
}

int VideoEnc_FFMPEG::openEnc(const char* output_filename, const char* codec_name ,int is_by_filename,
                              int framerate,int width, int height, int encode_pix_format, int bitrate,int sophon_idx)
{

    int ret = 0;
    AVCodec *encoder = NULL;
    AVDictionary *dict = NULL;
    frame_idx = 0;
    enc_pix_format = encode_pix_format;

    enc_frame_width = width;
    enc_frame_height = height;
    
    if( !output_filename )
    {
        av_log(NULL, AV_LOG_ERROR, "inputfile and outputfile cannot not be NULL\n");
        return -1;
    }

    // get output format
    int output_type = get_output_type(std::string(output_filename));
    switch(output_type)
    {
        case RTSP_STREAM:
            printf("you are pushing a rtsp stream.");
            avformat_alloc_output_context2(&pFormatCtx, NULL, "rtsp", output_filename);    
            break;
        case RTMP_STREAM:
            printf("you are pushing a rtmp stream.");
            avformat_alloc_output_context2(&pFormatCtx, NULL, "flv", output_filename);        
            break;
        case BASE_STREAM:
            printf("Not support tcp/udp stream yet.");
            break;
        case VIDEO_LOCAL_FILE:
            printf("sail.Encoder: you are writing a local video file.");
            avformat_alloc_output_context2(&pFormatCtx, NULL, NULL, output_filename);
            break;
        default:
            throw std::runtime_error("Failed to alloc output context.");
            break;
    }

    if (!pFormatCtx) {
        av_log(NULL, AV_LOG_ERROR, "Could not create output context\n");
        return AVERROR_UNKNOWN;
    }


    if(is_by_filename && output_filename){

        pOutfmtormat = const_cast<AVOutputFormat*>(av_guess_format(NULL, output_filename, NULL));
        if(pOutfmtormat->video_codec == AV_CODEC_ID_NONE){
            printf("Unable to assign encoder automatically by file name, please specify by parameter...\n");
            return -1;
        }
        pFormatCtx->oformat = pOutfmtormat;
        encoder = const_cast<AVCodec*>(avcodec_find_encoder(pOutfmtormat->video_codec));
    }
    if(codec_name != NULL)
        encoder = const_cast<AVCodec*>(avcodec_find_encoder_by_name(codec_name));
    if(!encoder){
        printf("Failed to find encoder please try again\n");
        return -1;
    }


    enc_ctx = avcodec_alloc_context3(encoder);
    if (!enc_ctx) {
        av_log(NULL, AV_LOG_FATAL, "Failed to allocate the encoder context\n");
        return AVERROR(ENOMEM);
    }
    enc_ctx->codec_id           = encoder->id;
    enc_ctx->width              = width;
    enc_ctx->height             = height;
    enc_ctx->pix_fmt            = (AVPixelFormat)enc_pix_format;
    enc_ctx->bit_rate_tolerance = bitrate;
    enc_ctx->bit_rate           = (int64_t)bitrate;
    enc_ctx->gop_size           = 32;
    /* video time_base can be set to whatever is handy and supported by encoder */
    enc_ctx->time_base          = (AVRational){1, framerate};
    enc_ctx->framerate          = (AVRational){framerate,1};
    av_log(NULL, AV_LOG_DEBUG, "enc_ctx->bit_rate = %ld\n", enc_ctx->bit_rate);

    out_stream = avformat_new_stream(pFormatCtx, encoder);
    out_stream->time_base       = enc_ctx->time_base;
    out_stream->avg_frame_rate  = enc_ctx->framerate;
    out_stream->r_frame_rate    = out_stream->avg_frame_rate;


    av_dict_set_int(&dict, "sophon_idx", sophon_idx, 0);
    av_dict_set_int(&dict, "gop_preset", 3, 0);
    /* Use system memory */

    av_dict_set_int(&dict, "is_dma_buffer", 1 , 0);

    /* Third parameter can be used to pass settings to encoder */
    ret = avcodec_open2(enc_ctx, encoder, &dict);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open video encoder ");
        return ret;
    }

    ret = avcodec_parameters_from_context(out_stream->codecpar, enc_ctx);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Failed to copy encoder parameters to output stream ");
        return ret;
    }

    if (!(pFormatCtx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open2(&pFormatCtx->pb, output_filename, AVIO_FLAG_WRITE,NULL,NULL);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Could not open output file '%s'", output_filename);
            return ret;
        }
    }

    /* init muxer, write output file header */
    ret = avformat_write_header(pFormatCtx, NULL);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Error occurred when opening output file\n");
        return ret;
    }
    av_dict_free(&dict);
    is_closed = 0;
    return 0;
}

/* data is alligned with 32 */
int VideoEnc_FFMPEG::writeFrame(AVFrame * inPic)
{
    int ret = 0 ;
    int got_output = 0;
    inPic->pts = frame_idx;
    frame_idx++;
    av_log(NULL, AV_LOG_DEBUG, "Encoding frame\n");

    /* encode filtered frame */
    AVPacket enc_pkt;
    enc_pkt.data = NULL;
    enc_pkt.size = 0;
    av_init_packet(&enc_pkt);

    ret = avcodec_encode_video2(enc_ctx, &enc_pkt, inPic, &got_output);
    if (ret < 0)
        return ret;
    if (got_output == 0) {
        av_log(NULL, AV_LOG_WARNING, "No output from encoder\n");
        return -1;
    }

    /* prepare packet for muxing */
    av_log(NULL, AV_LOG_DEBUG, "enc_pkt.pts=%ld, enc_pkt.dts=%ld\n",
           enc_pkt.pts, enc_pkt.dts);
    av_packet_rescale_ts(&enc_pkt, enc_ctx->time_base,out_stream->time_base);
    av_log(NULL, AV_LOG_DEBUG, "rescaled enc_pkt.pts=%ld, enc_pkt.dts=%ld\n",
           enc_pkt.pts,enc_pkt.dts);

    av_log(NULL, AV_LOG_DEBUG, "Muxing frame\n");

    /* mux encoded frame */
    ret = av_interleaved_write_frame(pFormatCtx, &enc_pkt);
    return ret;

}

int  VideoEnc_FFMPEG::flush_encoder()
{
    int ret;
    int got_frame = 0;
    if (!(enc_ctx->codec->capabilities & AV_CODEC_CAP_DELAY))
        return 0;
    while (1) {
        av_log(NULL, AV_LOG_INFO, "Flushing video encoder\n");
        AVPacket enc_pkt;
        enc_pkt.data = NULL;
        enc_pkt.size = 0;
        av_init_packet(&enc_pkt);
        //printf("xxxx\n");
        ret = avcodec_encode_video2(enc_ctx, &enc_pkt, NULL, &got_frame);
        //printf("xxxx1\n");
        if (ret < 0)
            return ret;

        if (!got_frame)
            break;

        /* prepare packet for muxing */
        av_log(NULL, AV_LOG_DEBUG, "enc_pkt.pts=%ld, enc_pkt.dts=%ld\n",
               enc_pkt.pts,enc_pkt.dts);
        av_packet_rescale_ts(&enc_pkt, enc_ctx->time_base,out_stream->time_base);
        av_log(NULL, AV_LOG_DEBUG, "rescaled enc_pkt.pts=%ld, enc_pkt.dts=%ld\n",
               enc_pkt.pts,enc_pkt.dts);
        /* mux encoded frame */
        av_log(NULL, AV_LOG_DEBUG, "Muxing frame\n");
        ret = av_interleaved_write_frame(pFormatCtx, &enc_pkt);
        if (ret < 0)
            break;
    }
    return ret;
}

void VideoEnc_FFMPEG::closeEnc()
{

    flush_encoder();
    av_write_trailer(pFormatCtx);

    if(frameWrite)
        av_frame_free(&frameWrite);

    avcodec_free_context(&enc_ctx);
    if (pFormatCtx && !(pFormatCtx->oformat->flags & AVFMT_NOFILE))
        avio_closep(&pFormatCtx->pb);
    avformat_free_context(pFormatCtx);
    is_closed = 1;
}

int VideoEnc_FFMPEG::isClosed()
{
    if(is_closed)
        return 1;
    else
        return 0;
}

