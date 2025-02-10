#include "ff_video_encode.h"
#include <iostream>
#include <fstream>
#include "bm_wrapper.hpp"

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

static bool string_start_with(const std::string &s, const std::string &head) {
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

void dump_yuv(std::string filename, AVFrame* frame, int card_id){
 
  bm_device_mem_t mems[3];

  int size0 = frame->height * frame->linesize[0];
  mems[0] = bm_mem_from_device((unsigned long long)frame->data[4], size0);
  int size1 = size0/4;
  mems[1] = bm_mem_from_device((unsigned long long)frame->data[5], size1);
  int size2 = size0/4;
  mems[2] = bm_mem_from_device((unsigned long long)frame->data[6], size2);
    
  char* buffer0 = new char[size0];
  char* buffer1 = new char[size1];
  char* buffer2 = new char[size2];

  bm_handle_t handle;
  bm_dev_request(&handle, card_id);
  bm_memcpy_d2s(handle, buffer0, mems[0]);
  bm_memcpy_d2s(handle, buffer1, mems[1]);
  bm_memcpy_d2s(handle, buffer2, mems[2]);

  std::ofstream outFile(filename);
  outFile.write(buffer0, size0);
  outFile.write(buffer1, size1);
  outFile.write(buffer2, size2);
  
  std::cout << "height: " << frame->height << ", stride: " << frame->linesize[0] << std::endl;

  delete[] buffer0;
  delete[] buffer1;
  delete[] buffer2;
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
    enc_pkt.pts= inPic->pts;
    if ((ret = avcodec_send_frame(enc_ctx, inPic)) < 0) {
        av_log(NULL, AV_LOG_ERROR, " avcodec_send_frame error ret=%d\n",ret);
    }
    while (1) {
        ret = avcodec_receive_packet(enc_ctx, &enc_pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF){
            return ret;
        }
        else if (ret == 0){
            got_output ++;
            enc_pkt.stream_index = 0;
            break;
        }
        else if (ret<0){
            av_log(NULL, AV_LOG_ERROR, " avcodec_receive_packet error ret=%d\n",ret);
        }
    }

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
        if ((ret = avcodec_send_frame(enc_ctx, NULL)) < 0) {
            av_log(NULL, AV_LOG_ERROR, " avcodec_send_frame error ret=%d\n",ret);
        }
        while (1) {
            ret = avcodec_receive_packet(enc_ctx, &enc_pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF){
                return ret;
            }
            else if (ret == 0){
                enc_pkt.stream_index = 0;
                break;
            }
            else if (ret<0){
                av_log(NULL, AV_LOG_ERROR, " avcodec_receive_packet error ret=%d\n",ret);
            }
        }

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

