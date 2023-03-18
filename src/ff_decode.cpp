//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
/*
 * This is a wrapper header of BMruntime & BMCV, aiming to simplify user's program.
 */
#include "ff_decode.hpp"
#include <unistd.h>
#include <iostream>
#include <thread>
#include <sys/time.h>
#include <unistd.h>

using namespace std;

bool hardware_decode = true;
bool data_on_device_mem = true;

const int hw_jpeg_header_fmt_words[] =
    {
        0x221111, // yuv420
        0x211111, // yuv422
        0x111111  // yuv444
};

int read_buffer(void *opaque, uint8_t *buf, int buf_size)
{
    bs_buffer_t *bs = (bs_buffer_t *)opaque;

    int r = bs->size - bs->pos;
    if (r <= 0)
    {
        // cout << "EOF of AVIO." << endl;
        return AVERROR_EOF;
    }

    uint8_t *p = bs->start + bs->pos;
    int len = (r >= buf_size) ? buf_size : r;
    memcpy(buf, p, len);

    bs->pos += len;

    return len;
}

bool determine_hardware_decode(uint8_t *buffer)
{
    int ret = 0;
    int offset = 2;
    int SOF0_Marker = 0xFFC0;
    bool is_hw = false;
    while (1)
    {
        uint8_t flag_high = *(uint8_t *)(buffer + offset);
        uint8_t flag_low = *(uint8_t *)(buffer + offset + 1);

        int word = (flag_high << 8) + flag_low;

        if (SOF0_Marker == word)
        {
            // gray
            if (1 == (*(buffer + offset + 9) & 255))
            {
                return true;
            }
            // color
            offset += 11;
            int ret1 = *(buffer + offset) & 255;
            offset += 3;
            int ret2 = *(buffer + offset) & 255;
            offset += 3;
            int ret3 = *(buffer + offset) & 255;

            ret = (ret1 << 16) + (ret2 << 8) + ret3;
            break;
        }

        else
        {
            offset += 2;
            uint8_t offset_high = *(uint8_t *)(buffer + offset);
            uint8_t offset_low = *(uint8_t *)(buffer + offset + 1);
            offset += (offset_high << 8) + offset_low;
        }
    }

    for (int i = 0; i < 3; i++)
    {
        // std::cout << "hw " << is_hw << std::endl;
        is_hw = (hw_jpeg_header_fmt_words[i] == ret) ? true : false;
        // std::cout << "hw " << is_hw << std::endl;
        if (is_hw)
            break;
    }
    return is_hw;
}

VideoDecFFM::VideoDecFFM()
{
    ifmt_ctx = NULL;
    video_dec_ctx = NULL;
    video_dec_par = NULL;
    decoder = NULL;

    is_rtsp = 0;
    width = 0;
    height = 0;
    pix_fmt = 0;

    video_stream_idx = -1;
    refcount = 1;

    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;

    frame = av_frame_alloc();
}

VideoDecFFM::~VideoDecFFM()
{
    closeDec();
    printf("#VideoDecFFM exit \n");
}

bool string_start_with(const string &s, const string &prefix)
{
    return (s.compare(0, prefix.size(), prefix) == 0);
}

int map_avformat_to_bmformat(int avformat)
{
    int format;
    switch (avformat)
    {
    case AV_PIX_FMT_RGB24:
        format = FORMAT_RGB_PACKED;
        break;
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
        format = FORMAT_YUV420P;
        break;
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUVJ422P:
        format = FORMAT_YUV422P;
        break;
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:
        format = FORMAT_YUV444P;
        break;
    case AV_PIX_FMT_NV12:
        format = FORMAT_NV12;
        break;
    case AV_PIX_FMT_NV16:
        format = FORMAT_NV16;
        break;
    case AV_PIX_FMT_GRAY8:
        format = FORMAT_GRAY;
        break;
    case AV_PIX_FMT_GBRP:
        format = FORMAT_RGBP_SEPARATE;
        break;
    default:
        printf("unsupported av_pix_format %d\n", avformat);
        return -1;
    }

    return format;
}

bm_status_t avframe_to_bm_image(bm_handle_t &handle, AVFrame *in, bm_image *out, bool is_jpeg)
{

    int plane = 0;
    int data_four_denominator = -1;
    int data_five_denominator = -1;
    int data_six_denominator = -1;
    static int mem_flags = USEING_MEM_HEAP2;

    switch (in->format)
    {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
        plane = 1;
        data_four_denominator = 1;
        data_five_denominator = -1;
        data_six_denominator = -1;
    case AV_PIX_FMT_GRAY8:
        plane = 1;
        data_four_denominator = -1;
        data_five_denominator = -1;
        data_six_denominator = -1;
        break;
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
        plane = 3;
        data_four_denominator = -1;
        data_five_denominator = 2;
        data_six_denominator = 2;
        break;
    case AV_PIX_FMT_NV12:
        plane = 2;
        data_four_denominator = -1;
        data_five_denominator = 1;
        data_six_denominator = -1;
        break;
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUVJ422P:
        plane = 3;
        data_four_denominator = -1;
        data_five_denominator = 1;
        data_six_denominator = 1;
        break;
    // case AV_PIX_FMT_YUV440P:
    // case AV_PIX_FMT_YUVJ440P:
    //     plane = 3;
    //     data_four_denominator = -1;
    //     data_five_denominator = 1;
    //     data_six_denominator = 4;
    //     break;
    case AV_PIX_FMT_NV16:
        plane = 2;
        data_four_denominator = -1;
        data_five_denominator = 2;
        data_six_denominator = -1;
        break;
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:
    case AV_PIX_FMT_GBRP:
        plane = 3;
        data_four_denominator = -1;
        data_five_denominator = 1;
        data_six_denominator = 1;
        break;
    default:
        printf("unsupported format, only gray,nv12,yuv420p,nv16,yuv422p horizontal,yuv444p,rgbp supported\n");
        break;
    }

    if (in->channel_layout == 101)
    { /* COMPRESSED NV12 FORMAT */
        if ((0 == in->height) || (0 == in->width) ||
            (0 == in->linesize[4]) || (0 == in->linesize[5]) || (0 == in->linesize[6]) || (0 == in->linesize[7]) ||
            (0 == in->data[4]) || (0 == in->data[5]) || (0 == in->data[6]) || (0 == in->data[7]))
        {
            printf("bm_image_from_frame: get yuv failed!!");
            return BM_ERR_PARAM;
        }
        bm_image cmp_bmimg;
        bm_image_create(handle,
                        in->height,
                        in->width,
                        FORMAT_COMPRESSED,
                        DATA_TYPE_EXT_1N_BYTE,
                        &cmp_bmimg);

        bm_device_mem_t input_addr[4];
        int size = in->height * in->linesize[4];
        input_addr[0] = bm_mem_from_device((unsigned long long)in->data[6], size);
        size = (in->height / 2) * in->linesize[5];
        input_addr[1] = bm_mem_from_device((unsigned long long)in->data[4], size);
        size = in->linesize[6];
        input_addr[2] = bm_mem_from_device((unsigned long long)in->data[7], size);
        size = in->linesize[7];
        input_addr[3] = bm_mem_from_device((unsigned long long)in->data[5], size);
        bm_image_attach(cmp_bmimg, input_addr);
        bm_image_create(handle,
                        in->height,
                        in->width,
                        FORMAT_YUV420P,
                        DATA_TYPE_EXT_1N_BYTE,
                        out);
        if (mem_flags == USEING_MEM_HEAP2 && bm_image_alloc_dev_mem_heap_mask(*out, USEING_MEM_HEAP2) != BM_SUCCESS)
        {
            mem_flags = USEING_MEM_HEAP1;
        }
        if (mem_flags == USEING_MEM_HEAP1 && bm_image_alloc_dev_mem_heap_mask(*out, USEING_MEM_HEAP1) != BM_SUCCESS)
        {
            printf("bmcv allocate mem failed!!!");
        }

        bmcv_rect_t crop_rect = {0, 0, in->width, in->height};
        bmcv_image_vpp_convert(handle, 1, cmp_bmimg, out, &crop_rect);
        bm_image_destroy(cmp_bmimg);
    }
    else
    {
        int stride[3];
        bm_image_format_ext bm_format;
        bm_device_mem_t input_addr[3] = {0};

        data_on_device_mem ? stride[0] = in->linesize[4] : stride[0] = in->linesize[0];

        if (plane > 1)
        {
            data_on_device_mem ? stride[1] = in->linesize[5] : stride[1] = in->linesize[1];
        }
        if (plane > 2)
        {
            data_on_device_mem ? stride[2] = in->linesize[6] : stride[2] = in->linesize[2];
        }
        bm_image tmp;
        bm_format = (bm_image_format_ext)map_avformat_to_bmformat(in->format);
        bm_image_create(handle,
                        in->height,
                        in->width,
                        bm_format,
                        DATA_TYPE_EXT_1N_BYTE,
                        &tmp,
                        stride);
        bm_image_create(handle,
                        in->height,
                        in->width,
                        FORMAT_BGR_PACKED,
                        DATA_TYPE_EXT_1N_BYTE,
                        out);
        bm_image_alloc_dev_mem_heap_mask(*out, USEING_MEM_HEAP2);

        int size = in->height * stride[0];
        if (data_four_denominator != -1)
        {
            size = in->height * stride[0] * 3;
        }
        if (data_on_device_mem)
        {
            input_addr[0] = bm_mem_from_device((unsigned long long)in->data[4], size);
        }
        else
        {
            bm_malloc_device_byte(handle, &input_addr[0], size);
            bm_memcpy_s2d_partial(handle, input_addr[0], in->data[0], size);
        }

        if (data_five_denominator != -1)
        {
            size = in->height * stride[1] / data_five_denominator;
            if (data_on_device_mem)
            {
                input_addr[1] = bm_mem_from_device((unsigned long long)in->data[5], size);
            }
            else
            {
                bm_malloc_device_byte(handle, &input_addr[1], size);
                bm_memcpy_s2d_partial(handle, input_addr[1], in->data[1], size);
            }
        }

        if (data_six_denominator != -1)
        {
            size = in->height * stride[2] / data_six_denominator;
            if (data_on_device_mem)
            {
                input_addr[2] = bm_mem_from_device((unsigned long long)in->data[6], size);
            }
            else
            {
                bm_malloc_device_byte(handle, &input_addr[2], size);
                bm_memcpy_s2d_partial(handle, input_addr[2], in->data[2], size);
            }
        }

        bm_image_attach(tmp, input_addr);
        if (is_jpeg)
        {
            csc_type_t csc_type = CSC_YPbPr2RGB_BT601;
            bmcv_image_vpp_csc_matrix_convert(handle, 1, tmp, out, csc_type, NULL, BMCV_INTER_NEAREST, NULL);
        }
        else
        {
            bmcv_rect_t crop_rect = {0, 0, in->width, in->height};
            bmcv_image_vpp_convert(handle, 1, tmp, out, &crop_rect);
        }
        bm_image_detach(tmp);
        
        if (!data_on_device_mem)
        {
            bm_free_device(handle, input_addr[0]);
            if (data_five_denominator != -1)
                bm_free_device(handle, input_addr[1]);
            if (data_six_denominator != -1)
                bm_free_device(handle, input_addr[2]);
        }
    }
    return BM_SUCCESS;
}

int VideoDecFFM::openDec(bm_handle_t *dec_handle, const char *input)
{
    if (strstr(input, "rtsp://"))
        this->is_rtsp = 1;
    this->handle = dec_handle;
    int ret = 0;
    AVDictionary *dict = NULL;
    av_dict_set(&dict, "rtsp_flags", "prefer_tcp", 0);
    ret = avformat_open_input(&ifmt_ctx, input, NULL, &dict);
    if (ret < 0)
    {
        av_log(NULL, AV_LOG_ERROR, "Cannot open input file\n");
        return ret;
    }

    ret = avformat_find_stream_info(ifmt_ctx, NULL);
    if (ret < 0)
    {
        av_log(NULL, AV_LOG_ERROR, "Cannot find stream information\n");
        return ret;
    }

    ret = openCodecContext(&video_stream_idx, &video_dec_ctx, ifmt_ctx, AVMEDIA_TYPE_VIDEO, bm_get_devid(*dec_handle));

    if (ret >= 0)
    {
        width = video_dec_ctx->width;
        height = video_dec_ctx->height;
        pix_fmt = video_dec_ctx->pix_fmt;
    }
    av_log(video_dec_ctx, AV_LOG_INFO,
           "openDec video_stream_idx = %d, pix_fmt = %d\n",
           video_stream_idx, pix_fmt);

    thread push(&VideoDecFFM::vidPushImage, this);
    push.detach();

    av_dict_free(&dict);

    return ret;
}

void VideoDecFFM::closeDec()
{
    if (video_dec_ctx)
    {
        avcodec_free_context(&video_dec_ctx);
        video_dec_ctx = NULL;
    }
    if (ifmt_ctx)
    {
        avformat_close_input(&ifmt_ctx);
        ifmt_ctx = NULL;
    }
    if (frame)
    {
        av_frame_free(&frame);
        frame = NULL;
    }
}

int VideoDecFFM::openCodecContext(int *stream_idx, AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx,
                                  enum AVMediaType type,
                                  int sophon_idx)
{
    int ret, stream_index;
    AVStream *st;
    AVCodec *dec = NULL;
    AVDictionary *opts = NULL;

    ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
    if (ret < 0)
    {
        av_log(NULL, AV_LOG_ERROR, "Could not find %s stream \n", av_get_media_type_string(type));
        return ret;
    }

    stream_index = ret;
    st = fmt_ctx->streams[stream_index];

    if (st->codecpar->codec_id != AV_CODEC_ID_H264 && st->codecpar->codec_id != AV_CODEC_ID_HEVC)
    {
        hardware_decode = false;
        data_on_device_mem = false;
    }

    /* find decoder for the stream */
    decoder = avcodec_find_decoder(st->codecpar->codec_id);

    if (!decoder)
    {
        av_log(NULL, AV_LOG_FATAL, "Failed to find %s codec\n",
               av_get_media_type_string(type));
        return AVERROR(EINVAL);
    }

    /* Allocate a codec context for the decoder */
    *dec_ctx = avcodec_alloc_context3(decoder);
    if (!*dec_ctx)
    {
        av_log(NULL, AV_LOG_FATAL, "Failed to allocate the %s codec context\n",
               av_get_media_type_string(type));
        return AVERROR(ENOMEM);
    }

    /* Copy codec parameters from input stream to output codec context */
    ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar);
    if (ret < 0)
    {
        av_log(NULL, AV_LOG_FATAL, "Failed to copy %s codec parameters to decoder context\n",
               av_get_media_type_string(type));
        return ret;
    }

    video_dec_par = st->codecpar;
    /* Init the decoders, with or without reference counting */
    av_dict_set(&opts, "refcounted_frames", refcount ? "1" : "0", 0);
    av_dict_set_int(&opts, "sophon_idx", sophon_idx, 0);
    av_dict_set_int(&opts, "extra_frame_buffer_num", EXTRA_FRAME_BUFFER_NUM, 0); // if we use dma_buffer mode

    ret = avcodec_open2(*dec_ctx, dec, &opts);
    if (ret < 0)
    {
        av_log(NULL, AV_LOG_FATAL, "Failed to open %s codec\n",
               av_get_media_type_string(type));
        return ret;
    }
    *stream_idx = stream_index;

    av_dict_free(&opts);

    return 0;
}

AVFrame *VideoDecFFM::grabFrame()
{
    int ret = 0;
    int got_frame = 0;
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    while (1)
    {
        av_packet_unref(&pkt);
        ret = av_read_frame(ifmt_ctx, &pkt);
        if (ret < 0)
        {
            if (ret == AVERROR(EAGAIN))
            {
                gettimeofday(&tv2, NULL);
                if (((tv2.tv_sec - tv1.tv_sec) * 1000 + (tv2.tv_usec - tv1.tv_usec) / 1000) > 1000 * 60)
                {
                    av_log(video_dec_ctx, AV_LOG_WARNING, "av_read_frame failed ret(%d) retry time >60s.\n", ret);
                    break;
                }
                usleep(10 * 1000);

                continue;
            }
            av_log(video_dec_ctx, AV_LOG_ERROR, "av_read_frame ret(%d) maybe eof...\n", ret);
            quit_flag = true;
            return NULL;
        }

        if (pkt.stream_index != video_stream_idx)
        {
            continue;
        }

        if (!frame)
        {
            av_log(video_dec_ctx, AV_LOG_ERROR, "Could not allocate frame\n");
            return NULL;
        }

        if (refcount)
            av_frame_unref(frame);
        gettimeofday(&tv1, NULL);
        ret = avcodec_decode_video2(video_dec_ctx, frame, &got_frame, &pkt);
        if (ret < 0)
        {
            av_log(video_dec_ctx, AV_LOG_ERROR, "Error decoding video frame (%d)\n", ret);
            continue;
        }

        if (!got_frame)
        {
            continue;
        }

        width = video_dec_ctx->width;
        height = video_dec_ctx->height;
        pix_fmt = video_dec_ctx->pix_fmt;
        if (frame->width != width || frame->height != height || frame->format != pix_fmt)
        {
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
    return frame;
}

void *VideoDecFFM::vidPushImage()
{
    while (1)
    {
        while (queue.size() == QUEUE_MAX_SIZE)
        {
            if (is_rtsp)
            {
                std::lock_guard<std::mutex> my_lock_guard(lock);
                bm_image *img = queue.front();
                bm_image_destroy(*img);
                queue.pop();
                cout << "rtsp pop, queue size " << queue.size() << endl;
            }
            else
            {
                usleep(2000);
            }
        }

        bm_image *img = new bm_image;
        AVFrame *avframe = grabFrame();
        if (quit_flag)
            break;
        avframe_to_bm_image(*(this->handle), avframe, img, false);

        std::lock_guard<std::mutex> my_lock_guard(lock);
        queue.push(img);
    }
    return NULL;
}

bm_image *VideoDecFFM::grab()
{
    while (queue.empty())
    {
        if (quit_flag)
            return nullptr;
        usleep(500);
    }
    bm_image *bm_img;
    {
        std::lock_guard<std::mutex> my_lock_guard(lock);
        bm_img = queue.front();
        queue.pop();
    }
    // cout << "grab, queue size " << queue.size() << endl;
    return bm_img;
}

bm_status_t picDec(bm_handle_t &handle, const char *path, bm_image &img)
{
    string input_name = path;
    auto pos1 = input_name.find(".jpg");
    auto pos2 = input_name.find(".jpeg");
    auto pos3 = input_name.find(".png");
    auto pos4 = input_name.find(".JPG");
    auto pos5 = input_name.find(".JPEG");
    auto pos6 = input_name.find(".PNG");
    if (pos1 == string::npos && pos2 == string::npos && pos3 == string::npos && pos4 == string::npos && pos5 == string::npos && pos6 == string::npos)
    {
        fprintf(stderr, "not support pic format, only support jpg and png\n");
        exit(1);
    }

    if (pos1 == string::npos && pos2 == string::npos && pos4 == string::npos && pos5 == string::npos)
    {
        bm_status_t ret = pngDec(handle, input_name, img);
        return ret;
    }
    else
    {
        bm_status_t ret = jpgDec(handle, input_name, img);
        return ret;
    }
}

bm_status_t pngDec(bm_handle_t &handle, string input_name, bm_image &img)
{
    FILE *infile = fopen(input_name.c_str(), "rb+");
    fseek(infile, 0, SEEK_END);
    int numBytes = ftell(infile);
    fseek(infile, 0, SEEK_SET);
    uint8_t *bs_buffer = (uint8_t *)av_malloc(numBytes);
    fread(bs_buffer, sizeof(uint8_t), numBytes, infile);
    fclose(infile);

    const AVCodec *codec;
    AVCodecContext *dec_ctx = NULL;
    AVPacket *pkt;
    AVFrame *frame;

    pkt = av_packet_alloc();
    if (!pkt)
    {
        fprintf(stderr, "could not alloc av packet\n");
        exit(1);
    }
    codec = avcodec_find_decoder(AV_CODEC_ID_PNG);
    // codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
    if (!codec)
    {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }
    dec_ctx = avcodec_alloc_context3(codec);
    if (!dec_ctx)
    {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    if (avcodec_open2(dec_ctx, codec, NULL) < 0)
    {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }
    frame = av_frame_alloc();
    if (!frame)
    {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }

    pkt->size = numBytes;
    pkt->data = (unsigned char *)bs_buffer;
    // dec_ctx->pix_fmt = AV_PIX_FMT_RGB24;
    if (pkt->size)
    {
        int ret;
        ret = avcodec_send_packet(dec_ctx, pkt);

        if (ret < 0)
        {
            fprintf(stderr, "Error sending a packet for decoding\n");
            exit(1);
        }

        ret = avcodec_receive_frame(dec_ctx, frame);

        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        {
            fprintf(stderr, "Error could not receive frame\n");
            exit(1);
        }
        else if (ret < 0)
        {
            fprintf(stderr, "Error during decoding\n");
            exit(1);
        }

        fflush(stdout);

        data_on_device_mem = false;
        avframe_to_bm_image(handle, frame, &img, false);
        free(bs_buffer);
        avcodec_free_context(&dec_ctx);
        av_frame_free(&frame);
        av_packet_free(&pkt);
        return BM_SUCCESS;
    }
    else
    {
        fprintf(stderr, "Error decode png, can not read file size\n");
        free(bs_buffer);
        avcodec_free_context(&dec_ctx);
        av_frame_free(&frame);
        av_packet_free(&pkt);
        return BM_ERR_FAILURE;
    }
}

bm_status_t jpgDec(bm_handle_t &handle, string input_name, bm_image &img)
{
    AVInputFormat *iformat = nullptr;
    AVFormatContext *pFormatCtx = nullptr;
    AVCodecContext *dec_ctx = nullptr;
    AVCodec *pCodec = nullptr;
    AVDictionary *dict = nullptr;
    AVIOContext *avio_ctx = nullptr;
    AVFrame *pFrame = nullptr;
    AVFrame *I420Frame = nullptr;
    AVPacket pkt;

    int got_picture;
    FILE *infile;
    int numBytes;

    uint8_t *aviobuffer = nullptr;
    int aviobuf_size = 32 * 1024; // 32K
    uint8_t *bs_buffer = nullptr;
    int bs_size;
    bs_buffer_t bs_obj = {0, 0, 0};
    int tmp = 0;
    bm_status_t ret;

    infile = fopen(input_name.c_str(), "rb+");
    if (infile == nullptr)
    {
        cerr << "open file1 failed" << endl;
        goto Func_Exit;
    }

    fseek(infile, 0, SEEK_END);
    numBytes = ftell(infile);
    // cout << "infile size: " << numBytes << endl;
    fseek(infile, 0, SEEK_SET);

    bs_buffer = (uint8_t *)av_malloc(numBytes);
    if (bs_buffer == nullptr)
    {
        cerr << "av malloc for bs buffer failed" << endl;
        ret = BM_ERR_FAILURE;
        goto Func_Exit;
    }

    fread(bs_buffer, sizeof(uint8_t), numBytes, infile);
    fclose(infile);
    infile = nullptr;

    hardware_decode = determine_hardware_decode(bs_buffer);

    aviobuffer = (uint8_t *)av_malloc(aviobuf_size); // 32k
    if (aviobuffer == nullptr)
    {
        cerr << "av malloc for avio failed" << endl;
        ret = BM_ERR_FAILURE;
        goto Func_Exit;
    }

    bs_obj.start = bs_buffer;
    bs_obj.size = numBytes;
    bs_obj.pos = 0;
    avio_ctx = avio_alloc_context(aviobuffer, aviobuf_size, 0,
                                  (void *)(&bs_obj), read_buffer, NULL, NULL);
    if (avio_ctx == NULL)
    {
        cerr << "avio_alloc_context failed" << endl;
        ret = BM_ERR_FAILURE;
        goto Func_Exit;
    }

    pFormatCtx = avformat_alloc_context();
    pFormatCtx->pb = avio_ctx;

    /* mjpeg demuxer */
    iformat = av_find_input_format("mjpeg");
    if (iformat == NULL)
    {
        cerr << "av_find_input_format failed." << endl;
        ret = BM_ERR_FAILURE;
        goto Func_Exit;
    }

    /* Open an input stream */
    tmp = avformat_open_input(&pFormatCtx, NULL, iformat, NULL);
    if (tmp != 0)
    {
        cerr << "Couldn't open input stream.\n"
             << endl;
        ret = BM_ERR_FAILURE;
        goto Func_Exit;
    }

    /* HW JPEG decoder: jpeg_bm */
    pCodec = hardware_decode ? avcodec_find_decoder_by_name("jpeg_bm") : avcodec_find_decoder_by_name("mjpeg");
    if (pCodec == NULL)
    {
        cerr << "Codec not found." << endl;
        ret = BM_ERR_FAILURE;
        goto Func_Exit;
    }

    dec_ctx = avcodec_alloc_context3(pCodec);
    if (dec_ctx == NULL)
    {
        cerr << "Could not allocate video codec context!" << endl;
        ret = BM_ERR_FAILURE;
        goto Func_Exit;
    }

    av_dict_set_int(&dict, "chroma_interleave", 0, 0);
#define BS_MASK (1024 * 16 - 1)
    bs_size = (numBytes + BS_MASK) & (~BS_MASK);
#undef BS_MASK
#define JPU_PAGE_UNIT_SIZE 256
    /* Avoid the false alarm that bs buffer is empty (SA3SW-252) */
    if (bs_size - numBytes < JPU_PAGE_UNIT_SIZE)
        bs_size += 16 * 1024;
#undef JPU_PAGE_UNIT_SIZE
    bs_size /= 1024;
    av_dict_set_int(&dict, "bs_buffer_size", bs_size, 0);
    /* Extra frame buffers: "0" for still jpeg, at least "2" for mjpeg */
    av_dict_set_int(&dict, "num_extra_framebuffers", 0, 0);
    av_dict_set_int(&dict, "zero_copy", 0, 0);
    av_dict_set_int(&dict, "sophon_idx", bm_get_devid(handle), 0);
    tmp = avcodec_open2(dec_ctx, pCodec, &dict);
    if (tmp < 0)
    {
        cerr << "Could not open codec." << endl;
        ret = BM_ERR_FAILURE;
        goto Func_Exit;
    }

    pFrame = av_frame_alloc();
    if (pFrame == nullptr)
    {
        cerr << "av frame malloc failed" << endl;
        ret = BM_ERR_FAILURE;
        goto Func_Exit;
    }

    av_read_frame(pFormatCtx, &pkt);
    tmp = avcodec_decode_video2(dec_ctx, pFrame, &got_picture, &pkt);
    if (tmp < 0)
    {
        cerr << "Decode Error." << endl;
        ret = BM_ERR_FAILURE;
        goto Func_Exit;
    }

    // filter convert format to I420
    // TODO
    if (!hardware_decode)
    {
        int height = pFrame->height;
        int width = pFrame->width;
        uchar bgr_buffer[height * width * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int indexY = y * pFrame->linesize[0] + x;
                int indexU = (y / 2) * pFrame->linesize[1] + x;
                int indexV = (y / 2) * pFrame->linesize[2] + x;

                uchar Y = pFrame->data[0][indexY];
                uchar U = pFrame->data[1][indexU];
                uchar V = pFrame->data[2][indexV];

                double M[12] = {1.0000000000000000, 0.0000000000000000, 1.4018863751529200, -179.4414560195740000,
                                1.0000000000000000, -0.3458066722146720, -0.7149028511111540, 135.7708189857060000,
                                1.0000000000000000, 1.7709825540494100, 0.0000000000000000, -226.6857669183240000};

                int R = M[0] * Y + M[1] * (U) + M[2] * (V) + M[3];
                int G = M[4] * Y + M[5] * (U) + M[6] * (V) + M[7];
                int B = M[8] * Y + M[9] * (U) + M[10] * (V) + M[11];

                R = (R < 0) ? 0 : R;
                G = (G < 0) ? 0 : G;
                B = (B < 0) ? 0 : B;
                R = (R > 255) ? 255 : R;
                G = (G > 255) ? 255 : G;
                B = (B > 255) ? 255 : B;

                bgr_buffer[(y * width + x) * 3 + 0] = uchar(B);
                bgr_buffer[(y * width + x) * 3 + 1] = uchar(G);
                bgr_buffer[(y * width + x) * 3 + 2] = uchar(R);

                // std::cout << "B " << B << " G " << G << " R " << R << std::endl;
            }
        }

        bm_device_mem_t mem;
        bm_malloc_device_byte(handle, &mem, height * width * 3);
        bm_memcpy_s2d_partial(handle, mem, bgr_buffer, height * width * 3);

        bm_image_create(handle, height, width, FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE, &img);
        bm_image_attach(img, &mem);
        goto Func_Exit;
    }
    // vpp_convert do not support YUV422P, use libyuv to filter
    if (AV_PIX_FMT_YUVJ422P == pFrame->format)
    {
        I420Frame = av_frame_alloc();

        I420Frame->width = pFrame->width;
        I420Frame->height = pFrame->height;
        I420Frame->format = AV_PIX_FMT_YUV420P;

        I420Frame->linesize[0] = pFrame->linesize[0];
        I420Frame->linesize[1] = pFrame->linesize[1];
        I420Frame->linesize[2] = pFrame->linesize[2];

        I420Frame->data[0] = (uint8_t *)malloc(pFrame->linesize[0] * I420Frame->height * 2);
        I420Frame->data[1] = (uint8_t *)malloc(pFrame->linesize[1] * I420Frame->height / 2);
        I420Frame->data[2] = (uint8_t *)malloc(pFrame->linesize[2] * I420Frame->height / 2);
        libyuv::I422ToI420(pFrame->data[0], pFrame->linesize[0], pFrame->data[1], pFrame->linesize[1], pFrame->data[2], pFrame->linesize[2],
                           I420Frame->data[0], I420Frame->linesize[0], I420Frame->data[1], I420Frame->linesize[1], I420Frame->data[2], I420Frame->linesize[2],
                           I420Frame->width, I420Frame->height);

        av_frame_free(&pFrame);
        pFrame = I420Frame;
        data_on_device_mem = false;
    }

    avframe_to_bm_image(handle, pFrame, &img, true);

Func_Exit:
    av_packet_unref(&pkt);

    if (pFrame)
    {
        av_frame_free(&pFrame);
    }

    avformat_close_input(&pFormatCtx);

    if (avio_ctx)
    {
        av_freep(&avio_ctx->buffer);
        av_freep(&avio_ctx);
    }

    if (infile)
    {
        fclose(infile);
    }

    if (dict)
    {
        av_dict_free(&dict);
    }

    if (dec_ctx)
    {
        avcodec_close(dec_ctx);
    }
    if (bs_buffer)
    {
        av_free(bs_buffer);
    }
    return ret; // TODO
}