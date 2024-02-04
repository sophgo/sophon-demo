// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#include"ff_avframe_convert.h"

int map_bmformat_to_avformat(int bmformat)
{
    int format;
    switch(bmformat){
        case FORMAT_YUV420P: format = AV_PIX_FMT_YUV420P; break;
        case FORMAT_YUV422P: format = AV_PIX_FMT_YUV422P; break;
        case FORMAT_YUV444P: format = AV_PIX_FMT_YUV444P; break;
        case FORMAT_NV12:    format = AV_PIX_FMT_NV12; break;
        case FORMAT_NV16:    format = AV_PIX_FMT_NV16; break;
        case FORMAT_GRAY:    format = AV_PIX_FMT_GRAY8; break;
        case FORMAT_RGBP_SEPARATE: format = AV_PIX_FMT_GBRP; break;
        default: printf("unsupported image format %d\n", bmformat); return -1;
    }
    return format;
}
int map_avformat_to_bmformat(int avformat)
{
    int format;
    switch(avformat){
        case AV_PIX_FMT_YUV420P: format = FORMAT_YUV420P; break;
        case AV_PIX_FMT_YUV422P: format = FORMAT_YUV422P; break;
        case AV_PIX_FMT_YUV444P: format = FORMAT_YUV444P; break;
        case AV_PIX_FMT_NV12:    format = FORMAT_NV12; break;
        case AV_PIX_FMT_NV16:    format = FORMAT_NV16; break;
        case AV_PIX_FMT_GRAY8:   format = FORMAT_GRAY; break;
        case AV_PIX_FMT_GBRP:    format = FORMAT_RGBP_SEPARATE; break;
        default: printf("unsupported av_pix_format %d\n", avformat); return -1;
    }

    return format;
}
int bm_image_sizeof_data_type(bm_image *image){

    switch(image->data_type){
    case DATA_TYPE_EXT_FLOAT32:
        return sizeof(float);
    case DATA_TYPE_EXT_1N_BYTE:
    case DATA_TYPE_EXT_1N_BYTE_SIGNED:
        return sizeof(char);
    case DATA_TYPE_EXT_4N_BYTE:
    case DATA_TYPE_EXT_4N_BYTE_SIGNED:
        return sizeof(char) * 4;
    default:
        return 1;
    }
}

void bmBufferDeviceMemFree(void *opaque, uint8_t *data)
{
    if(opaque == NULL){
        printf("parameter error\n");
    }
    transcode_t *testTranscoed = (transcode_t *)opaque;
    av_freep(&testTranscoed->buf0);
    testTranscoed->buf0 = NULL;

    int ret =  0;
    ret = bm_image_destroy(*(testTranscoed->bmImg));
    if(testTranscoed->bmImg){
        free(testTranscoed->bmImg);
        testTranscoed->bmImg =NULL;
    }
    if(ret != 0)
        printf("bm_image destroy failed\n");
    free(testTranscoed);
    testTranscoed = NULL;
    return ;
}
static void bmBufferDeviceMemFree2(void *opaque, uint8_t *data)
{
    return ;
}

int avframe_to_bm_image(bm_handle_t &bm_handle,AVFrame &in, bm_image &out){

    int plane                 = 0;
    int data_five_denominator = -1;
    int data_six_denominator  = -1;
    static int mem_flags = USEING_MEM_HEAP1;


    switch(in.format){
    case AV_PIX_FMT_GRAY8:
        plane = 1;
        data_five_denominator = -1;
        data_six_denominator = -1;
        break;
    case AV_PIX_FMT_YUV420P:
        plane = 3;
        data_five_denominator = 4;
        data_six_denominator = 4;
        break;
    case AV_PIX_FMT_NV12:
        plane = 2;
        data_five_denominator = 2;
        data_six_denominator = -1;
        break;
    case AV_PIX_FMT_YUV422P:
        plane = 3;
        data_five_denominator = 2;
        data_six_denominator = 2;
        break;
    case AV_PIX_FMT_NV16:
        plane = 2;
        data_five_denominator = 2;
        data_six_denominator = -1;
        break;
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_GBRP:
        plane = 3;
        data_five_denominator = 1;
        data_six_denominator = 1;
        break;
    default:
        printf("unsupported format, only gray,nv12,yuv420p,nv16,yuv422p horizontal,yuv444p,rgbp supported\n");
        break;
    }

    if (in.channel_layout == 101) {/* COMPRESSED NV12 FORMAT */
        if ((0 == in.height) || (0 == in.width) || \
         (0 == in.linesize[4]) || (0 == in.linesize[5]) || (0 == in.linesize[6]) || (0 == in.linesize[7]) || \
         (0 == in.data[4]) || (0 == in.data[5]) || (0 == in.data[6]) || (0 == in.data[7])) {
          printf("bm_image_from_frame: get yuv failed!!");
          return BM_ERR_PARAM;
        }
        bm_image cmp_bmimg;
        bm_image_create (bm_handle,
                  in.height,
                  in.width,
                  FORMAT_COMPRESSED,
                  DATA_TYPE_EXT_1N_BYTE,
                  &cmp_bmimg);

        bm_device_mem_t input_addr[4];
        int size = in.height * in.linesize[4];
        input_addr[0] = bm_mem_from_device((unsigned long long)in.data[6], size);
        size = (in.height / 2) * in.linesize[5];
        input_addr[1] = bm_mem_from_device((unsigned long long)in.data[4], size);
        size = in.linesize[6];
        input_addr[2] = bm_mem_from_device((unsigned long long)in.data[7], size);
        size = in.linesize[7];
        input_addr[3] = bm_mem_from_device((unsigned long long)in.data[5], size);
        bm_image_attach(cmp_bmimg, input_addr);
        bm_image_create (bm_handle,
                in.height,
                in.width,
                FORMAT_YUV420P,
                DATA_TYPE_EXT_1N_BYTE,
                &out);
        //bm_image_dev_mem_alloc(out);
        if(mem_flags == USEING_MEM_HEAP2 && bm_image_alloc_dev_mem_heap_mask(out,USEING_MEM_HEAP2) != BM_SUCCESS){
            mem_flags = USEING_MEM_HEAP1;
        }
        if(mem_flags == USEING_MEM_HEAP1 && bm_image_alloc_dev_mem_heap_mask(out,USEING_MEM_HEAP1) != BM_SUCCESS){
            printf("bmcv allocate mem failed!!!");
        }

        bmcv_rect_t crop_rect = {0, 0, in.width, in.height};
        bmcv_image_vpp_convert(bm_handle, 1, cmp_bmimg, &out, &crop_rect);
        bm_image_destroy(cmp_bmimg);
    }
    else {
        int stride[3];
        bm_image_format_ext bm_format;
        bm_device_mem_t input_addr[3] = {0};
        if(plane == 1){
            if ((0 == in.height) || (0 == in.width) ||(0 == in.linesize[4]) || (0 == in.data[4])) {
                return BM_ERR_PARAM;
            }
            stride[0] = in.linesize[4];
        }
        else if (plane == 2){
            if ((0 == in.height) || (0 == in.width) || \
                (0 == in.linesize[4]) || (0 == in.linesize[5]) || \
                (0 == in.data[4]) || (0 == in.data[5])) {
                return BM_ERR_PARAM;
            }

            stride[0] = in.linesize[4];
            stride[1] = in.linesize[5];
        }
        else if(plane == 3){
            if ((0 == in.height) || (0 == in.width) || \
                (0 == in.linesize[4]) || (0 == in.linesize[5]) || (0 == in.linesize[6]) || \
                (0 == in.data[4]) || (0 == in.data[5]) || (0 == in.data[6])) {
              return BM_ERR_PARAM;
            }

            stride[0] = in.linesize[4];
            stride[1] = in.linesize[5];
            stride[2] = in.linesize[6];
        }

        bm_format = (bm_image_format_ext)map_avformat_to_bmformat(in.format);
        bm_image_create (bm_handle,
                  in.height,
                  in.width,
                  bm_format,
                  DATA_TYPE_EXT_1N_BYTE,
                  &out,
                  stride);

        int size = in.height * stride[0];
        input_addr[0] = bm_mem_from_device((unsigned long long)in.data[4], size);
        if(data_five_denominator != -1 ){
            size = in.height * stride[1] / data_five_denominator;
            input_addr[1] = bm_mem_from_device((unsigned long long)in.data[5], size);
        }
        if(data_six_denominator != -1){
            size = in.height * stride[2] / data_six_denominator;
            input_addr[2] = bm_mem_from_device((unsigned long long)in.data[6], size);
        }
        bm_image_attach(out, input_addr);
    }
   return BM_SUCCESS;
}

int bm_image_to_avframe(bm_handle_t &bm_handle,bm_image *in,AVFrame *out){
    transcode_t *ImgOut  = NULL;
    ImgOut = (transcode_t *)malloc(sizeof(transcode_t));
    ImgOut->bmImg = in;
    bm_image_format_info_t image_info;
    int idx       = 0;
    int plane     = 0;
    if(in == NULL || out == NULL){
        free(ImgOut);
        return -1;
    }

    if(ImgOut->bmImg->image_format == FORMAT_NV12){
        plane = 2;
    }
    else if(ImgOut->bmImg->image_format == FORMAT_YUV420P){
        plane = 3;
    }
    else{
        free(ImgOut);
        free(in);
        return -1;
    }

    out->format = (AVPixelFormat)map_bmformat_to_avformat(ImgOut->bmImg->image_format);
    out->height = ImgOut->bmImg->height;
    out->width = ImgOut->bmImg->width;

    if(ImgOut->bmImg->width > 0 && ImgOut->bmImg->height > 0
        && ImgOut->bmImg->height * ImgOut->bmImg->width <= 8192*4096) {
        ImgOut->buf0 = (uint8_t*)av_malloc(ImgOut->bmImg->height * ImgOut->bmImg->width * 3 / 2);
        ImgOut->buf1 = ImgOut->buf0 + (unsigned int)(ImgOut->bmImg->height * ImgOut->bmImg->width);
        if(plane == 3){
            ImgOut->buf2 = ImgOut->buf0 + (unsigned int)(ImgOut->bmImg->height * ImgOut->bmImg->width * 5 / 4);
        }
    }

    out->buf[0] = av_buffer_create(ImgOut->buf0,ImgOut->bmImg->width * ImgOut->bmImg->height,
        bmBufferDeviceMemFree,ImgOut,AV_BUFFER_FLAG_READONLY);
    out->buf[1] = av_buffer_create(ImgOut->buf1,ImgOut->bmImg->width * ImgOut->bmImg->height / 2 /2 ,
        bmBufferDeviceMemFree2,NULL,AV_BUFFER_FLAG_READONLY);
    out->data[0] = ImgOut->buf0;
    out->data[1] = ImgOut->buf0;

    if(plane == 3){
        out->buf[2] = av_buffer_create(ImgOut->buf2,ImgOut->bmImg->width * ImgOut->bmImg->height / 2 /2 ,
            bmBufferDeviceMemFree2,NULL,AV_BUFFER_FLAG_READONLY);
        out->data[2] = ImgOut->buf0;
    }

    if(plane == 3 && !out->buf[2]){
        av_buffer_unref(&out->buf[0]);
        av_buffer_unref(&out->buf[1]);
        av_buffer_unref(&out->buf[2]);
        free(ImgOut);
        free(in);
        return -1;
    }
    else if(plane == 2 && !out->buf[1]){
        av_buffer_unref(&out->buf[0]);
        av_buffer_unref(&out->buf[1]);
        free(ImgOut);
        free(in);
        return -1;
    }

    bm_device_mem_t mem_tmp[3];
    if(bm_image_get_device_mem(*(ImgOut->bmImg),mem_tmp) != BM_SUCCESS ){
        free(ImgOut);
        free(in);
        return -1;
    }
    if(bm_image_get_format_info(ImgOut->bmImg, &image_info) != BM_SUCCESS ){
        free(ImgOut);
        free(in);
        return -1;
    }
    for (idx=0; idx< plane; idx++) {
        out->data[4+idx]     = (uint8_t *)mem_tmp[idx].u.device.device_addr;
        out->linesize[idx]   = image_info.stride[idx];
        out->linesize[4+idx] = image_info.stride[idx];
    }
    return 0;
}