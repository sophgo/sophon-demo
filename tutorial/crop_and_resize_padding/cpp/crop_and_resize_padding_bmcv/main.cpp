//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <string>
#include <fstream>
#include <iostream>

#include <bmlib_runtime.h>
#include <bmcv_api_ext.h>
#include <opencv2/opencv.hpp>


bool is_file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return (file.good()); 
}

#ifndef BMCV_VERSION_MAJOR
#define BMCV_VERSION_MAJOR 1
#endif
/*for multi version compatible*/
#if BMCV_VERSION_MAJOR > 1
typedef bmcv_padding_attr_t bmcv_padding_atrr_t;
/**
 * @name    bm_image_destroy
 * @brief   To solve incompatible issue in a2 sdk.
 * @ingroup bmcv
 *
 * @param [image]        input bm_image
 * @retval BM_SUCCESS    change success.
 * @retval other values  change failed.
 */
static inline bm_status_t bm_image_destroy(bm_image& image){
  return bm_image_destroy(&image);
}
#endif

float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
  float ratio;
  float r_w = (float)dst_w / src_w;
  float r_h = (float)dst_h / src_h;
  if (r_h > r_w){
    *pIsAligWidth = true;
    ratio = r_w;
  }
  else{
    *pIsAligWidth = false;
    ratio = r_h;
  }
  return ratio;
}

int main(int argc, char *argv[]){

    if (argc != 2) {
        std::cout << "USAGE:" << std::endl;
        std::cout << "  " << argv[0] << " <image_path>" << std::endl;
        exit(1);
    }
    std::string input_path = argv[1];
    int dev_id = 0;

    if (!is_file_exists(input_path)){
        std::cout << "[ERROR]" << input_path << " is not existed." << std::endl;
        exit(1);
    }

    cv::Mat cv_image = cv::imread(input_path, cv::IMREAD_COLOR, dev_id);
    if (!cv_image.data){
        std::cout << "[ERROR]read failed" << std::endl;
        exit(1);
    }

    // init params
    bm_handle_t bm_handle;
    bm_dev_request(&bm_handle, dev_id);
    int ret;
    int output_w = 640;
    int output_h = 640;

    // init bmimage
    bm_image input_image;
    bm_image resized_image;

    ret =  cv::bmcv::toBMI(cv_image, &input_image, true);
    if (ret != BM_SUCCESS) {
        std::cout << "Error! bm_image_from_mat: " << ret << std::endl;
        exit(1);
    }
    
    bm_image_create(bm_handle, output_h, output_w, 
        input_image.image_format, DATA_TYPE_EXT_1N_BYTE, &resized_image, NULL);

    // crop
    bm_image input_image_aligned;
    bool need_copy = input_image.width & (64-1);
    if(need_copy){
      int stride1[3], stride2[3];
      bm_image_get_stride(input_image, stride1);
      stride2[0] = FFALIGN(stride1[0], 64);
      stride2[1] = FFALIGN(stride1[1], 64);
      stride2[2] = FFALIGN(stride1[2], 64);
      bm_image_create(bm_handle, input_image.height, input_image.width,
          input_image.image_format, input_image.data_type, &input_image_aligned, stride2);

      bm_image_alloc_dev_mem(input_image_aligned, BMCV_IMAGE_FOR_IN);
      bmcv_copy_to_atrr_t copyToAttr;
      memset(&copyToAttr, 0, sizeof(copyToAttr));
      copyToAttr.start_x = 0;
      copyToAttr.start_y = 0;
      copyToAttr.if_padding = 1;
      bmcv_image_copy_to(bm_handle, copyToAttr, input_image, input_image_aligned);
    } else {
      input_image_aligned = input_image;
    }

    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(input_image_aligned.width, input_image_aligned.height, output_w, output_h, &isAlignWidth);
    bmcv_padding_atrr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_sty = 0;
    padding_attr.dst_crop_stx = 0;
    padding_attr.padding_b = 114;
    padding_attr.padding_g = 114;
    padding_attr.padding_r = 114;
    padding_attr.if_memset = 1;
    if (isAlignWidth) {
      padding_attr.dst_crop_h = input_image_aligned.height*ratio;
      padding_attr.dst_crop_w = output_w;

      int ty1 = (int)((output_h - padding_attr.dst_crop_h) / 2);
      padding_attr.dst_crop_sty = ty1;
      padding_attr.dst_crop_stx = 0;
    }else{
      padding_attr.dst_crop_h = output_h;
      padding_attr.dst_crop_w = input_image_aligned.width*ratio;

      int tx1 = (int)((output_w - padding_attr.dst_crop_w) / 2);
      padding_attr.dst_crop_sty = 0;
      padding_attr.dst_crop_stx = tx1;
    }

    bmcv_rect_t crop_rect{0, 0, (unsigned int)input_image_aligned.width, (unsigned int)input_image_aligned.height};

    ret = bmcv_image_vpp_convert_padding(bm_handle, 1, input_image_aligned, &resized_image,
        &padding_attr, &crop_rect, BMCV_INTER_NEAREST);
    if (ret != BM_SUCCESS) {
        std::cout << "[ERROR]bmcv_image_vpp_convert failed." << std::endl;
        exit(1);
    }
    
    if(need_copy) bm_image_destroy(input_image_aligned);


    // save
    cv::Mat out_mat;
    cv::bmcv::toMAT(&resized_image, out_mat);
    cv::imwrite("crop_and_resize_padding.jpg", out_mat);

    bm_image_destroy(resized_image);
    bm_image_destroy(input_image);
    bm_dev_free(bm_handle);

    std::cout << "[PASS]All done." << std::endl;

    return 0;
}
