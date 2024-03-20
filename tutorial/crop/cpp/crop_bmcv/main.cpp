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
    int input_width = cv_image.cols;
    int input_height = cv_image.rows;

    // init params
    bm_handle_t bm_handle;
    bm_dev_request(&bm_handle, dev_id);
    int ret;

    // init bmimage
    bm_image input_image;
    bm_image crop_image;

    ret =  cv::bmcv::toBMI(cv_image, &input_image, true);
    if (ret != BM_SUCCESS) {
        std::cout << "Error! bm_image_from_mat: " << ret << std::endl;
        exit(1);
    }

    int crop_x = input_image.width / 4;
    int crop_y = input_image.height / 4;
    int crop_w = input_image.width / 2;
    int crop_h = input_image.height / 2;

    bm_image_create(bm_handle, crop_h, crop_w, 
        input_image.image_format, DATA_TYPE_EXT_1N_BYTE, &crop_image, NULL);

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

    bmcv_rect_t rect[1] = {{(unsigned int)crop_x, (unsigned int)crop_y, (unsigned int)crop_w, (unsigned int)crop_h}};
    ret = bmcv_image_vpp_convert(bm_handle, 1, input_image_aligned, &crop_image, rect);
    if (ret != BM_SUCCESS) {
        std::cout << "[ERROR]bmcv_image_vpp_convert failed." << std::endl;
        exit(1);
    }
    
    if(need_copy) bm_image_destroy(input_image_aligned);

    // save
    cv::Mat out_mat;
    cv::bmcv::toMAT(&crop_image, out_mat);
    cv::imwrite("crop.jpg", out_mat);

    bm_image_destroy(crop_image);
    bm_image_destroy(input_image);
    bm_dev_free(bm_handle);

    std::cout << "[PASS]All done." << std::endl;

    return 0;
}
