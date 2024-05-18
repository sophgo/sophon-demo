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

    if (argc != 3) {
        std::cout << "USAGE:" << std::endl;
        std::cout << "  " << argv[0] << " <image_path>" << std::endl;
        std::cout << "  " << argv[1] << " <left image_path>" << std::endl;
        std::cout << "  " << argv[2] << " <right image_path>" << std::endl;
        exit(1);
    }
    std::string input1 = argv[1];
    std::string input2 = argv[2];
    int dev_id = 0;

    if (!is_file_exists(input1)){
        std::cout << "[ERROR] input1 is not existed." << std::endl;
        exit(1);
    }

    if (!is_file_exists(input2)){
        std::cout << "[ERROR] input2 is not existed." << std::endl;
        exit(1);
    }


    cv::Mat cv_image1 = cv::imread(input1, cv::IMREAD_COLOR, dev_id);
    if (!cv_image1.data){
        std::cout << "[ERROR]read input1 failed" << std::endl;
        exit(1);
    }
    cv::Mat cv_image2 = cv::imread(input2, cv::IMREAD_COLOR, dev_id);
    if (!cv_image1.data){
        std::cout << "[ERROR]read input2 failed" << std::endl;
        exit(1);
    }

    int src_w1 = cv_image1.cols;
    int src_h1 = cv_image1.rows;

    int src_w2 = cv_image2.cols;
    int src_h2 = cv_image2.rows;

    // init params
    bm_handle_t bm_handle;
    bm_dev_request(&bm_handle, dev_id);
    int ret;

    // init bmimage
    bm_image input_image1;
    bm_image input_image2;

    
    ret =  cv::bmcv::toBMI(cv_image1, &input_image1, true);
    if (ret != BM_SUCCESS) {
        std::cout << "Error! bm_image_from_mat: " << ret << std::endl;
        exit(1);
    }

    ret =  cv::bmcv::toBMI(cv_image2, &input_image2, true);
    if (ret != BM_SUCCESS) {
        std::cout << "Error! bm_image_from_mat: " << ret << std::endl;
        exit(1);
    }

    int input_num = 2;
    bmcv_rect_t dst_crop[input_num];

    dst_crop[0] = {
                    .start_x = 0,
                    .start_y = 0,
                    .crop_w = src_w1,
                    .crop_h = src_h1};
    dst_crop[1] = {
                    .start_x = src_w1 * 1 + 0,
                    .start_y = 0,
                    .crop_w = src_w2,
                    .crop_h = src_h2};


    int stitch_w=dst_crop[0].crop_w+dst_crop[1].crop_w;
    int stitch_h=std::max(dst_crop[0].crop_h, dst_crop[1].crop_h);
    
    bm_image stitch_image;

    bm_image_create(bm_handle, stitch_h, stitch_w, 
        input_image1.image_format, DATA_TYPE_EXT_1N_BYTE, &stitch_image, NULL);
    bm_image_alloc_dev_mem(stitch_image, BMCV_IMAGE_FOR_IN);

    
    bm_image src_img[2] = {input_image1, input_image2};

    ret = bmcv_image_vpp_stitch(bm_handle, input_num, src_img, stitch_image, dst_crop,
                        NULL);
    if (ret != BM_SUCCESS) {
        std::cout << "[ERROR]bmcv_image_vpp_stitch failed." << std::endl;
        exit(1);
    }
    
    // save
    cv::Mat out_mat;
    cv::bmcv::toMAT(&stitch_image, out_mat);
    cv::imwrite("stitch_image.jpg", out_mat);

    bm_image_destroy(stitch_image);
    bm_image_destroy(input_image1);
    bm_image_destroy(input_image2);
    bm_dev_free(bm_handle);

    std::cout << "[PASS]All done." << std::endl;

    return 0;
}
