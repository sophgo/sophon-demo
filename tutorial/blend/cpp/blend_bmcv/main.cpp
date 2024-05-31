#include <string>
#include <fstream>
#include <iostream>

#include <bmlib_runtime.h>
#include <bmcv_api_ext.h>
#include <opencv2/opencv.hpp>

#define ALIGN(x, a)      (((x) + ((a)-1)) & ~((a)-1))

bool is_file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return (file.good()); 
}

// 只支持BM1688/CV186X(SoC)
static inline bm_status_t bm_image_destroy(bm_image& image){
  return bm_image_destroy(&image);
}

extern void bm_dem_read_bin(bm_handle_t handle, bm_device_mem_t* dmem, const char *input_name, unsigned int size) __attribute__((weak));

int main(int argc, char *argv[]){

    if (argc != 5) {
        std::cout << "USAGE:" << std::endl;
        std::cout << "  " << argv[1] << " <left image_path>" << std::endl;
        std::cout << "  " << argv[2] << " <right image_path>" << std::endl;
        std::cout << "  " << argv[3] << " <left wgt_path>" << std::endl;
        std::cout << "  " << argv[4] << " <right wgt_path>" << std::endl;
        exit(1);
    }
    std::string input1 = argv[1];
    std::string input2 = argv[2];
    std::string wgt_name1 = argv[3];
    std::string wgt_name2 = argv[4];
    bm_image_format_ext_ src_fmt=FORMAT_YUV420P;
    int dev_id = 0;

    if (!is_file_exists(input1)){
        std::cout << "[ERROR] input1 is not existed." << std::endl;
        exit(1);
    }

    if (!is_file_exists(input2)){
        std::cout << "[ERROR] input2 is not existed." << std::endl;
        exit(1);
    }

    if (!is_file_exists(wgt_name1)){
        std::cout << "[ERROR] wgt_name1 is not existed." << std::endl;
        exit(1);
    }

    if (!is_file_exists(wgt_name2)){
        std::cout << "[ERROR] wgt_name2 is not existed." << std::endl;
        exit(1);
    }
    char *wgt_name[2] = {NULL};
    wgt_name[0] = (char *)wgt_name1.c_str();
    wgt_name[1] = (char *)wgt_name2.c_str();



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

    
    struct stitch_param stitch_config;
    stitch_config.ovlap_attr.ovlp_lx[0] = 2304;
    stitch_config.ovlap_attr.ovlp_rx[0] = 4607;

    stitch_config.bd_attr.bd_lx[0] = 0;//left img, bd_attr from algo
    stitch_config.bd_attr.bd_rx[0] = 0;
    stitch_config.bd_attr.bd_lx[1] = 0;//right img, bd_attr from algo
    stitch_config.bd_attr.bd_rx[1] = 0;

    stitch_config.wgt_mode = BM_STITCH_WGT_YUV_SHARE;


    int wgtWidth = ALIGN(stitch_config.ovlap_attr.ovlp_rx[0] - stitch_config.ovlap_attr.ovlp_lx[0] + 1, 16);
    int wgtHeight = src_h1;


    // init params
    bm_handle_t bm_handle;
    bm_dev_request(&bm_handle, dev_id);
    int ret;

    for(int i = 0;i < 2; i++)
    {
        int wgt_len = wgtWidth * wgtHeight;
        if (stitch_config.wgt_mode == BM_STITCH_WGT_UV_SHARE)
        wgt_len = wgt_len << 1;

        bm_dem_read_bin(bm_handle, &stitch_config.wgt_phy_mem[0][i], wgt_name[i],  wgt_len);
    }

    // input image prepare
    int input_num=2;
    int dst_h= src_h1;
    int dst_w= src_w1 + stitch_config.ovlap_attr.ovlp_lx[0];

    // init bmimage
    bm_image src[input_num];
    ret =  cv::bmcv::toBMI(cv_image1, &src[0], true);
    if (ret != BM_SUCCESS) {
        std::cout << "Error! bm_image_from_mat: " << ret << std::endl;
        exit(1);
    }

    ret =  cv::bmcv::toBMI(cv_image2, &src[1], true);
    if (ret != BM_SUCCESS) {
        std::cout << "Error! bm_image_from_mat: " << ret << std::endl;
        exit(1);
    }

    bm_image input[input_num];
    ret = bm_image_create(bm_handle,
                        src[0].height,
                        src[0].width, src_fmt,
                        src[0].data_type, &input[0], NULL);
    bm_image_alloc_dev_mem(input[0], 1);
    ret = bmcv_image_storage_convert(bm_handle, 1, &src[0],&input[0]);

    ret = bm_image_create(bm_handle,
                        src[1].height,
                        src[1].width, src_fmt,
                        src[1].data_type, &input[1], NULL);
    bm_image_alloc_dev_mem(input[0], 1);
    ret = bmcv_image_storage_convert(bm_handle, 1, &src[1],&input[1]);

    bm_image blend_image;
    bm_image_create(bm_handle, dst_h, dst_w, input[0].image_format, DATA_TYPE_EXT_1N_BYTE, &blend_image,NULL);
    bm_image_alloc_dev_mem(blend_image, 1);

    bmcv_blending(bm_handle,input_num,input,blend_image,stitch_config);
        
    // save
    cv::Mat out_mat;
    cv::bmcv::toMAT(&blend_image, out_mat);
    cv::imwrite("blend_image.jpg", out_mat);

    bm_image_destroy(blend_image);
    bm_image_destroy(src[0]);
    bm_image_destroy(src[1]);
    bm_image_destroy(input[0]);
    bm_image_destroy(input[1]);
    bm_dev_free(bm_handle);

    std::cout << "[PASS]All done." << std::endl;

    return 0;
}
