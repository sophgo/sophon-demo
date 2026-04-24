#include "dpu.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cstring>
#include <bmcv_api_ext.h>
#include <bmlib_runtime.h>
#include <stdexcept>


DPU::DPU(int dev_id, bool debug, int width, int height) 
    : debug_(debug)
{
    // 1. 对齐宽高
    width_ = (width < 64) ? 64 : ((width > 1920) ? 1920 : FFALIGN(width, align_to_width_));
    height_ = (height < 64) ? 64 : ((height > 1080) ? 1080 : FFALIGN(height, align_to_height_));

    // 2. 初始化handle
    int ret = bm_dev_request(&handle_, dev_id);
    if (ret != BM_SUCCESS) {
        std::cerr << "Failed to request device" << std::endl;
        if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    }
    // 3. 初始化中间bm_image
    ret = bm_image_create(handle_, height_, width_, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, &aligned_left_);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    ret = bm_image_alloc_dev_mem(aligned_left_, 1);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }

    ret = bm_image_create(handle_, height_, width_, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, &aligned_right_);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    ret = bm_image_alloc_dev_mem(aligned_right_, 1);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }

    // 4. 初始化深度图
    ret = bm_image_create(handle_, height_, width_, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, &depth_img_);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    ret = bm_image_alloc_dev_mem(depth_img_, 1);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }

    // 5. 设置SGBM默认参数
    sgbm_params_.bfw_mode_en = DPU_BFW_MODE_7x7;
    sgbm_params_.disp_range_en = BMCV_DPU_DISP_RANGE_128;
    sgbm_params_.disp_start_pos = 0;
    sgbm_params_.dcc_dir_en = BMCV_DPU_DCC_DIR_A13;
    sgbm_params_.dpu_census_shift = 2;
    sgbm_params_.dpu_rshift1 = 0;
    sgbm_params_.dpu_rshift2 = 2;
    sgbm_params_.dpu_ca_p1 = 2880;
    sgbm_params_.dpu_ca_p2 = 14400;
    sgbm_params_.dpu_uniq_ratio = 0;
    sgbm_params_.dpu_disp_shift = 4;

    // 6. 设置FGS默认参数
    fgs_params_.fgs_max_count = 19;           // 设置为3次
    fgs_params_.fgs_max_t = 3;               // 设置为3次迭代
    fgs_params_.fxbase_line = 864000;           // 设置基线值为120mm
    fgs_params_.depth_unit_en = BMCV_DPU_DEPTH_UNIT_MM;  // 使用毫米作为深度单位

    initialized_ = true;
}

DPU::~DPU() {
    release();
}

void DPU::release() noexcept {
    if (initialized_) {
        bm_image_destroy(&aligned_left_);
        bm_image_destroy(&aligned_right_);
        bm_image_destroy(&depth_img_);
        bm_dev_free(handle_);
        initialized_ = false;
    }
    else {
        return;
    }
}

bool DPU::validateImageFormat(const bm_image& img, bool is_input) const {
    // 检查图像格式
    if (img.image_format != FORMAT_GRAY) {
        std::cerr << "Image format must be GRAY" << std::endl;
        return false;
    }

    // 检查数据类型
    if (is_input && img.data_type != DATA_TYPE_EXT_1N_BYTE) {
        std::cerr << "Input image data type must be 8-bit" << std::endl;
        return false;
    }

    // 输出图像可以是8位或16位
    if (!is_input && img.data_type != DATA_TYPE_EXT_1N_BYTE && 
        img.data_type != DATA_TYPE_EXT_U16) {
        std::cerr << "Output image data type must be 8-bit or 16-bit" << std::endl;
        return false;
    }

    return true;
}

bool DPU::validateImageSize(const bm_image& left, const bm_image& right) const {
    // 打印图像尺寸信息
    std::cout << "Validating image sizes:" << std::endl;
    std::cout << "Left image: " << left.width << "x" << left.height << std::endl;
    std::cout << "Right image: " << right.width << "x" << right.height << std::endl;
    
    // 检查分辨率范围
    if (left.height < 64 || left.height > 1080 || 
        left.width < 64 || left.width > 1920) {
        std::cerr << "Image size must be between 64x64 and 1920x1080" << std::endl;
        return false;
    }

    // 检查左右图像尺寸是否相同
    if (left.height != right.height || left.width != right.width) {
        std::cerr << "Left and right images must have the same size" << std::endl;
        return false;
    }

    // 检查width是否4对齐
    if (left.width % 4 != 0) {
        std::cerr << "Image width must be 4-byte aligned" << std::endl;
        std::cerr << "Current width: " << left.width << ", remainder: " << left.width % 4 << std::endl;
        return false;
    }

    // 检查height是否2对齐
    if (left.height % 2 != 0) {
        std::cerr << "Image height must be 2-byte aligned" << std::endl;
        std::cerr << "Current height: " << left.height << ", remainder: " << left.height % 2 << std::endl;
        return false;
    }

    return true;
}

bool DPU::validateDisparityRange(const bm_image& right_img) const {
    // 计算视差搜索范围
    int disp_range = 0;
    switch (sgbm_params_.disp_range_en) {
        case BMCV_DPU_DISP_RANGE_16: disp_range = 16; break;
        case BMCV_DPU_DISP_RANGE_32: disp_range = 32; break;
        case BMCV_DPU_DISP_RANGE_48: disp_range = 48; break;
        case BMCV_DPU_DISP_RANGE_64: disp_range = 64; break;
        case BMCV_DPU_DISP_RANGE_80: disp_range = 80; break;
        case BMCV_DPU_DISP_RANGE_96: disp_range = 96; break;
        case BMCV_DPU_DISP_RANGE_112: disp_range = 112; break;
        case BMCV_DPU_DISP_RANGE_128: disp_range = 128; break;
        default: disp_range = 16; break;
    }

    // 检查右图像宽度是否满足要求
    if (right_img.width < (sgbm_params_.disp_start_pos + disp_range)) {
        std::cerr << "Right image width must be >= disp_start_pos + disp_range" << std::endl;
        return false;
    }

    return true;
}

int DPU::pre_process(bm_image& input_image, bm_image& converted_img) {
    int ret;
    // 打印原图宽、高、stride
    int stride[3];
    bm_image_get_stride(input_image, stride);
    std::cout << "DPU Original image width: " << input_image.width << ", height: " << input_image.height << ", stride[0]: " << stride[0] << std::endl; 
    std::cout << "DPU Aligned image width: " << width_ << ", height: " << height_ << std::endl; 
    
    // 1. align image size and convert
    if(input_image.width <= width_ || input_image.height <= height_){
        bmcv_copy_to_atrr_t copyToAttr;
        memset(&copyToAttr, 0, sizeof(copyToAttr));
        copyToAttr.start_x = 0;
        copyToAttr.start_y = height_ - input_image.height;
        copyToAttr.if_padding = 1;
        copyToAttr.padding_r = 0;
        copyToAttr.padding_g = 0;
        copyToAttr.padding_b = 0;
        ret = bmcv_image_copy_to(handle_, copyToAttr, input_image, converted_img);
        if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }  
    // 2. only convert image 
    }else{
        bmcv_image_storage_convert(handle_, 1, &input_image, &converted_img);
    }

    return 0;
}


int DPU::process(bm_image& left_img, bm_image& right_img, 
                bm_image& depth_img, DPUMode mode, bmcv_dpu_sgbm_mode sgbm_mode, bmcv_dpu_online_mode online_mode) {
    int ret;
    if (!initialized_) {
        std::cerr << "DPU not initialized" << std::endl;
        return -1;
    }
    m_ts->save("dpu preprocess two images");

    // 1. 先进行预处理，得到对齐后的灰度图
    if (pre_process(left_img, aligned_left_) != 0 ||
        pre_process(right_img, aligned_right_) != 0) {
        return -1;
    }
    m_ts->save("dpu preprocess two images");

    // 2. 参数验证（使用预处理后的图像）
    if (!validateImageFormat(aligned_left_, true) || 
        !validateImageFormat(aligned_right_, true)) {
        return -1;
    }

    if (!validateImageSize(aligned_left_, aligned_right_)) {
        return -1;
    }

    if (mode == DPUMode::SGBM && !validateDisparityRange(aligned_right_)) {
        return -1;
    }

    m_ts->save("dpu process");
    // 5. 使用预处理后的图像进行处理
    switch (mode) {
        case DPUMode::SGBM:
            ret = bmcv_dpu_sgbm_disp(handle_, 
                                    &aligned_left_, 
                                    &aligned_right_, 
                                    &depth_img_, 
                                    &sgbm_params_, 
                                    sgbm_mode);
            break;
        case DPUMode::ONLINE:
            ret = bmcv_dpu_online_disp(handle_, 
                                        &aligned_left_, 
                                        &aligned_right_, 
                                        &depth_img_, 
                                        &sgbm_params_, 
                                        &fgs_params_, 
                                        online_mode);
            break;
        default:
            std::cerr << "Invalid DPU mode" << std::endl;
            return -1;
    }
    m_ts->save("dpu process");

    depth_img = depth_img_;

    if (ret != BM_SUCCESS) {
        std::cerr << "Failed to process images" << std::endl;
        return -1;
    }

    // debug
    if (debug_) {
        if (!save_image(left_img, "debug/left_img.jpg")) {
            std::cerr << "Failed to save left image" << std::endl;
            return -1; 
        }   
        if (!save_image(right_img, "debug/right_img.jpg")) {
            std::cerr << "Failed to save right image" << std::endl;
            return -1;
        }
        if (!save_image(aligned_left_, "debug/preprocess_img_aligned_left.jpg")) {
            std::cerr << "Failed to save preprocess image" << std::endl;
            return -1;
        }
        if (!save_image(aligned_right_, "debug/preprocess_img_aligned_right.jpg")) {
            std::cerr << "Failed to save preprocess image" << std::endl;
            return -1;
        }

        if (!save_image(depth_img, "debug/depth_img.jpg")) {
            std::cerr << "Failed to save depth image" << std::endl;
            return -1;
        }
        // // 保存bin文件
        // bm_write_bin(left_img, "debug/left_img.bin");
        // bm_write_bin(right_img, "debug/right_img.bin");
        // bm_write_bin(depth_img, "debug/depth_img.bin");
    }

    return 0;
}


// 保存图像
bool DPU::save_image(bm_image& image, const std::string& output_path) {
    void* jpeg_data = nullptr;

    // 创建临时图像用于格式转换
    bm_image temp_image;
    bm_status_t ret = bm_image_create(handle_, image.height, image.width, 
                                    FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, &temp_image);

    if (ret != BM_SUCCESS) {
        return false;
    }
    
    ret = bm_image_alloc_dev_mem(temp_image, 1);
    if (ret != BM_SUCCESS) {
        bm_image_destroy(&temp_image);
        return false;
    }
    
    // 转换图像格式
    ret = bmcv_image_storage_convert(handle_, 1, &image, &temp_image);
    if (ret != BM_SUCCESS) {
        bm_image_destroy(&temp_image);
        return false;
    }

    size_t out_size = 0;
    if (bmcv_image_jpeg_enc(handle_, 1, &temp_image, &jpeg_data, &out_size) != BM_SUCCESS) {
        std::cerr << "Failed to encode image" << std::endl;
        return false;
    } else {
        std::ofstream file(output_path, std::ios::binary);
        if (file) {
            file.write(static_cast<char*>(jpeg_data), out_size);
            file.close();
        }
        free(jpeg_data);
    }
    // 清理临时图像
    bm_image_destroy(&temp_image);   
    return true;
}