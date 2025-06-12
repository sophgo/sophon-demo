//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "dwa.hpp"

Dwa::Dwa(){}

Dwa::~Dwa(){
    release();
}

int Dwa::init(std::string grid_path, int input_width, int input_height, bm_image_format_ext input_dwa_fmt, bool debug){
    // 初始化bm_handle
    bm_dev_request(&handle, 0);
    this->debug = debug;

    // 读取grid文件，初始化ldc_attr
    FILE* fp = fopen(grid_path.c_str(), "rb");
    fseek(fp, 0, SEEK_END);
    grid_size = ftell(fp);
    grid_buffer = (char*)malloc(grid_size);
    memset(grid_buffer, 0, grid_size);
    rewind(fp);
    fread(grid_buffer, grid_size, 1, fp);
    fclose(fp);
    ldc_attr.grid_info.u.system.system_addr = (void*)grid_buffer;
    ldc_attr.grid_info.size = grid_size;

    // converted_image 
    ret = bm_image_create(handle, 
        input_height, input_width, 
        input_dwa_fmt, 
        DATA_TYPE_EXT_1N_BYTE, &converted_image, NULL);
    ret = bm_image_alloc_dev_mem(converted_image, 1);
    if(ret != BM_SUCCESS){
        std::cout << "bm_image_create converted_image failed" << std::endl;
        return ret;
    }

    // dwa_image
    ret = bm_image_create(handle, 
        input_height, input_width, 
        input_dwa_fmt, 
        DATA_TYPE_EXT_1N_BYTE, &dwa_image, NULL);
    ret = bm_image_alloc_dev_mem(dwa_image, 1);
    if(ret != BM_SUCCESS){
        std::cout << "bm_image_create dwa_image failed" << std::endl;
        return ret;
    }
    return ret;
}

int Dwa::release(){
    if(grid_buffer){
        free(grid_buffer);
        grid_buffer = nullptr;
    }

    ret = bm_image_destroy(&converted_image);
    if(ret != BM_SUCCESS){
        std::cout << "bm_image_destroy converted_image failed" << std::endl;
        return ret;
    }
    ret = bm_image_destroy(&dwa_image);
    if(ret != BM_SUCCESS){
        std::cout << "bm_image_destroy dwa_image failed" << std::endl;
        return ret;
    }

    if(handle){
        bm_dev_free(handle);
        handle = nullptr;
    }

    return 0;
}

int Dwa::process_image(bm_image input_image){
    // 转换图像格式
    ret = bmcv_image_storage_convert(handle, 1,
                                    &input_image, &converted_image);
    if(ret != BM_SUCCESS){
        std::cout << "bmcv_image_storage_convert failed" << std::endl;
        return ret;
    }

    std::cout<<"=============start dwa_gdc bmimage======================" <<std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    // 调用dwa处理
    ret = bmcv_dwa_gdc(handle, converted_image, dwa_image, ldc_attr);
    if(ret != BM_SUCCESS){
        std::cout << "bmcv_dwa_gdc failed" << std::endl;
        return ret;
    }
    std::cout << "dwa_gdc done" << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "dwa_gdc time: " << duration.count() << "ms" << std::endl;
    std::cout<<"=============end dwa_gdc bmimage========================" <<std::endl;

    // 保存图像
    if(debug){
        std::string filename = "./converted_image.bmp";
        bm_image_write_to_bmp(converted_image, filename.c_str());
        std::cout << "格式转换处理后的图像已保存到: " << filename << std::endl;
        std::cout << "确认grid_data: " << std::endl;
        char* grid_data = static_cast<char*>(ldc_attr.grid_info.u.system.system_addr);
        for(int i = 0; i < 8; i++){
            std::cout << grid_data[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "确认converted_image格式: " << std::endl;
        std::cout << "converted_image.width: " << converted_image.width << std::endl;
        std::cout << "converted_image.height: " << converted_image.height << std::endl;
        std::cout << "converted_image.image_format: " << converted_image.image_format << std::endl;
        std::cout << "converted_image.data_type: " << converted_image.data_type << std::endl;

        std::cout << "确认dwa_image格式: " << std::endl;
        std::cout << "dwa_image.width: " << dwa_image.width << std::endl;
        std::cout << "dwa_image.height: " << dwa_image.height << std::endl;
        std::cout << "dwa_image.image_format: " << dwa_image.image_format << std::endl;
        std::cout << "dwa_image.data_type: " << dwa_image.data_type << std::endl;

        filename = "./dwa_image.bmp";
        bm_image_write_to_bmp(dwa_image, filename.c_str());
        std::cout << "dwa_image已保存到: " << filename << std::endl;
    }
    // 返回处理后的图像
}

