#include <string>
#include <fstream>
#include <iostream>

#include <bmlib_runtime.h>
#include <bmcv_api_ext.h>
#include <opencv2/opencv.hpp>

#define DEBUG 1

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
#define VPP_HEAP_ID 1
#else
#define VPP_HEAP_ID 2
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
    // 准备仿真的上游数据
    cv::Mat cv_image = cv::imread(input_path, cv::IMREAD_COLOR, dev_id);
    if (!cv_image.data){
        std::cout << "[ERROR]read failed" << std::endl;
        exit(1);
    }
    int input_width = cv_image.cols;
    int input_height = cv_image.rows;

    // handle初始化
    bm_handle_t handle;
    auto ret = bm_dev_request(&handle,dev_id);
    assert(ret == BM_SUCCESS);

    // 计算元素大小
    unsigned int image_size = input_width * input_height * cv_image.elemSize();
    bm_device_mem_t device_memory;

    // 预分配堆内存
    ret = bm_malloc_device_byte_heap_mask(handle,&device_memory,VPP_HEAP_ID,image_size);
    assert(ret == BM_SUCCESS);

    // mmap映射
    unsigned long long cpu_mem;
    ret = bm_mem_mmap_device_mem(handle,&device_memory,&cpu_mem);
    assert(ret == BM_SUCCESS);

    // 以上完成初始化操作
    // 模拟仿真FPGA或者网络传输，将数据传输到CPU指定内存位置
    memcpy(reinterpret_cast<void*>(cpu_mem),cv_image.data,image_size);

    // 完成数据传输，需要flush内存
    ret = bm_mem_flush_device_mem(handle,&device_memory);
    assert(ret == BM_SUCCESS);

    // 以上就完成从下位机传输裸BGR数据到微服务器的设备内存;
    // 使用方法1：将设备内存attach到bm_image，使用bmcv库的接口对数据进行处理
    // cv::Mat 读取的数据排列格式为FORMAT_BGR_PACKED,DATA_TYPE_EXT_1N_BYTE
    // 其他扩展格式请参考BMCV开发参考文档
    bm_image bmimg;
    ret = bm_image_create(handle,input_height,input_width,FORMAT_BGR_PACKED,DATA_TYPE_EXT_1N_BYTE,&bmimg);
    assert(ret == BM_SUCCESS);
    ret = bm_image_attach(bmimg,&device_memory);
    assert(ret == BM_SUCCESS);

    // 数据格式变换
    bm_image bmimg_rgb_planar;
    ret = bm_image_create(handle,input_height,input_width,FORMAT_RGB_PLANAR,DATA_TYPE_EXT_1N_BYTE,&bmimg_rgb_planar);
    assert(ret == BM_SUCCESS);
    ret = bmcv_image_storage_convert(handle,1,&bmimg,&bmimg_rgb_planar);
    assert(ret == BM_SUCCESS);

#if DEBUG
    bm_image_write_to_bmp(bmimg_rgb_planar,"./debug.bmp");
#endif

    // 使用方法2：直接作为网络的input_tensor,以下是伪代码用法
    // *m_inputTensors = device_memory
    // bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, m_inputTensors, m_netinfo->input_num,m_outputTensors, m_netinfo->output_num, user_mem, false);
    
    // 析构部分
    // 解除mmap关系
    ret = bm_mem_unmap_device_mem(handle,reinterpret_cast<void*>(cpu_mem),image_size);
    assert(ret == BM_SUCCESS);

    // detach 内存
    ret = bm_image_detach(bmimg);
    assert(ret == BM_SUCCESS);

    // 手动申请的，需要手动释放
    bm_free_device(handle,device_memory);

    // 释放bm_image
    assert(bm_image_destroy(bmimg) == BM_SUCCESS);
    assert(bm_image_destroy(bmimg_rgb_planar) == BM_SUCCESS);

    // 释放handle
    bm_dev_free(handle);

    std::cout << "[PASS]All done." << std::endl;

    return 0;
}
