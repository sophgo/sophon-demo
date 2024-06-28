#ifndef YOLOV5_H
#define YOLOV5_H

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

#include "opencv2/opencv.hpp"     // opencv头文件
#include "bmruntime_interface.h"  // bmruntime c接口头文件
#include "bm_wrapper.hpp"         // 一些bmcv接口的封装

struct YoloV5Box {
    float x, y, width, height;
    float score;
    int class_id;
};

using YoloV5BoxVec = std::vector<YoloV5Box>;

class YOLOv5 {
   public:
    YOLOv5(int dev_id,
           std::string bmodel_path,
           float conf_thresh,
           float nms_thresh,
           const std::string& coco_names_file);

    ~YOLOv5();

    // var
   private:
    // runtime
    bm_handle_t m_handle;
    void* m_bmrt;
    const char** m_net_names = NULL;
    const bm_net_info_t* m_net_info;
    bool can_mmap;

    int m_core_id = 0;
    int m_batch_size;
    int m_net_h, m_net_w;
    int m_input_num;
    float m_input_scale;
    bm_data_type_t m_input_dtype;
    bm_shape_t m_input_shape;
    bm_image_data_format_ext img_dtype;
    std::vector<bm_image> m_resized_bmimgs;
    bmcv_convert_to_attr m_converto_attr;

    std::vector<float> m_output_scales;
    std::vector<bm_shape_t> m_output_shapes;
    std::vector<bm_data_type_t> m_output_dtypes;
    int m_output_num;
    int m_class_num = 80;
    float m_conf_thresh;
    float m_nms_thresh;
    std::vector<std::string> m_class_names;

    // func
   public:
    void set_core_id(int core_id) { m_core_id = core_id; }
    int get_batch_size();
    int yolov5_detect(std::vector<cv::Mat>& dec_images, std::vector<YoloV5BoxVec>& boxes);
    void yolov5_draw(int classId,
                     float conf,
                     int left,
                     int top,
                     int right,
                     int bottom,
                     cv::Mat& frame);  // put it here?

   private:
    int yolov5_preprocess(std::vector<cv::Mat>& dec_images, std::vector<bm_tensor_t>& input_tensors);
    int yolov5_inference(std::vector<bm_tensor_t>& input_tensors, std::vector<bm_tensor_t>& output_tensors);
    int yolov5_postprocess(std::vector<cv::Mat>& dec_images,
                           std::vector<bm_tensor_t>& output_tensors,
                           std::vector<YoloV5BoxVec>& boxe);

    float* get_cpu_data(bm_tensor_t& tensor, int out_idx);
    int argmax(float* data, int dsize);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* alignWidth);
    static float sigmoid(float x);
    void NMS(YoloV5BoxVec& dets, float nmsConfidence);
};

#endif