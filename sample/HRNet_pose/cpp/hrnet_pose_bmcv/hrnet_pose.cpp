//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream> 

#include "json.hpp"
#include "hrnet_pose.hpp"
#include "opencv2/opencv.hpp"
#include "../dependencies/include/yolov5.hpp"

#define DUMP_FILE 0
using namespace std;

void get_log_json(vector<cv::Mat> heapMaps, string name){

    using json = nlohmann::json;
    json data;
    for (size_t i = 0; i < heapMaps.size(); ++i) {
        const cv::Mat& mat = heapMaps[i];
        
        if (!mat.data) {
            std::cerr << "Error: Mat at index " << i << " is not valid." << std::endl;
            continue;
        }
        
        json mat_data;
        for (int row = 0; row < mat.rows; ++row) {
            json row_data;
            for (int col = 0; col < mat.cols; ++col) {
                float value = mat.at<float>(row, col); 
                row_data.push_back(value);
            }
            
            mat_data.push_back(row_data);
        }

        data["Mat_" + std::to_string(i)] = mat_data;
    }

    std::ofstream file(name);

    file << std::setw(4) << data << std::endl;
    file.close();

    std::cout << "Data has been saved to :" << name << std::endl;

}


// SKELETON
const vector<vector<int>> SKELETON = {
    {1, 3}, {1, 0}, {2, 4}, {2, 0}, {0, 5}, {0, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

// CocoColors
const vector<vector<int>> CocoColors = {
    {255, 0, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {170, 255, 0}, {85, 255, 0}, {0, 255, 0},
    {0, 255, 85}, {0, 255, 170}, {0, 255, 255}, {0, 170, 255}, {0, 85, 255}, {0, 0, 255}, {85, 0, 255},
    {170, 0, 255}, {255, 0, 255}, {255, 0, 170}, {255, 0, 85}
};

// FLIP_PAIRS
const vector<vector<int>> FLIP_PAIRS = {
    {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}
};


HRNetPose::HRNetPose(shared_ptr<BMNNContext> context) : m_bmContext(context) {
   cout << "HRNetPose Constructor" << endl;
}

HRNetPose::~HRNetPose() {
    cout << "HRNetPose Destructor" << endl;

    bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
    
    for(int i = 0; i < max_batch; i++){
        bm_image_destroy(m_resized_imgs[i]);
        bm_image_destroy(m_converto_imgs[i]);
    }

}

int HRNetPose::Init(bool flip, const string& coco_names_file) {

    int ret = 0;
    
    ifstream ifs(coco_names_file);
    if (ifs.is_open()) {
        std::string line;
        while(getline(ifs, line)) {
            line = line.substr(0, line.length() - 1);
            m_class_names.push_back(line);
        }
    }
    
    m_flip = flip;
    m_bmNetwork = m_bmContext->network(0);
    max_batch = m_bmNetwork->maxBatch();  
    auto inputTensor = m_bmNetwork->inputTensor(0); 
    m_net_h = inputTensor->get_shape()->dims[2]; 
    m_net_w = inputTensor->get_shape()->dims[3]; 

    m_resized_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);

    int aligned_net_w = FFALIGN(m_net_w, 64);  
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w}; 
    for(int i = 0; i < max_batch; i++){
        ret= bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
        assert(BM_SUCCESS == ret);
    }
    ret = bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());

    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32; 
    if (inputTensor->get_dtype() == BM_INT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED; 
    }
    ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_converto_imgs.data(), max_batch);
    assert(BM_SUCCESS == ret);

    linear_trans_param_.alpha_0 = scale_[0] / 255.0;
    linear_trans_param_.alpha_1 = scale_[1] / 255.0;
    linear_trans_param_.alpha_2 = scale_[2] / 255.0;
    linear_trans_param_.beta_0 = (0.0 - mean_[0]) * (scale_[0]);
    linear_trans_param_.beta_1 = (0.0 - mean_[1]) * (scale_[1]);
    linear_trans_param_.beta_2 = (0.0 - mean_[2]) * (scale_[2]);

    return 0;
}

void HRNetPose::enableProfile(TimeStamp *ts) {
    m_ts = ts;
}

int HRNetPose::get_batch_size() {
    return max_batch;
}

vector<vector<YoloV5Box>> HRNetPose::get_person_detection_boxes(vector<vector<YoloV5Box>>& yolov5_boxes, float person_thresh) { 
  
  vector<vector<YoloV5Box>> person_boxes;
  auto person_index = find(m_class_names.begin(),m_class_names.end(), "person") - m_class_names.begin();

  for (const vector<YoloV5Box>& boxes_per_image : yolov5_boxes){
    
    vector<YoloV5Box> person_boxes_per_image = {};

    for (const YoloV5Box& box : boxes_per_image){
      
      float score = box.score;
      int class_id = box.class_id;

      if ((score >= person_thresh) & (class_id == person_index)){

        person_boxes_per_image.push_back(box);
      }

    }

    person_boxes.push_back(person_boxes_per_image);

  }

  return person_boxes;

}


// Adjust bounding box to fit a fixed aspect ratio
YoloV5Box adjust_box(YoloV5Box& box, const cv::Size& fixed_size) {
    
    float xmin = box.x;
    float ymin = box.y;
    float xmax = box.x + box.width;
    float ymax = box.y + box.height;
    float hw_ratio = static_cast<float>(fixed_size.height) / fixed_size.width;

    if (box.height / box.width > hw_ratio) {
        // Need padding in width direction
        float wi = box.height / hw_ratio;
        float pad_w = (wi - box.width) / 2;
        xmin -= pad_w;
        xmax += pad_w;
    } else {
        // Need padding in height direction
        float hi = box.width * hw_ratio;
        float pad_h = (hi - box.height) / 2;
        ymin -= pad_h;
        ymax += pad_h;
    }

    box.x = xmin;
    box.y = ymin;
    box.width = xmax - xmin;
    box.height = ymax - ymin;

    return box;
}

// Get affine transform matrix 
cv::Mat get_affine_transform(YoloV5Box& box, const cv::Size& fixed_size, bool inv = true) {
    
    YoloV5Box adjustedBox = adjust_box(box, fixed_size);

    // Get the box after adjust
    float src_xmin = adjustedBox.x;
    float src_ymin = adjustedBox.y;
    float src_xmax =adjustedBox.x +adjustedBox.width;
    float src_ymax = adjustedBox.y + adjustedBox.height;
    float src_h = adjustedBox.height;
    float src_w = adjustedBox.width;

    cv::Point2f src_center((src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2);
    cv::Point2f src_p2(src_center.x, src_center.y - src_h / 2);
    cv::Point2f src_p3(src_center.x + src_w / 2, src_center.y);

    cv::Point2f dst_center(static_cast<float>(fixed_size.width - 1) / 2.0f, static_cast<float>(fixed_size.height - 1) / 2.0f);
    cv::Point2f dst_p2(static_cast<float>(fixed_size.width - 1) / 2.0f, 0);
    cv::Point2f dst_p3(fixed_size.width - 1, static_cast<float>(fixed_size.height - 1) / 2.0f);

    cv::Point2f src[3] = {src_center, src_p2, src_p3};
    cv::Point2f dst[3] = {dst_center, dst_p2, dst_p3}; 
    
    cv::Mat trans;
    if (inv) {
        trans = getAffineTransform(src, dst);
    } else {
        for (int i = 0; i < 3; i++){
            dst[i].x /= 4.0f;
            dst[i].y /= 4.0f;
        }
        trans = getAffineTransform(dst, src);
    }

    return trans;
}

int HRNetPose::pre_process(const bm_image& image, YoloV5Box& box) {
    
    int ret = 0;
    shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);

    bm_image src;
    ret = bm_image_create(m_bmContext->handle(), image.height, image.width, FORMAT_RGB_PLANAR, image.data_type, &src);
    ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, image, &src);    //RGB

    cv::Mat trans = get_affine_transform(box, cv::Size(m_net_w, m_net_h));

    cv::Mat mat_src;
    cv::bmcv::toMAT(&src, mat_src);
    
    cv::Mat mat_dst;
    cv::Size dst_size(m_net_w, m_net_h);
    warpAffine(mat_src, mat_dst, trans, dst_size, cv::INTER_LINEAR);

    // string fname = cv::format("affine_img_opencv.jpg");
    // cv::imwrite(fname, mat_dst);

    bm_image dst;
    ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, src.image_format, src.data_type, &dst);
    ret = cv::bmcv::toBMI(mat_dst, &dst, true);
    bm_image_destroy(src);

    bm_image image_aligned;
    bool need_copy = dst.width & (64 - 1);
    
    if (need_copy){
        int stride1[3], stride2[3];

        ret = bm_image_get_stride(dst, stride1);
        stride2[0] = FFALIGN(stride1[0], 64);
        stride2[1] = FFALIGN(stride1[1], 64);
        stride2[2] = FFALIGN(stride1[2], 64);

        ret = bm_image_create(m_bmContext->handle(), dst.height, dst.width, dst.image_format, dst.data_type, &image_aligned, stride2);
        ret = bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN); 

        bmcv_copy_to_atrr_t copyToAttr;
        memset(&copyToAttr, 0, sizeof(copyToAttr));

        copyToAttr.start_x = 0;
        copyToAttr.start_y = 0;
        copyToAttr.if_padding = 1;
        
        ret = bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, dst, image_aligned);
    }
    else{
        image_aligned = dst;
    }

    ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, image_aligned, m_resized_imgs.data());

#if DUMP_FILE
    cv::Mat cv_image_aligned;
    cv::bmcv::toMAT(m_resized_imgs.data(), cv_image_aligned);
    string fname = cv::format("resized_img.jpg");
    cv::imwrite(fname, cv_image_aligned);
#endif

    ret = bm_image_destroy(dst);
    if(need_copy) bm_image_destroy(image_aligned);

    ret = bmcv_image_convert_to(m_bmContext->handle(), 1, linear_trans_param_, m_resized_imgs.data(), m_converto_imgs.data());
    CV_Assert(ret == 0);

    bm_device_mem_t imput_dev_mem;
    ret = bm_image_get_contiguous_device_mem(1, m_converto_imgs.data(), &imput_dev_mem);
    input_tensor->set_device_mem(&imput_dev_mem);
    input_tensor->set_shape_by_dim(0, 1); 
    
    return ret;
}


// Function to flip images along the width axis
cv::Mat flip_image(const cv::Mat& image) {
   
    cv::Mat flip_image;
    cv::flip(image, flip_image, 1);
    
    return flip_image;
}

//Function to flip the output back according to the matched parts
void flip_back(vector<cv::Mat>& output_flipped, const vector<vector<int>>& matched_parts){

    for (cv::Mat& mat : output_flipped){
        cv::flip(mat, mat, 1);
    }

    for (const auto& pair : matched_parts){
        cv::Mat tmp = output_flipped[pair[0]];
        output_flipped[pair[0]] = output_flipped[pair[1]];
        output_flipped[pair[1]] = tmp;
    }
}

// Function to shift the output
void shift_output(vector<cv::Mat>& flippedBackMat){

    for (cv::Mat& mat : flippedBackMat){
      
      int width = mat.cols;
      for(int col = width - 1; col >= 1; col--){

        cv::Mat beforeMatCol = mat(cv::Rect(col - 1, 0, 1, mat.rows));
        cv::Mat matCol = mat(cv::Rect(col, 0, 1, mat.rows));
        beforeMatCol.copyTo(matCol);

      }
  }
}

vector<cv::Mat> clone_output(vector<cv::Mat>& heatMaps){

    vector<cv::Mat> newMat;
    for (cv::Mat& mat : heatMaps){
      cv::Mat newmat = mat.clone();
      newMat.emplace_back(newmat);
    }

    return newMat;
}

vector<cv::Mat> add_mat(vector<cv::Mat>& outputMat, vector<cv::Mat>& finalFlippedMat){
    
    if (outputMat.size() != finalFlippedMat.size()){
        cout << "Vector sizes do not match." << endl;
        exit(1);    
    }

    vector<cv::Mat> result;
    for (int i = 0; i < outputMat.size(); i++){
        
        if (outputMat[i].size() != finalFlippedMat[i].size()){
            cout << "Matrices sizes do not match at index " << i << "." << endl;
            exit(1);
        }

        cv::Mat res(outputMat[i].size(), outputMat[i].type());
        cv::addWeighted(outputMat[i], 0.5, finalFlippedMat[i], 0.5, 0.0, res);
        result.emplace_back(res);
    }

    return result;
}

void get_output_mat(shared_ptr<BMNNTensor>& outputTensor, vector<cv::Mat>& outputMat){
    
    auto output_shape = outputTensor->get_shape();
    auto output_dims = output_shape->dims;
    int batch_size = output_shape->dims[0];
    int keypoints_num = output_shape->dims[1];
    int heatmap_h = output_shape->dims[2];
    int heatmap_w = output_shape->dims[3];

    // Return an array pointer to system memory of tensor.
    float* predict = (float*)outputTensor->get_cpu_data();

    // batch_size = 1
    for (int i = 0; i < batch_size; i++){
        
        for (int j = 0; j < keypoints_num; j++){
            
            float* start_ptr = predict + i * keypoints_num * heatmap_h * heatmap_w + j * heatmap_h * heatmap_w;
            cv::Mat single_heatmap(heatmap_h, heatmap_w, CV_32FC1, start_ptr);
            outputMat.emplace_back(single_heatmap);
        }
    }
}

void get_max_preds(vector<cv::Mat>& heatmaps, vector<cv::Point2f>& preds, vector<float>& maxvals) {
    
    int num_joints = heatmaps.size();
    int h = heatmaps[0].rows;
    int w = heatmaps[0].cols;

    maxvals.resize(num_joints);
    preds.resize(num_joints);

    for (int i = 0; i < num_joints; ++i) {
        
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(heatmaps[i], &minVal, &maxVal, &minLoc, &maxLoc);

        float x = static_cast<float>(maxLoc.x);
        float y = static_cast<float>(maxLoc.y);

        if (maxVal > 0.0f) {
            preds[i] = cv::Point2f(x, y);
            maxvals[i] = static_cast<float>(maxVal);
        } else {
            preds[i] = cv::Point2f(-1, -1); 
            maxvals[i] = 0.0f;
        }
    }
}

cv::Point2f affine_points(cv::Point2f& pred, const cv::Mat& trans){
    
    cv::Mat point(3, 1, CV_32F);
    point.at<float>(0) = pred.x;
    point.at<float>(1) = pred.y;
    point.at<float>(2) = 1.0f;

    cv::Mat transformed_point;
    cv::Mat trans_32f;
    trans.convertTo(trans_32f, CV_32F);

    cv::gemm(trans_32f, point, 1.0, cv::noArray(), 0.0, transformed_point);

    return cv::Point2f(transformed_point.at<float>(0,0), transformed_point.at<float>(1,0));

}

void HRNetPose::transform_preds(vector<cv::Point2f>& preds, YoloV5Box& box, vector<cv::Point2f>& keypoints){
    
    keypoints.resize(preds.size());

    cv::Mat trans = get_affine_transform(box, cv::Size(m_net_w, m_net_h), false);

    for (int i = 0; i < preds.size(); i++){
        keypoints[i] = affine_points(preds[i], trans);
    }
    
}

int HRNetPose::post_process(vector<cv::Mat>& heatMaps, YoloV5Box& box, vector<cv::Point2f>& keypoints, vector<float>& maxvals)
{
    vector<cv::Point2f> preds;
    get_max_preds(heatMaps, preds, maxvals);

    int heatmap_height = heatMaps[0].rows;
    int heatmap_width = heatMaps[0].cols;

    for (int i = 0; i < heatMaps.size(); i++){
        cv::Point2f& pred = preds[i];

        int px = static_cast<int>(floor(pred.x + 0.5));
        int py = static_cast<int>(std::floor(pred.y + 0.5));

        if (1 < px && px < heatmap_width - 1 && 1 < py && py < heatmap_height - 1) {

            float dx = heatMaps[i].at<float>(py, px + 1) - heatMaps[i].at<float>(py, px - 1);
            float dy = heatMaps[i].at<float>(py + 1, px) - heatMaps[i].at<float>(py - 1, px);

            float offset_x = std::copysign(0.25f, dx);
            float offset_y = std::copysign(0.25f, dy);

            pred.x += offset_x;
            pred.y += offset_y;
        }
    }

    transform_preds(preds, box, keypoints);
    
    return 0;
}

void HRNetPose::drawPose(vector<cv::Point2f> keypoints, cv::Mat& image){

    for (int i = 0; i < SKELETON.size(); i++){
        
        int kpt_a = SKELETON[i][0];
        int kpt_b = SKELETON[i][1];
	
        cv::Point center_a;
        center_a.x = static_cast<int>(keypoints[kpt_a].x);
        center_a.y = static_cast<int>(keypoints[kpt_a].y);
        
        cv::Point center_b;
        center_b.x = static_cast<int>(keypoints[kpt_b].x);
        center_b.y = static_cast<int>(keypoints[kpt_b].y);

        cv::Scalar color(CocoColors[i][0], CocoColors[i][1], CocoColors[i][2]);
        cv::circle(image, center_a, 6, color, -1);
        cv::circle(image, center_a, 6, color, -1);

        cv::line(image, center_a, center_b, color, 2);
    }
}

int HRNetPose::poseEstimate(const bm_image& image, YoloV5Box& box, vector<cv::Point2f>& keypoints, vector<float>& maxvals, vector<cv::Mat>& heatMaps) {
    
    int ret = 0;
    m_ts->save("hrnet preprocess", 1);
    ret = pre_process(image, box);
    CV_Assert(ret == 0);
    m_ts->save("hrnet preprocess", 1);

    m_ts->save("hrnet inference", 1);
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    m_ts->save("hrnet inference", 1);

    m_ts->save("hrnet postprocess", 1);
    shared_ptr<BMNNTensor> outputTensor = m_bmNetwork->outputTensor(0);
    get_output_mat(outputTensor, heatMaps);
    m_ts->save("hrnet postprocess", 1);

    shared_ptr<BMNNTensor> outputTensorFlip;
    vector<cv::Mat> heatMapsFlip;
    if (m_flip){
        
        m_ts->save("hrnet postprocess", 1);
        heatMaps = clone_output(heatMaps);
        m_ts->save("hrnet postprocess", 1);
        shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
        
        m_ts->save("hrnet preprocess", 1);
        cv::Mat cv_mat_image;
        bm_image bm_image_to_mat = m_resized_imgs[0];
        ret = cv::bmcv::toMAT(&bm_image_to_mat, cv_mat_image); 

        cv::Mat flipped_image = flip_image(cv_mat_image);
        bm_image flipped_bm_image;
        ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, m_resized_imgs[0].image_format, m_resized_imgs[0].data_type, &flipped_bm_image);
        ret = cv::bmcv::toBMI(flipped_image, &flipped_bm_image, true); 

        bm_image flipped_convert_bm_image;
        ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, m_converto_imgs[0].image_format, m_converto_imgs[0].data_type, &flipped_convert_bm_image);
        ret = bmcv_image_convert_to(m_bmContext->handle(), 1, linear_trans_param_, &flipped_bm_image, &flipped_convert_bm_image);

        bm_device_mem_t input_dev_mem_;
        ret = bm_image_get_contiguous_device_mem(1, &flipped_convert_bm_image, &input_dev_mem_);
        input_tensor->set_device_mem(&input_dev_mem_);
        input_tensor->set_shape_by_dim(0, 1);
        m_ts->save("hrnet preprocess", 1);

        m_ts->save("hrnet inference", 1);
        ret = m_bmNetwork->forward();
        CV_Assert(ret == 0);
        m_ts->save("hrnet inference", 1);

        ret = bm_image_destroy(flipped_bm_image);
        ret = bm_image_destroy(flipped_convert_bm_image);

        m_ts->save("hrnet postprocess", 1);
        outputTensorFlip = m_bmNetwork->outputTensor(0);
        get_output_mat(outputTensorFlip, heatMapsFlip);
        flip_back(heatMapsFlip, FLIP_PAIRS);
        shift_output(heatMapsFlip);
        heatMaps = add_mat(heatMaps, heatMapsFlip);
        m_ts->save("hrnet postprocess", 1);

    }

    m_ts->save("hrnet postprocess", 1);
    ret = post_process(heatMaps, box, keypoints, maxvals);
    CV_Assert(ret == 0);
    m_ts->save("hrnet postprocess", 1);

    return ret;
}


