//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <string>
#include <vector>
#include "centernet.hpp"
#include <queue>
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0
#define OUTPUT_CHECK 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},    {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {170, 255, 0},  {85, 255, 0},    {0, 255, 0},     {0, 255, 85},
    {0, 255, 170},  {0, 255, 255},   {0, 170, 255},   {0, 85, 255},
    {0, 0, 255},    {85, 0, 255},    {170, 0, 255},   {255, 0, 255},
    {255, 0, 170},  {255, 0, 85},    {255, 0, 0},     {255, 0, 255},
    {255, 85, 255}, {255, 170, 255}, {255, 255, 255}, {170, 255, 255},
    {85, 255, 255}};

Centernet::Centernet(std::shared_ptr<BMNNContext> context)
    : m_bmContext(context) {
    std::cout << "Centernet ctor .." << std::endl;
}

Centernet::~Centernet() {
    std::cout << "Centernet dtor ..." << std::endl;
    bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
    for (int i = 0; i < max_batch; i++) {
        bm_image_destroy(m_converto_imgs[i]);
        bm_image_destroy(m_resized_imgs[i]);
    }
}

int Centernet::Init(float confThresh, const std::string& coco_names_file) {
    m_confThreshold = confThresh;

    std::ifstream ifs(coco_names_file);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            line = line.substr(0, line.length() - 1);
            m_class_names.push_back(line);
        }
    }

    // 1. get network
    m_bmNetwork = m_bmContext->network(0);

    // 2. get input
    max_batch = m_bmNetwork->maxBatch();
    auto tensor = m_bmNetwork->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];

    // 3. get output
    output_num = m_bmNetwork->outputTensorNum();
    assert(output_num == 1);

    // 4. initialize bmimages
    m_resized_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);
    // some API only accept bm_image whose stride is aligned to 64
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for (int i = 0; i < max_batch; i++) {
        auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w,
                                   FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                   &m_resized_imgs[i], strides);
        assert(BM_SUCCESS == ret);
    }
    bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (tensor->get_dtype() == BM_INT8) {
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w,
                                     FORMAT_RGB_PLANAR, img_dtype,
                                     m_converto_imgs.data(), max_batch);
    assert(BM_SUCCESS == ret);

    // 5.converto
    float input_scale = tensor->get_scale();
    const std::vector<float> mean = {0.40789655, 0.44719303, 0.47026116};
    const std::vector<float> std = {0.2886383, 0.27408165, 0.27809834};

    converto_attr.alpha_0 = input_scale / (std[0] * 255);
    converto_attr.beta_0 = -mean[0] * input_scale / std[0];
    converto_attr.alpha_1 = input_scale / (std[1] * 255);
    converto_attr.beta_1 = -mean[1] * input_scale / std[1];
    converto_attr.alpha_2 = input_scale / (std[2] * 255);
    converto_attr.beta_2 = -mean[2] * input_scale / std[2];

    // converto_attr.alpha_0 = input_scale * 0.01358f ; 
    // converto_attr.beta_0 = input_scale * -1.4131f ; 
    // converto_attr.alpha_1 = input_scale * 0.0143f ; 
    // converto_attr.beta_1 = input_scale * -1.6316f ; 
    // converto_attr.alpha_2 = input_scale * 0.0141f ; 
    // converto_attr.beta_2 = input_scale * -1.69103f ; 

    return 0;
}

void Centernet::enableProfile(TimeStamp* ts) {
    m_ts = ts;
}

int Centernet::batch_size() {
    return max_batch;
};

int Centernet::Detect(const std::vector<bm_image>& input_images,
                      std::vector<CenternetBoxVec>& boxes) {
    int ret = 0;
    // 3. preprocess
    m_ts->save("Centernet preprocess", input_images.size());
    ret = pre_process(input_images);
    CV_Assert(ret == 0);
    m_ts->save("Centernet preprocess", input_images.size());

    // 4. forward
    m_ts->save("Centernet inference", input_images.size());
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    m_ts->save("Centernet inference", input_images.size());

    // 5. post process
    m_ts->save("Centernet postprocess", input_images.size());
    ret = post_process(input_images, boxes);
    CV_Assert(ret == 0);
    m_ts->save("Centernet postprocess", input_images.size());
    return ret;
}

int Centernet::pre_process(const std::vector<bm_image>& images) {
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
    int image_n = images.size();
    // 1. resize image
    int ret = 0;
    for (int i = 0; i < image_n; ++i) {
        bm_image image1 = images[i];
        bm_image image_aligned;
        bool need_copy = image1.width & (64 - 1);
        if (need_copy) {
            int stride1[3], stride2[3];
            bm_image_get_stride(image1, stride1);
            stride2[0] = FFALIGN(stride1[0], 64);
            stride2[1] = FFALIGN(stride1[1], 64);
            stride2[2] = FFALIGN(stride1[2], 64);
            bm_image_create(m_bmContext->handle(), image1.height, image1.width,
                            image1.image_format, image1.data_type,
                            &image_aligned, stride2);

            bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
            bmcv_copy_to_atrr_t copyToAttr;
            memset(&copyToAttr, 0, sizeof(copyToAttr));
            copyToAttr.start_x = 0;
            copyToAttr.start_y = 0;
            copyToAttr.if_padding = 1;
            bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1,
                               image_aligned);
        } else {
            image_aligned = image1;
        }
#if USE_ASPECT_RATIO
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(images[i].width, images[i].height,
                                              m_net_w, m_net_h, &isAlignWidth);
        bmcv_padding_atrr_t padding_attr;
        memset(&padding_attr, 0, sizeof(padding_attr));
        padding_attr.dst_crop_sty = 0;
        padding_attr.dst_crop_stx = 0;
        padding_attr.padding_b = 0;
        padding_attr.padding_g = 0;
        padding_attr.padding_r = 0;
        padding_attr.if_memset = 1;
        if (isAlignWidth) {
            padding_attr.dst_crop_h = images[i].height * ratio;
            padding_attr.dst_crop_w = m_net_w;

            int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
            padding_attr.dst_crop_sty = ty1;
            padding_attr.dst_crop_stx = 0;
        } else {
            padding_attr.dst_crop_h = m_net_h;
            padding_attr.dst_crop_w = images[i].width * ratio;

            int tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
            padding_attr.dst_crop_sty = 0;
            padding_attr.dst_crop_stx = tx1;
        }

        bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
        auto ret = bmcv_image_vpp_convert_padding(
            m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
            &padding_attr, &crop_rect);
#else
        auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, images[i],
                                          &m_resized_imgs[i]);
#endif
        assert(BM_SUCCESS == ret);

#if DUMP_FILE
        cv::Mat resized_img;
        cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
        // std::string fname = cv::format("resized_img_%d.jpg", i);
        // cv::imwrite(fname, resized_img);
        uchar* array;
        array = resized_img.data;
        std::fstream f;
        f.open("bmnn_decode.txt",std::ios::out);
        for (int i = 0; i < resized_img.rows; i++){
            for (int j = 0; j < resized_img.cols; j++){
                f<< (int)*(array+j+i*resized_img.cols)<<"   ";
            }
            f << std::endl<<std::endl;
        }
        f.close();
        
        
#endif
        // bm_image_destroy(image1);
        if (need_copy)
            bm_image_destroy(image_aligned);
    }

    // 2. converto
    ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr,
                                m_resized_imgs.data(), m_converto_imgs.data());
    CV_Assert(ret == 0);
#if DUMP_FILE
        
        // cv::Mat after_convert;
        // cv::bmcv::toMAT(&m_converto_imgs[0], after_convert);
        // std::string fname = cv::format("after_convert.jpg");
        // cv::imwrite(fname, after_convert);
#endif

    // 3. attach to tensor
    if (image_n != max_batch)
        image_n = m_bmNetwork->get_nearest_batch(image_n);
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(),
                                       &input_dev_mem);
    input_tensor->set_device_mem(&input_dev_mem);
    input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
    return 0;
}

float Centernet::get_aspect_scaled_ratio(int src_w,
                                         int src_h,
                                         int dst_w,
                                         int dst_h,
                                         bool* pIsAligWidth) {
    float ratio;
    float r_w = (float)dst_w / src_w;
    float r_h = (float)dst_h / src_h;
    if (r_h > r_w) {
        *pIsAligWidth = true;
        ratio = r_w;
    } else {
        *pIsAligWidth = false;
        ratio = r_h;
    }
    return ratio;
}

int Centernet::post_process(const std::vector<bm_image>& images,
                            std::vector<CenternetBoxVec>& detected_boxes) {
    CenternetBoxVec ctbox_vec;
    std::shared_ptr<BMNNTensor> output_tensor = m_bmNetwork->outputTensor(0);
    m_feat_c = output_tensor->get_shape()->dims[1];
    m_feat_h = output_tensor->get_shape()->dims[2];
    m_feat_w = output_tensor->get_shape()->dims[3];
    m_area = m_feat_h*m_feat_w;
    m_confidence_mask_ptr = std::make_unique<int[]>(m_area);
    m_confidence_ptr      = std::make_unique<float[]>(m_area);
    float* pred_hms[m_class_num];    // heatmap
    float* pred_whs[m_hw_channels];    // height and width
    float* pred_off[m_offset_channels];    // offset
    
    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
        ctbox_vec.clear();
        auto& frame = images[batch_idx];
        int frame_width = frame.width;
        int frame_height = frame.height;

        int tx1 = 0, ty1 = 0;
#if USE_ASPECT_RATIO
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(frame.width, frame.height,
                                              m_net_w, m_net_h, &isAlignWidth);
        if (isAlignWidth) {
            ty1 = (int)((m_net_h - (int)(frame_height * ratio)) / 2);
        } else {
            tx1 = (int)((m_net_w - (int)(frame_width * ratio)) / 2);
        }
#endif
        

        // data from one batch
        float *tensor_data = (float*)output_tensor->get_cpu_data() + batch_idx*m_feat_c*m_area;
            
        for (int i = 0; i < m_feat_c; ++i){
            // every pic
            float *ptr = tensor_data + i*m_area;
            // heatmap
            if (i < m_class_num){
                pred_hms[i] = ptr;
            }
            // hw
            else if (i < m_class_num+m_hw_channels){
                pred_whs[i-m_class_num] = ptr;
            }
            // offset
            else{
                pred_off[i - m_class_num-m_hw_channels] = ptr;
            }
        }


        // sigmoid activation in heatmap
        for (int i = 0; i < m_class_num; ++i) {
            for (int j = 0; j < m_area; ++j) {
                pred_hms[i][j] = 1.0 / (1 + expf(-pred_hms[i][j]));
            }
        }

        // heatmap max pool
        for (size_t i = 0; i < m_class_num; ++i) {
            SimplyMaxpool2D(pred_hms[i]);
        }

        // find max cls and index in each pixel location, result should be 128*128
        std::pair<int*, float*> pairs = MaxOfLocation(pred_hms);
        float* arr = pairs.second;
        int* cls = pairs.first;
        // get topk index in 1d
        int k = 75;
        int* index_topk = topk(arr, k);

        // mask
        std::vector<float> xv, yv, wv, hv;

        for (int i = 0; i < k; ++i) {
            int idx = index_topk[i];
            // offset x y
            float x = (idx % m_feat_w + pred_off[0][idx]) / m_feat_w * m_net_w;
            float y = (idx / m_feat_w + pred_off[1][idx]) / m_feat_h * m_net_h;
            float w = pred_whs[0][idx] / m_feat_w * m_net_w;
            float h = pred_whs[1][idx] / m_feat_h * m_net_h;
            
            xv.push_back(x);
            yv.push_back(y);
            // half width and height
            wv.push_back(w);
            hv.push_back(h);
            
        }

        assert(xv.size() == yv.size());
        m_detected_count = k;

    
        // m_detected_objs = std::make_unique<float[]>(m_detected_count * 6); // x1,y1,x2,y2,conf,cls
        for (int j = 0; j < m_detected_count; ++j) {
            int i = index_topk[j];
            float centerX = (xv[j]+1-tx1)/ratio -1;
            float centerY = (yv[j]+1-ty1)/ratio -1;
            float width = (wv[j]+0.5) / ratio;
            float height = (hv[j]+0.5) / ratio;

            CenternetBox box;
            box.x = int(centerX - width / 2);
            if (box.x < 0) box.x = 0;
            box.y = int(centerY - height / 2);
            if (box.y < 0) box.y = 0;
            box.width = width;
            box.height = height;
            box.class_id = cls[i];
            box.score = arr[i];
            ctbox_vec.push_back(box);    
        }

        detected_boxes.push_back(ctbox_vec);      
    }
    return 0;
};

void Centernet::SimplyMaxpool2D(float *data) {
    static std::vector<std::pair<int, int>> poolOffset = {
        {-1, -1}, {0,  -1}, {1,  -1},
        {-1,  0}, {0,   0}, {1,   0},
        {-1,  1}, {0,   1}, {1,   1}
    };

    std::unique_ptr<float[]> pMask = std::make_unique<float[]>(m_area);

    // h*w = 512 * 512
    for (int p = 0; p < m_feat_h; p++) {
        for (int q = 0; q < m_feat_w; q++) {
            float max_hm = 0.f;
            for (auto offset : poolOffset) {
                int target_x = q + offset.first;
                int target_y = p + offset.second;
                if (target_x >= 0 && target_x < m_feat_w &&
                    target_y >= 0 && target_y < m_feat_h) {
                    max_hm = std::max(max_hm, data[m_feat_w * target_y + target_x]);
                }
            }
            pMask[m_feat_w* p + q] = max_hm;
        }
    }

    for (int i = 0; i < m_feat_w*m_feat_h; ++i) {
        if (pMask[i] - data[i] > std::numeric_limits<float>::epsilon()) {
            data[i] = 0.f;
        }
    }
}

// ---------------------add v--------------------------
std::pair<int*, float*> Centernet::MaxOfLocation(float ** heatmap){
    float *max_arr = new float[128*128];
    int *cls_arr = new int[128*128];
    for (int j = 0; j < 128*128; j++){
        float max = -1.0;
        int cls = -1;
        for (int i = 0; i < 80; i++){
            cls = heatmap[i][j] > max ? i : cls;
            max = heatmap[i][j] > max ? heatmap[i][j] : max;  
        }
        max_arr[j] = max;
        cls_arr[j] = cls;
    }
    std::pair<int*,float*> res = std::make_pair(cls_arr,max_arr);
    return res;
}

// bool cmp(const int & p, const int & q){
//     return Centernet::arrr[p] > Centernet::arrr[q];
// }

int* Centernet::topk(float * a, int k){
    // float *a = new float[k]();
    auto cmp = [&](const int & m, const int & n){return a[m] > a[n];};
    std::priority_queue<int, std::vector<int>, decltype(cmp)> q(cmp);
    int Len = 128*128;
    for(int i = 0;i<Len;++i)
    {
        if(q.size() < k)    q.push(i);
        else
        {
            int idx = q.top();
            if(a[idx] > a[i])   continue;
            else
            {
                q.pop();
                q.push(i);
            }
        }
    }
    int* res = new int[k];
    for(int i = 0;i<k;++i)
    {
        res[i] = q.top();
        q.pop();
    }
    return res;

}

// -----------------------add ^-----------------------

// float Centernet::FindMaxConfidenceObject(float score[], int count, int& idx) {
//     float max_score = -1.0f;
//     int   max_idx   = -1;
//     for (int i = 0; i < count; ++i) {
//         if (score[i] - m_confThreshold > std::numeric_limits<float>::epsilon() && 
//             score[i] - max_score  > std::numeric_limits<float>::epsilon()) {
//             max_score = score[i];
//             max_idx   = i;
//         }
//     }
//     idx = max_idx;
//     return max_score;
// }

void Centernet::draw_bmcv(bm_handle_t &handle, int classId, float conf, int left, int top, int width, int height, bm_image& frame, bool put_text_flag, float conf_threshold)   // Draw the predicted bounding box
{
    if (conf >= conf_threshold){
         int colors_num = colors.size();
        //Draw a rectangle displaying the bounding box
        bmcv_rect_t rect;
        rect.start_x = left;
        rect.start_y = top;
        rect.crop_w = width;
        rect.crop_h = height;
        // std::cout << rect.start_x << "," << rect.start_y << "," << rect.crop_w << "," << rect.crop_h << std::endl;

        int thickness = 3;
        if(width < thickness * 2 || height < thickness * 2){
            std::cout << "width or height too small, this rect will not be drawed: " << 
                "[" << left << ", "<< top << ", " << width << ", " << height << "]" << std::endl;
        } else{
            bmcv_image_draw_rectangle(handle, frame, 1, &rect, thickness, colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]);
        }   
        // cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3);

        if (put_text_flag){
            //Get the label for the class name and its confidence
            std::string label = m_class_names[classId] + ":" + cv::format("%.2f", conf);
            // Display the label at the top of the bounding box
            // int baseLine;
            // cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            // top = std::max(top, labelSize.height);
            // //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
            // cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
            bmcv_point_t org = {left, top};
            bmcv_color_t color = {colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]};
            int thickness = 2;
            float fontScale = 2; 
            if (BM_SUCCESS != bmcv_image_put_text(handle, frame, label.c_str(), org, color, fontScale, thickness)) {
                std::cout << "bmcv put text error !!!" << std::endl;   
            }
        }
    }
 
  
}