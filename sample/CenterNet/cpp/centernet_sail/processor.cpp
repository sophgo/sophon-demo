//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <limits>
#include <numeric>
#include <queue>
#include <vector>
#include "processor.h"
// #include "glog/logging.h"
 
CenterNetPreprocessor::CenterNetPreprocessor(sail::Bmcv& bmcv, int result_width, int result_height, float scale)
  : bmcv_{bmcv},
    resize_w_{result_width},
    resize_h_{result_width},
    ab_{scale * 0.01358f, scale * -1.4131f, scale * 0.0143f, scale * -1.6316f, scale * 0.0141f, scale * -1.69103f} {}
 
 
void CenterNetPreprocessor::Process(sail::BMImage& input, sail::BMImage& output, bool& align_width, float& ratio) {
    // bgr normalization
    sail::BMImage tmp;
    float resize_ratio;
    int target_w, target_h;
 
    if (input.width() > input.height()) {
        resize_ratio = (float)resize_w_ / input.width();
        target_w     = resize_w_;
        target_h     = int(input.height() * resize_ratio);
        align_width  = true;
        ratio        = (float)target_h / target_w;
    } else {
        resize_ratio = (float)resize_h_ / input.height();
        target_w     = int(input.width() * resize_ratio);
        target_h     = resize_h_;
        align_width  = false;
        ratio        = (float)target_w / target_h;
    }
 
    sail::PaddingAtrr pad = sail::PaddingAtrr();
    int offset_x = target_w >= target_h ? 0 : ((resize_w_  - target_w) / 2);
    int offset_y = target_w <= target_h ? 0 : ((resize_h_  - target_h) / 2);
    pad.set_stx(offset_x);
    pad.set_sty(offset_y);
    pad.set_w(target_w);
    pad.set_h(target_h);
    pad.set_r(0);
    pad.set_g(0);
    pad.set_b(0);
    tmp = bmcv_.crop_and_resize_padding(input, 0, 0, 
                                        input.width(), input.height(),
                                        resize_w_, resize_h_,
                                        pad);
    //bmcv_.vpp_resize(input, tmp, resize_w_, resize_h_);
    // linear: bgr-planer -> bgr-planar
    bmcv_.convert_to(tmp, output,
                     std::make_tuple(std::make_pair(ab_[0], ab_[1]),
                                     std::make_pair(ab_[2], ab_[3]),
                                     std::make_pair(ab_[4], ab_[5])));
    
}
 
void CenterNetPreprocessor::BGRNormalization(float* data) {
    constexpr static float kBGRMeans[] = {0.40789655, 0.44719303, 0.47026116};
    constexpr static float kBGRStd[]   = {0.2886383,  0.27408165, 0.27809834};
 
    // input is BGR format
    for (int i = 0; i < 3; ++i) {
        int planer_size = resize_w_ * resize_h_;
        for (int j = 0; j < planer_size; ++j) {
            data[i * planer_size + j] = 
                (data[i * planer_size + j] / 255. - kBGRMeans[i]) / kBGRStd[i];
        }
    }
}
 
 
CenterNetPostprocessor::CenterNetPostprocessor(std::vector<int>& output_shape, float threshold, float scale)
  : output_shape_{output_shape},
    output_size_{0, 0, 0, 0},
    threshold_{threshold},
    scale_{scale} {
    // centernet output dims 1*84*128*128
    assert(output_shape_.size() == 4);
    // LOG(INFO) << "Got output shape " << output_shape_[0] << ", "
    //                                  << output_shape_[1] << ", "
    //                                  << output_shape_[2] << ", "
    //                                  << output_shape_[3];
    std::cout << "Got output shape " << output_shape_[0] << ", "
                                     << output_shape_[1] << ", "
                                     << output_shape_[2] << ", "
                                     << output_shape_[3];
    output_size_[0] = output_shape[1] * output_shape[2] * output_shape[3];
    output_size_[1] = output_shape[2] * output_shape[3];
    output_size_[2] = output_shape[3];
    output_size_[3] = 1;
 
    confidence_mask_ptr_ = std::make_unique<int[]>(output_size_[1]);
    confidence_ptr_      = std::make_unique<float[]>(output_size_[1]);
 
  }
 
CenterNetPostprocessor::~CenterNetPostprocessor() {
    confidence_mask_ptr_.reset();
    confidence_ptr_.reset();
    detected_objs_.reset();
}
 
void CenterNetPostprocessor::Process(float* output_data, bool align_width, float ratio) {
    // Once one batch
    if (scale_ - 1 >= std::numeric_limits<float>::epsilon()) {
        int elem_count = std::accumulate(output_shape_.begin() + 1, output_shape_.end(), 1, std::multiplies<int>());
        for (int i = 0; i < elem_count; ++i) {
            output_data[i] *= scale_;
        }
    }
    for (int i = 0; i < output_shape_[1]; ++i) {
//        for (int j = 0; j < output_shape_[0]; ++j) {
        int offset = i * output_size_[1];
        float* ptr = output_data + offset;
        if (i < output_shape_[1] - 4) {
            // heatmap
            pred_hms_[i] = ptr;
        } else if (i < output_shape_[1] - 2) {
            // width&height
            pred_whs_[i - kHeatMapChannels] = ptr;
        } else if (i < output_shape_[1]) {
            // offset
            pred_off_[i - kHeatMapChannels - kHeightWChannels] = ptr;
        }
//        }
    }
 
 
    // sigmoid activation in heatmap
    for (int i = 0; i < kHeatMapChannels; ++i) {
        for (int j = 0; j < output_size_[1]; ++j) {
            pred_hms_[i][j] = 1.0 / (1 + expf(-pred_hms_[i][j]));
        }
    }
    // heatmap max pool
    for (size_t i = 0; i < kHeatMapChannels; ++i) {
        SimplyMaxpool2D(pred_hms_[i]);
    }
    
    // find max cls and index in each pixel location, result should be 128*128
    std::pair<int*, float*> pairs = MaxOfLocation(pred_hms_);
    float* arr = pairs.second;
    int* cls = pairs.first;
    // get topk index in 1d
    int k = 75;
    int* index_topk = topk(arr, k);

    // mask
    std::vector<float> xv,         yv,
                       half_w,     half_h;

 
    // // get maxium object confidence
    // for (int i = 0; i < output_size_[1]; ++i) {
    //     int validChannelCount = validChannels.size();
    //     float score[validChannelCount];
    //     for (int j = 0; j < validChannelCount; ++j) {
    //         score[j] = pred_hms_[validChannels[j]][i];
    //     }
    //     int channelIndex = -1;
    //     confidence_ptr_[i] = FindMaxConfidenceObject(score, validChannelCount, channelIndex);
    //     confidence_mask_ptr_[i] = channelIndex >= 0 ? validChannels[channelIndex] : -1;
    //     if (confidence_mask_ptr_[i] >= 0) {
    //         class_conf.push_back(confidence_ptr_[i]);
    //         class_cls.push_back(confidence_mask_ptr_[i]);
    //     }
    // }

    for (int j = 0; j < k; ++j) {
        int i = index_topk[j];
        // offset x y
        float x = (i % output_shape_[3]) + pred_off_[0][i];
        float y = (i / output_shape_[3]) + pred_off_[1][i];
        float w = pred_whs_[0][i] / 2.0;
        float h = pred_whs_[1][i] / 2.0;
        
        xv.push_back(x);
        yv.push_back(y);
        // half width and height
        half_w.push_back(w);
        half_h.push_back(h);
        
    }
    assert(xv.size() == yv.size());
    detected_count_ = xv.size();
    detected_objs_ = std::make_unique<float[]>(detected_count_ * 6); // x1,y1,x2,y2,conf,cls
    for (int i = 0; i < detected_count_; ++i) {
        
        float x1 = (xv[i] - half_w[i]) / output_shape_[3];
        float y1 = (yv[i] - half_h[i]) / output_shape_[2];
        float x2 = (xv[i] + half_w[i]) / output_shape_[3];
        float y2 = (yv[i] + half_h[i]) / output_shape_[2];
        // letterbox
        if (align_width) {
            y1 = (y1 - (1 - ratio) / 2) / ratio;
            y2 = (y2 - (1 - ratio) / 2) / ratio;
        } else {
            x1 = (x1 - (1 - ratio) / 2) / ratio;
            x2 = (x2 - (1 - ratio) / 2) / ratio;
        }
        // bbox
        detected_objs_[i * 6 + 0] = x1;
        detected_objs_[i * 6 + 1] = y1;
        detected_objs_[i * 6 + 2] = x2;
        detected_objs_[i * 6 + 3] = y2;

        int idx = index_topk[i];
        // class
        detected_objs_[i * 6 + 4] = cls[idx];
        // confidence
        detected_objs_[i * 6 + 5] = arr[idx];
    }
}
 
void CenterNetPostprocessor::SimplyMaxpool2D(float* data) {
    static std::vector<std::pair<int, int>> poolOffset = {
        {-1, -1}, {0,  -1}, {1,  -1},
        {-1,  0}, {0,   0}, {1,   0},
        {-1,  1}, {0,   1}, {1,   1}
    };
 
    std::unique_ptr<float[]> pMask = std::make_unique<float[]>(output_size_[1]);
 
    // h*w = 512 * 512
    for (int p = 0; p < output_shape_[2]; p++) {
        for (int q = 0; q < output_shape_[3]; q++) {
            float max_hm = 0.f;
            for (auto offset : poolOffset) {
                int target_x = q + offset.first;
                int target_y = p + offset.second;
                if (target_x >= 0 && target_x < output_shape_[3] &&
                    target_y >= 0 && target_y < output_shape_[2]) {
                    max_hm = std::max(max_hm, data[output_size_[2] * target_y + target_x]);
                }
            }
            pMask[output_size_[2] * p + q] = max_hm;
        }
    }
 
    for (int i = 0; i < output_size_[1]; ++i) {
        if (pMask[i] - data[i] > std::numeric_limits<float>::epsilon()) {
            data[i] = 0.f;
        }
    }
}

// ---------------------add v--------------------------
std::pair<int*, float*> CenterNetPostprocessor::MaxOfLocation(float ** heatmap){
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

int* CenterNetPostprocessor::topk(float * a, int k){
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
 
// float CenterNetPostprocessor::FindMaxConfidenceObject(float score[], int count, int& idx) {
//     float max_score = -1.0f;
//     int   max_idx   = -1;
//     for (int i = 0; i < count; ++i) {
//         if (score[i] - threshold_ > std::numeric_limits<float>::epsilon() && 
//             score[i] - max_score  > std::numeric_limits<float>::epsilon()) {
//             max_score = score[i];
//             max_idx   = i;
//         }
//     }
//     idx = max_idx;
//     return max_score;
// }
 
std::shared_ptr<std::vector<BMBBox>> CenterNetPostprocessor::CenternetCorrectBBox(int height, int width) {
    auto pVectorBBox = std::make_shared<std::vector<BMBBox>>();
    for (int i = 0; i < detected_count_; ++i) {
        // bbox
        pVectorBBox->emplace_back(
            detected_objs_[i * 6 + 0] * width,                                  // x
            detected_objs_[i * 6 + 1] * height,                                 // y
           (detected_objs_[i * 6 + 2] - detected_objs_[i * 6 + 0]) * width,     // w
           (detected_objs_[i * 6 + 3] - detected_objs_[i * 6 + 1]) * height,    // h
            detected_objs_[i * 6 + 4],                                          // class
            detected_objs_[i * 6 + 5]                                           // confidence
 
        );
    }
    return pVectorBBox;
}