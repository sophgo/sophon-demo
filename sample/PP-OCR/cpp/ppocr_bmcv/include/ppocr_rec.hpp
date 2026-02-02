#ifndef PPOCR_REC_HPP
#define PPOCR_REC_HPP

#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bmnn_utils.h"

#include "ppocr_det.hpp"

struct RecModelSize
{
    int w;
    int h;
};

class PPOCR_Rec
{   
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    TimeStamp* m_ts;
    public:
        PPOCR_Rec(std::shared_ptr<BMNNContext> context);
        virtual ~PPOCR_Rec();
        int Init(const std::string &label_path);
        int run(std::vector<bm_image> input_bmimgs, std::vector<OCRBoxVec> &boxes, std::vector<std::pair<int, int>> ids, bool beam_search, int beam_size);
        int preprocess_bmcv(const std::vector<bm_image> batch_bmimgs, int stage_w);
        int post_process(std::vector<std::pair<std::string, float>>& results, bool beam_search=false, int beam_width=3);

        template <class ForwardIterator>
        inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
            return std::distance(first, std::max_element(first, last));
        }
        static std::vector<std::string> ReadDict(const std::string &path);

        void enableProfile(TimeStamp *ts){
            m_ts = ts;
        };

    private:
        int image_n = 1;
        bool input_is_int8_;
        bool output_is_int8_;
        bm_image_data_format_ext data_type;
        
        std::vector<RecModelSize> img_size; //ppocr network stage shapes.
        std::vector<float> img_ratio; //ppocr network stage ratios, ratio = w / h
        int stage; //current stage.
        std::unordered_set<int> incomplete_stages; //store widths

        // bm image objects for storing intermediate results
        std::map<int, std::vector<bm_image>> resize_bmcv_map; //{width: vector<bm_image>}
        std::map<int, std::vector<bm_image>> linear_trans_bmcv_map; //{width: vector<bm_image>}

        int max_batch;
        int net_h_;
        int net_w_;
        int out_net_h_;
        int out_net_w_;

        // crop arguments of BMCV
        bmcv_rect_t crop_rect_;

        std::vector<std::string> label_list_;
        // linear transformation arguments of BMCV
        bmcv_convert_to_attr linear_trans_param_;
};



#endif //!PPOCR_REC_HPP