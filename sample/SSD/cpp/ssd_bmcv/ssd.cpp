//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "ssd.hpp"
extern bool IS_DIR;
extern bool CONFIDENCE;
extern bool NMS;
float overlap_FM(float x1, float w1, float x2, float w2)
{
	float l1 = x1;
	float l2 = x2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1;
	float r2 = x2 + w2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float box_intersection_FM(SSDObjRect a, SSDObjRect b)
{
	float w = overlap_FM(a.x1, a.x2 - a.x1, b.x1, b.x2 - b.x1);
	float h = overlap_FM(a.y1, a.y2 - a.y1, b.y2, b.y2 - b.y1);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

float box_union_FM(SSDObjRect a, SSDObjRect b)
{
	float i = box_intersection_FM(a, b);
	float u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - i;
	return u;
}

float box_iou_FM(SSDObjRect a, SSDObjRect b)
{
	return box_intersection_FM(a, b) / box_union_FM(a, b);
}

static bool sort_ObjRect(SSDObjRect a, SSDObjRect b)
{
    return a.score > b.score;
}

static void nms_sorted_bboxes(const std::vector<SSDObjRect>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = objects.size();

    for (int i = 0; i < n; i++)    {
        const SSDObjRect& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const SSDObjRect& b = objects[picked[j]];

            float iou = box_iou_FM(a, b);
            if (iou > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

//public:
SSD::SSD(std::shared_ptr<BMNNContext> context, int dev_id, float conf_thre, float nms_thre):
        m_bmContext(context),m_dev_id(dev_id),m_conf_thre(conf_thre),m_nms_thre(nms_thre){
    std::cout << "SSD create bm_context" << std::endl;
}

SSD::~SSD(){
    std::cout << "SSD free." << std::endl;   
    bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
    for(int i = 0; i < max_batch; i++){
        bm_image_destroy(m_converto_imgs[i]);
        bm_image_destroy(m_resized_imgs[i]);
    }
}

void SSD::Init(){
    //1. Get network.
    m_bmNetwork = m_bmContext->network(0);

    //2. Get input.
    max_batch = m_bmNetwork->maxBatch();
    auto tensor = m_bmNetwork->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];
    m_resized_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);
    //3. Align stride, some API only accept bm_image which's stride is aligned to 64
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for(int i=0; i<max_batch; i++){
        auto ret= bm_image_create(m_bmContext->handle(), m_net_h, m_net_w,
            FORMAT_RGB_PLANAR,
            DATA_TYPE_EXT_1N_BYTE,
            &m_resized_imgs[i], strides);
        assert(BM_SUCCESS == ret);
    }
    bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());

    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (tensor->get_dtype() == BM_INT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }

    //4. Create input data.
    auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w,
                                    FORMAT_RGB_PLANAR,
                                    img_dtype,
                                    m_converto_imgs.data(), max_batch);
    assert(BM_SUCCESS == ret);
}

int SSD::batch_size(){
    return max_batch;
}

int SSD::detect(const std::vector<cv::Mat> &input_images, const std::vector<std::string> &input_names,
                  std::vector<std::vector<SSDObjRect> > &results, cv::VideoWriter *VideoWriter){
    int ret = 0;
    //1. Preprocess, convert raw images to format which fits SSD network.
    m_ts->save("SSD preprocess");
    ret = pre_process(input_images);
    m_ts->save("SSD preprocess");

    CV_Assert(ret == 0);
    //2. Run SSD inference.
    m_ts->save("SSD inference");
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    m_ts->save("SSD inference");

    //3. Postprocess, get detected boxes and draw them on original images.
    m_ts->save("SSD postprocess");
    ret = post_process(input_images, input_names, results, VideoWriter);
    CV_Assert(ret == 0);
    m_ts->save("SSD postprocess");
    return ret;
}

void SSD::enableProfile(TimeStamp *ts){
    m_ts = ts;
}

//private:
float SSD::get_aspect_scaled_ratio(int src_w, int src_h, 
                                   int dst_w, int dst_h, bool *pIsAligWidth){
    float ratio;
    float r_w = (float)dst_w / src_w;
    float r_h = (float)dst_h / src_h;
    if (r_h > r_w){
        *pIsAligWidth = true;
        ratio = r_w;
    }
    else{
        *pIsAligWidth = false;
        ratio = r_h;
    }
    return ratio;
}

int SSD::pre_process(const std::vector<cv::Mat> &images){
    //Safety check.
    assert(images.size() <= max_batch);
    
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
    assert(images.size() <= input_tensor->get_shape()->dims[0]);

    //1. Resize input images.
    int ret = 0;
    for(int i = 0; i < images.size(); i++){
        bm_image image1;
        bm_image image_aligned;
        CV_Assert(0 == cv::bmcv::toBMI((cv::Mat&)images[i], &image1, true));
        bool need_copy = image1.width & (64 - 1);
        if(need_copy){
            int stride1[3], stride2[3];
            bm_image_get_stride(image1, stride1);
            stride2[0] = FFALIGN(stride1[0], 64);
            stride2[1] = FFALIGN(stride1[1], 64);
            stride2[2] = FFALIGN(stride1[2], 64);
            bm_image_create(m_bmContext->handle(), image1.height, image1.width,
                image1.image_format, image1.data_type, &image_aligned, stride2);
            bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
            bmcv_copy_to_atrr_t copyToAttr;
            memset(&copyToAttr, 0, sizeof(copyToAttr));
            copyToAttr.start_x = 0;
            copyToAttr.start_y = 0;
            copyToAttr.if_padding = 1;

            bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1, image_aligned);
        }else {
            image_aligned = image1;
        }
        #if USE_ASPECT_RATIO
            bool isAlignWidth = false;
            float ratio = get_aspect_scaled_ratio(images[i].cols, images[i].rows, m_net_w, m_net_h, &isAlignWidth);
            bmcv_padding_atrr_t padding_attr;
            memset(&padding_attr, 0, sizeof(padding_attr));
            padding_attr.dst_crop_sty = 0;
            padding_attr.dst_crop_stx = 0;
            padding_attr.padding_b = 114;
            padding_attr.padding_g = 114;
            padding_attr.padding_r = 114;
            padding_attr.if_memset = 1;
            if (isAlignWidth) {
            padding_attr.dst_crop_h = images[i].rows*ratio;
            padding_attr.dst_crop_w = m_net_w;

            int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
            padding_attr.dst_crop_sty = ty1;
            padding_attr.dst_crop_stx = 0;
            }else{
            padding_attr.dst_crop_h = m_net_h;
            padding_attr.dst_crop_w = images[i].cols*ratio;

            int tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
            padding_attr.dst_crop_sty = 0;
            padding_attr.dst_crop_stx = tx1;
            }

            bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
            auto ret = bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
                &padding_attr, &crop_rect);
        #else
            auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i], nullptr, BMCV_INTER_LINEAR);
        #endif

        assert(BM_SUCCESS == ret);
        bm_image_destroy(image1);
        if(need_copy) bm_image_destroy(image_aligned);
    }
    //2. Convert data scale.
    float input_scale = input_tensor->get_scale();
    bmcv_convert_to_attr converto_attr;
    converto_attr.alpha_0 = input_scale;
    converto_attr.beta_0 = -104.0 * input_scale; //B
    converto_attr.alpha_1 = input_scale;
    converto_attr.beta_1 = -117.0 * input_scale; //G
    converto_attr.alpha_2 = input_scale;
    converto_attr.beta_2 = -123.0 * input_scale; //R
    ret = bmcv_image_convert_to(m_bmContext->handle(), images.size(), converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
    CV_Assert(ret == 0);

    //3. Attach to input tensor.
    int image_n = images.size();
    if(images.size() != max_batch) image_n = m_bmNetwork->get_nearest_batch(image_n); 
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
    input_tensor->set_device_mem(&input_dev_mem);
    input_tensor->set_shape_by_dim(0, image_n); 
    return 0;
}

int SSD::post_process(const std::vector<cv::Mat> &images, const std::vector<std::string> &input_names,
                            std::vector<std::vector<SSDObjRect> > &results, cv::VideoWriter *VideoWriter){
    std::shared_ptr<BMNNTensor> outputTensor = m_bmNetwork->outputTensor(0);
    auto output_shape = outputTensor->get_shape();
    auto output_dims = output_shape->num_dims; //[1 1 200 7]
    assert(m_bmNetwork->outputTensorNum() == 1);// [1 1 200/800 7]
#if DEBUG
    std::cout<<"Output shape infos: "<<output_shape->dims[0]<<" "
             <<output_shape->dims[1]<<" "<<output_shape->dims[2]<<" "
             <<output_shape->dims[3]<<std::endl;
#endif
    assert(output_dims == 4 &&
           output_shape->dims[0] == 1 &&
           output_shape->dims[1] == 1 &&
           output_shape->dims[3] == 7 );
    auto output_data = outputTensor->get_cpu_data();
    //1. Get output bounding boxes.
    int box_num_raw = output_shape->dims[0] * 
                      output_shape->dims[1] * 
                      output_shape->dims[2] ;
    for(int bid = 0; bid < box_num_raw; bid++){ //bid: box id
        SSDObjRect temp_bbox;
        temp_bbox.class_id = *(output_data + 7 * bid + 1);
        temp_bbox.score = *(output_data + 7 * bid + 2);
        int i = *(output_data + 7 * bid);
        if(i >= results.size())
            continue;
        if(CONFIDENCE || !IS_DIR){
            if(temp_bbox.score < m_conf_thre || i >= images.size()){
                continue;
            }        
        }
        temp_bbox.x1 = *(output_data + 7 * bid + 3) * m_net_w;
        temp_bbox.y1 = *(output_data + 7 * bid + 4) * m_net_h;
        temp_bbox.x2 = *(output_data + 7 * bid + 5) * m_net_w;
        temp_bbox.y2 = *(output_data + 7 * bid + 6) * m_net_h;
        results[i].push_back(temp_bbox);
    }
    for(int i = 0; i < results.size(); i++){        
        #if USE_ASPECT_RATIO
            int tx1 = 0, ty1 = 0;
            bool isAlignWidth = false;
            float ratio = get_aspect_scaled_ratio(images[i].cols, images[i].rows,
                                                m_net_w, m_net_h, &isAlignWidth);
            if(isAlignWidth){
                ty1 = (int)((m_net_h - (int)(images[i].rows * ratio)) / 2);
            }else{
                tx1 = (int)((m_net_w - (int)(images[i].cols * ratio)) / 2);
            }
            for(int j = 0; j < results[i].size(); j++){
                //if tx1, align height
                results[i][j].x1 -= tx1;
                results[i][j].x2 -= tx1;
                //if ty1, align width
                results[i][j].y1 -= ty1;
                results[i][j].y2 -= ty1;
                results[i][j].x1 /= ratio;
                results[i][j].x2 /= ratio;
                results[i][j].y1 /= ratio;
                results[i][j].y2 /= ratio;
            }
        #else
            #if DEBUG
                std::cout << "width: " << images[i].cols 
                          << " height: " << images[i].rows << std::endl;
            #endif
            for(int j = 0; j < results[i].size(); j++){
                results[i][j].x1 *= (float)images[i].cols / (float)m_net_w;
                results[i][j].x2 *= (float)images[i].cols / (float)m_net_w;
                results[i][j].y1 *= (float)images[i].rows / (float)m_net_h;
                results[i][j].y2 *= (float)images[i].rows / (float)m_net_h;
            }
        #endif
    }
    //2. NMS.
    if(NMS || !IS_DIR){
        for(int i = 0; i < results.size(); i++){
            std::vector<int> class_ids;
            std::vector<SSDObjRect> bboxes_per_class;
            std::vector<SSDObjRect> nmsed_results;
            for(int j = 0; j < results[i].size(); j++){
                if(std::find(class_ids.begin(), class_ids.end(), results[i][j].class_id) 
                    == class_ids.end()){
                    class_ids.push_back(results[i][j].class_id);
                }
            }
            for(int j = 0; j < class_ids.size(); j++){
                bboxes_per_class.clear();
                for(int k = 0; k < results[i].size(); k++){
                    if(results[i][k].class_id == class_ids[j])
                        bboxes_per_class.push_back(results[i][k]);
                }
                std::sort(bboxes_per_class.begin(), bboxes_per_class.end(), sort_ObjRect);
                std::vector<int> picked;
                nms_sorted_bboxes(bboxes_per_class, picked, m_nms_thre);
                for(int k = 0; k < picked.size(); k++){
                    nmsed_results.push_back(bboxes_per_class[picked[k]]);
                }
            }
            results[i].clear();
            results[i] = nmsed_results;
        }
    }
    

    for(int i = 0; i < results.size(); i++){
        auto &frame_ = images[i];
        cv::Mat frame(m_net_h, m_net_w, CV_8UC3, cv::SophonDevice(m_dev_id));
        cv::cvtColor(frame_, frame, cv::COLOR_RGB2BGR);
        int frame_width = frame.cols;
        int frame_height = frame.rows;

        for(int j = 0; j < results[i].size(); j++){
            SSDObjRect rect = results[i][j];
            cv::rectangle(frame, cv::Rect(rect.x1, rect.y1,
                                          rect.x2 - rect.x1, rect.y2 - rect.y1),
                                          cv::Scalar(255, 0, 0), 2);
        }
        if(IS_DIR){
            //save result images.
            if(access("results", 0) != F_OK)
                mkdir("results", S_IRWXU);
            if(m_bmNetwork->inputTensor(0)->get_dtype() == BM_FLOAT32){
                if(batch_size() == 1){
                    if(access("results/fp32-b1", 0) != F_OK)
                        mkdir("results/fp32-b1", S_IRWXU);
                    cv::imwrite("results/fp32-b1/" + input_names[i], frame);
                }else{
                    if(access("results/fp32-b4", 0) != F_OK)
                        mkdir("results/fp32-b4", S_IRWXU);
                    cv::imwrite("results/fp32-b4/" + input_names[i], frame);
                }
            }else{
                if(batch_size() == 1){
                    if(access("results/int8-b1", 0) != F_OK)
                        mkdir("results/int8-b1", S_IRWXU);
                    cv::imwrite("results/int8-b1/" + input_names[i], frame);
                }else{
                    if(access("results/int8-b4", 0) != F_OK)
                        mkdir("results/int8-b4", S_IRWXU);
                    cv::imwrite("results/int8-b4/" + input_names[i], frame);
                }
            }
        }else{
            //save video.
            (*VideoWriter) << frame;
        }

    }
    return 0;
}