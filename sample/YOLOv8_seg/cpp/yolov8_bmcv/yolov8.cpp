//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov8.hpp"

#include <fstream>
#include <string>
#include <vector>
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 0

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

YoloV8::YoloV8(std::shared_ptr<BMNNContext> context) : m_bmContext(context) {
    std::cout << "YoloV8 ctor .." << std::endl;
}

YoloV8::~YoloV8() {
    std::cout << "YoloV8 dtor ..." << std::endl;
    bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
    for (int i = 0; i < max_batch; i++) {
        bm_image_destroy(m_converto_imgs[i]);
        bm_image_destroy(m_resized_imgs[i]);
    }
    bm_free_device(m_tpu_mask_bmContext->handle(), m_mask_info_dev);
}

void YoloV8::tpumask_Init(std::shared_ptr<BMNNContext> tpu_mask_context) {
    tpu_post = true;
    m_tpu_mask_bmContext = tpu_mask_context;

    // get tpu_mask network
    m_tpu_mask_bmNetwork = m_tpu_mask_bmContext -> network(0);

    // get input
    auto mask_info  = m_tpu_mask_bmNetwork->inputTensor(0);
    auto output1 = m_tpu_mask_bmNetwork->inputTensor(1);
    m_tpumask_net_h = output1->get_shape()->dims[2];
    m_tpumask_net_w = output1->get_shape()->dims[3];
    m_mask_info_dev = *mask_info->get_device_mem();
    CV_Assert(BM_SUCCESS == bm_malloc_device_byte(m_tpu_mask_bmContext->handle(), &m_mask_info_dev, 4*bmrt_shape_count(mask_info->get_shape())));

    assert(output1->get_shape()->dims[1] == mask_info->get_shape()->dims[2]);
    mask_num = output1->get_shape()->dims[1];
    tpumask_max_batch =  output1->get_shape()->dims[0];

    auto outputTensor_type = m_tpu_mask_bmNetwork->outputTensor(0)->get_dtype();
    assert(outputTensor_type == BM_UINT8);

}

int YoloV8::Init(float confThresh, float nmsThresh, const std::string& coco_names_file) {
    m_confThreshold = confThresh;
    m_nmsThreshold = nmsThresh;

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
    assert(output_num > 0);
    min_dim = m_bmNetwork->outputTensor(0)->get_shape()->num_dims;

    // 4. initialize bmimages
    m_resized_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);

    // some API only accept bm_image whose stride is aligned to 64
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for (int i = 0; i < max_batch; i++) {
        auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                   &m_resized_imgs[i], strides);
        assert(BM_SUCCESS == ret);
    }
    bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (tensor->get_dtype() == BM_INT8) {
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype,
                                     m_converto_imgs.data(), max_batch);
    assert(BM_SUCCESS == ret);

    // 5.converto
    float input_scale = tensor->get_scale();
    input_scale = input_scale * 1.0 / 255.f;
    converto_attr.alpha_0 = input_scale;
    converto_attr.beta_0 = 0;
    converto_attr.alpha_1 = input_scale;
    converto_attr.beta_1 = 0;
    converto_attr.alpha_2 = input_scale;
    converto_attr.beta_2 = 0;

    return 0;
}

void YoloV8::enableProfile(TimeStamp* ts) {
    m_ts = ts;
}

int YoloV8::batch_size() {
    return max_batch;
};

int YoloV8::Detect(const std::vector<bm_image>& input_images, std::vector<YoloV8BoxVec>& boxes) {
    int ret = 0;
    // 3. preprocess
    m_ts->save("yolov8 preprocess", input_images.size());
    ret = pre_process(input_images);
    CV_Assert(ret == 0);
    m_ts->save("yolov8 preprocess", input_images.size());

    // 4. forward
    m_ts->save("yolov8 inference", input_images.size());
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    m_ts->save("yolov8 inference", input_images.size());

    // 5. post process
    m_ts->save("yolov8 postprocess", input_images.size());
    ret = post_process(input_images, boxes);
    CV_Assert(ret == 0);
    m_ts->save("yolov8 postprocess", input_images.size());
    return ret;
}

int YoloV8::pre_process(const std::vector<bm_image>& images) {
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
    int image_n = images.size();

    // 1. resize image letterbox
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
            bm_image_create(m_bmContext->handle(), image1.height, image1.width, image1.image_format, image1.data_type,
                            &image_aligned, stride2);

            bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
            bmcv_copy_to_atrr_t copyToAttr;
            memset(&copyToAttr, 0, sizeof(copyToAttr));
            copyToAttr.start_x = 0;
            copyToAttr.start_y = 0;
            copyToAttr.if_padding = 1;
            bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1, image_aligned);
        } else {
            image_aligned = image1;
        }
#if USE_ASPECT_RATIO
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(images[i].width, images[i].height, m_net_w, m_net_h, &isAlignWidth);
        bmcv_padding_atrr_t padding_attr;
        memset(&padding_attr, 0, sizeof(padding_attr));
        padding_attr.dst_crop_sty = 0;
        padding_attr.dst_crop_stx = 0;
        padding_attr.padding_b = 114;
        padding_attr.padding_g = 114;
        padding_attr.padding_r = 114;
        padding_attr.if_memset = 1;
        if (isAlignWidth) {
            padding_attr.dst_crop_h = images[i].height * ratio;
            padding_attr.dst_crop_w = m_net_w;

            int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);  // padding 大小
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
        auto ret = bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
                                                  &padding_attr, &crop_rect);
#else
        auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, images[i], &m_resized_imgs[i]);
#endif
        assert(BM_SUCCESS == ret);
        if (need_copy)
            bm_image_destroy(image_aligned);
    }

    // 2. converto img /= 255
    ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr, m_resized_imgs.data(),
                                m_converto_imgs.data());
    CV_Assert(ret == 0);

    // 3. attach to tensor
    if (image_n != max_batch)
        image_n = m_bmNetwork->get_nearest_batch(image_n);
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
    input_tensor->set_device_mem(&input_dev_mem);
    input_tensor->set_shape_by_dim(0, image_n);  // set real batch number

    return 0;
}

float YoloV8::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
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

void YoloV8::getmask_tpu(YoloV8BoxVec &yolobox_vec, std::shared_ptr<BMNNTensor> &out_tensor1,Paras& paras,YoloV8BoxVec &yolobox_vec_tmp) {
    //dynamic shape
    std::shared_ptr<BMNNTensor> mask_info = m_tpu_mask_bmNetwork->inputTensor(0);
    std::shared_ptr<BMNNTensor> output1 = m_tpu_mask_bmNetwork->inputTensor(1);
    
    const int mask_info_shape[3] = {1,yolobox_vec.size(),mask_num};
    mask_info->set_shape(mask_info_shape,3);

    //parepare bmodel inputs
    for (size_t i = 0; i < yolobox_vec.size(); i++)
    {
        CV_Assert(BM_SUCCESS == bm_memcpy_s2d_partial_offset(m_tpu_mask_bmContext->handle(), m_mask_info_dev, reinterpret_cast<void*>(yolobox_vec[i].mask.data()), 32*4, 32*4*i));
    }
    mask_info->set_device_mem(const_cast<bm_device_mem_t*>(&m_mask_info_dev));
    output1->set_device_mem(const_cast<bm_device_mem_t*>(out_tensor1->get_device_mem()));

    // run bmodel
    auto ret = m_tpu_mask_bmNetwork->forward();
    CV_Assert(ret == 0);

    //get outputs
    auto outputTensor = m_tpu_mask_bmNetwork->outputTensor(0);
    bm_device_mem_t outputTensor_dev= *outputTensor->get_device_mem();

    int num = yolobox_vec.size();
    // 转bm_image
    std::vector<bm_image> bmimg_vec(num);
    ret = bm_image_create_batch(m_tpu_mask_bmContext->handle(),m_tpumask_net_h, m_tpumask_net_w,FORMAT_GRAY,DATA_TYPE_EXT_1N_BYTE,bmimg_vec.data(),num);
    assert(ret == BM_SUCCESS);
    
    // attach
    ret = bm_image_attach_contiguous_mem(num,bmimg_vec.data(),outputTensor_dev);
    assert(ret == BM_SUCCESS);

    // create bmimage
    bm_image image_resize;
    bm_image_create(m_tpu_mask_bmContext->handle(), paras.height,paras.width,FORMAT_GRAY,DATA_TYPE_EXT_1N_BYTE,&image_resize);
    
    // resize config
    bmcv_rect_t rects{paras.r_x, paras.r_y, paras.r_w, paras.r_h};
    int crop_num = 1;
    bmcv_resize_t resize_img_attr{paras.r_x, paras.r_y,paras.r_w, paras.r_h,paras.width, paras.height};

    for (int i=0; i < bmimg_vec.size(); i++){
        // resize
        auto ret =  bmcv_image_vpp_convert(m_tpu_mask_bmContext->handle(), 1, bmimg_vec[i], &image_resize, &rects);
        assert(ret == BM_SUCCESS);

        // toMAT
        cv::Mat mask;
        ret = cv::bmcv::toMAT(&image_resize,mask,true);
        assert(ret == BM_SUCCESS);

        // crop + mask
        cv::Rect bound=cv::Rect{yolobox_vec[i].x1, yolobox_vec[i].y1, yolobox_vec[i].x2 - yolobox_vec[i].x1,yolobox_vec[i].y2 - yolobox_vec[i].y1};
        yolobox_vec[i].mask_img=mask(bound) > m_confThreshold * 255.0;
        yolobox_vec_tmp.push_back(yolobox_vec[i]);

    }
    
    bm_image_destroy(image_resize);
    // detach
    ret = bm_image_dettach_contiguous_mem(num,bmimg_vec.data());
    assert(ret == BM_SUCCESS);
    // bm_image_destroy_batch(bmimg_vec.data(),num); //Do not destroy these bm_image，just detach. Otherwise, you will fail in bm_free_device.
}


int YoloV8::post_process(const std::vector<bm_image>& images, std::vector<YoloV8BoxVec>& detected_boxes) {
    YoloV8BoxVec yolobox_vec;

    std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);

    for (int i = 0; i < output_num; i++) {
        outputTensors[i] = m_bmNetwork->outputTensor(i);
    }

    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
        yolobox_vec.clear();
        auto& frame = images[batch_idx];
        int frame_width = frame.width;
        int frame_height = frame.height;

        int min_idx = 0;
        int box_num = 0;

        // Single output
        auto out_tensor = outputTensors[min_idx];
        auto out_tensor1 = outputTensors[1];
        float* out1 = out_tensor1->get_cpu_data();
        const bm_shape_t* shape1 = out_tensor1->get_shape();
        int dims = 4;
        int sizes[] = {shape1->dims[0], shape1->dims[1], shape1->dims[2], shape1->dims[3]};
        cv::Mat output1(dims, sizes, CV_32F, out1 + batch_idx * shape1->dims[2] * shape1->dims[1] * shape1->dims[3]);
#if USE_ASPECT_RATIO

        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(frame_width, frame_height, m_net_w, m_net_h, &isAlignWidth);
        int tx1 = 0, ty1 = 0;
        if (isAlignWidth) {
            ty1 = (int)((m_net_h - frame_height * ratio) / 2);  // padding 大小
        } else {
            tx1 = (int)((m_net_w - frame_width * ratio) / 2);
        }
        ImageInfo para = {cv::Size(frame_width, frame_height), {ratio, ratio, tx1, ty1}};
#else
        ImageInfo para = {cv::Size(frame_width, frame_height),
                          {m_net_w / frame_width, m_net_h / frame_height, tx1, ty1}};
#endif
        m_class_num = out_tensor->get_shape()->dims[1] - mask_num - 4;
        int feat_num = out_tensor->get_shape()->dims[2];
        int nout = m_class_num + mask_num + 4;
        float* output_data = nullptr;
        std::vector<float> decoded_data;

        LOG_TS(m_ts, "post 1: get output");
        assert(box_num == 0 || box_num == out_tensor->get_shape()->dims[1]);
        box_num = out_tensor->get_shape()->dims[1];
        output_data = (float*)out_tensor->get_cpu_data() + batch_idx * feat_num * nout;
        LOG_TS(m_ts, "post 1: get output");

        // Candidates
        LOG_TS(m_ts, "post 2: get detections matrix nx6 (xyxy, conf, cls, mask)");
        float* cls_conf = output_data + 4 * feat_num;
        for (int i = 0; i < feat_num; i++) {
#if USE_MULTICLASS_NMS
            // multilabel
            for (int j = 0; j < m_class_num; j++) {
                float cur_value = cls_conf[i + j * feat_num];
                if (cur_value >= m_confThreshold) {
                    YoloV8Box box;
                    box.score = cur_value;
                    box.class_id = j;
                    int c = box.class_id * max_wh;
                    float centerX = output_data[i + 0 * feat_num];
                    float centerY = output_data[i + 1 * feat_num];
                    float width = output_data[i + 2 * feat_num];
                    float height = output_data[i + 3 * feat_num];

                    box.x1 = centerX - width / 2 + c;
                    box.y1 = centerY - height / 2 + c;
                    box.x2 = box.x1 + width;
                    box.y2 = box.y1 + height;
                    box.mask = vector<float>(output_data + 4 + m_class_num, output_data + nout);

                    yolobox_vec.push_back(box);
                }
            }
#else
            // best class
            float max_value = 0.0;
            int max_index = 0;
            for (int j = 0; j < m_class_num; j++) {
                float cur_value = cls_conf[i + j * feat_num];
                if (cur_value > max_value) {
                    max_value = cur_value;
                    max_index = j;
                }
            }

            if (max_value >= m_confThreshold) {
                YoloV8Box box;
                box.score = max_value;
                box.class_id = max_index;
                int c = box.class_id * max_wh;
                float centerX = output_data[i + 0 * feat_num];
                float centerY = output_data[i + 1 * feat_num];
                float width = output_data[i + 2 * feat_num];
                float height = output_data[i + 3 * feat_num];

                box.x1 = centerX - width / 2 + c;
                box.y1 = centerY - height / 2 + c;
                box.x2 = box.x1 + width;
                box.y2 = box.y1 + height;
                for (int k = 0; k < mask_num; k++) {
                    box.mask.push_back(output_data[i + (nout - mask_num + k) * feat_num]);
                }
                // box.mask=std::vector<float>(output_data+(nout-mask_num)*feat_num
                // + 4 + m_class_num, output_data + nout);
                yolobox_vec.push_back(box);
            }
#endif
        }
        LOG_TS(m_ts, "post 2: get detections matrix nx6 (xyxy, conf, cls, mask)");

        LOG_TS(m_ts, "post 3: nms");
        NMS(yolobox_vec, m_nmsThreshold);

        if (yolobox_vec.size() > max_det) {
            yolobox_vec.erase(yolobox_vec.begin(), yolobox_vec.begin() + (yolobox_vec.size() - max_det));
        }

        for (int i = 0; i < yolobox_vec.size(); i++) {
            int c = yolobox_vec[i].class_id * max_wh;
            yolobox_vec[i].x1 = yolobox_vec[i].x1 - c;
            yolobox_vec[i].y1 = yolobox_vec[i].y1 - c;
            yolobox_vec[i].x2 = yolobox_vec[i].x2 - c;
            yolobox_vec[i].y2 = yolobox_vec[i].y2 - c;
        }

        // float tx1 = 0, ty1 = 0;
        // bool isAlignWidth = false;
        // float ratio = get_aspect_scaled_ratio(frame.width, frame.height,
        // m_net_w, m_net_h, &isAlignWidth); if (isAlignWidth) {
        //     ty1 = (m_net_h - (float)(frame_height * ratio)) / 2;
        // } else {
        //     tx1 = (m_net_w - (float)(frame_width * ratio)) / 2;
        // }
        YoloV8BoxVec yolobox_vec_tmp;
        for (int i = 0; i < yolobox_vec.size(); i++) {
            float centerx = ((yolobox_vec[i].x2 + yolobox_vec[i].x1) / 2 - tx1) / ratio;
            float centery = ((yolobox_vec[i].y2 + yolobox_vec[i].y1) / 2 - ty1) / ratio;
            float width = (yolobox_vec[i].x2 - yolobox_vec[i].x1) / ratio;
            float height = (yolobox_vec[i].y2 - yolobox_vec[i].y1) / ratio;
            yolobox_vec[i].x1 = centerx - width / 2;
            yolobox_vec[i].y1 = centery - height / 2;
            yolobox_vec[i].x2 = centerx + width / 2;
            yolobox_vec[i].y2 = centery + height / 2;
        }
        clip_boxes(yolobox_vec, frame_width, frame_height);

        LOG_TS(m_ts, "post 3: nms");

        LOG_TS(m_ts, "post 4: get mask");
        if(tpu_post){
            cv::Vec4f trans = para.trans;
            int r_x = floor(trans[2] / 640 * 160);
            int r_y = floor(trans[3] / 640 * 160);
            int r_w = 160 - 2 * r_x;
            int r_h = 160 - 2 * r_y;
            r_w = MAX(r_w, 1);
            r_h = MAX(r_h, 1);
            struct Paras paras={r_x,r_y,r_w,r_h,para.raw_size.width,para.raw_size.height};
            getmask_tpu(yolobox_vec,outputTensors[1],paras,yolobox_vec_tmp);
         }else{
                for (int i = 0; i < yolobox_vec.size(); i++) {
                    // std::cout<<yolobox_vec[i].mask[0]<<std::endl;
                    if (yolobox_vec[i].x2 > yolobox_vec[i].x1 + 1 && yolobox_vec[i].y2 > yolobox_vec[i].y1 + 1) {
                        get_mask(cv::Mat(yolobox_vec[i].mask).t(), output1, para,
                                cv::Rect{yolobox_vec[i].x1, yolobox_vec[i].y1, yolobox_vec[i].x2 - yolobox_vec[i].x1,
                                        yolobox_vec[i].y2 - yolobox_vec[i].y1},
                                yolobox_vec[i].mask_img);
                        yolobox_vec_tmp.push_back(yolobox_vec[i]);
                    }
                }
        }
        detected_boxes.push_back(yolobox_vec_tmp);
        LOG_TS(m_ts, "post 4: get mask");
    }

    return 0;
}

void YoloV8::get_mask(const cv::Mat& mask_info,
                      const cv::Mat& mask_data,
                      const ImageInfo& para,
                      cv::Rect bound,
                      cv::Mat& mast_out) {
    cv::Vec4f trans = para.trans;
    int r_x = floor(trans[2] / 640 * 160);
    int r_y = floor(trans[3] / 640 * 160);
    int r_w = 160 - 2 * r_x;
    int r_h = 160 - 2 * r_y;
    r_w = MAX(r_w, 1);
    r_h = MAX(r_h, 1);

    std::vector<cv::Range> roi_rangs = {cv::Range(0, 1), cv::Range::all(), cv::Range(r_y, r_h + r_y),
                                        cv::Range(r_x, r_w + r_x)};
    cv::Mat temp_mask = mask_data(roi_rangs).clone();  // crop
    cv::Mat protos = temp_mask.reshape(0, {32, r_w * r_h});
    cv::Mat matmul_res = (mask_info * protos);
    cv::Mat masks_feature = matmul_res.reshape(1, {r_h, r_w});
    // cv::Mat dest;
    // exp(-masks_feature, dest);//sigmoid
    // dest = 1.0 / (1.0 + dest);
    int left = bound.x;         
    int top = bound.y;          
    int width = bound.width;    
    int height = bound.height;  
    cv::Mat mask;
    resize(masks_feature, mask, cv::Size(para.raw_size.width, para.raw_size.height));

    mast_out = mask(bound) > m_nmsThreshold;
}

void YoloV8::clip_boxes(YoloV8BoxVec& yolobox_vec, int src_w, int src_h) {
    for (int i = 0; i < yolobox_vec.size(); i++) {
        yolobox_vec[i].x1 = std::max((float)0.0, std::min(yolobox_vec[i].x1, (float)src_w));
        yolobox_vec[i].y1 = std::max((float)0.0, std::min(yolobox_vec[i].y1, (float)src_h));
        yolobox_vec[i].x2 = std::max((float)0.0, std::min(yolobox_vec[i].x2, (float)src_w));
        yolobox_vec[i].y2 = std::max((float)0.0, std::min(yolobox_vec[i].y2, (float)src_h));
    }
}

void YoloV8::xywh2xyxy(YoloV8BoxVec& xyxyboxes, std::vector<std::vector<float>> box) {
    for (int i = 0; i < box.size(); i++) {
        YoloV8Box tmpbox;
        tmpbox.x1 = box[i][0] - box[i][2] / 2;
        tmpbox.y1 = box[i][1] - box[i][3] / 2;
        tmpbox.x2 = box[i][0] + box[i][2] / 2;
        tmpbox.y2 = box[i][1] + box[i][3] / 2;
        xyxyboxes.push_back(tmpbox);
    }
}

void YoloV8::NMS(YoloV8BoxVec& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const YoloV8Box& a, const YoloV8Box& b) { return a.score < b.score; });

    std::vector<float> areas(length);
    for (int i = 0; i < length; i++) {
        float width = dets[i].x2 - dets[i].x1;
        float height = dets[i].y2 - dets[i].y1;
        areas[i] = width * height;
    }

    while (index > 0) {
        int i = 0;
        while (i < index) {
            float left = std::max(dets[index].x1, dets[i].x1);
            float top = std::max(dets[index].y1, dets[i].y1);
            float right = std::min(dets[index].x2, dets[i].x2);
            float bottom = std::min(dets[index].y2, dets[i].y2);
            float overlap = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
            if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence) {
                areas.erase(areas.begin() + i);
                dets.erase(dets.begin() + i);
                index--;
            } else {
                i++;
            }
        }
        index--;
    }
}

void YoloV8::draw_result(cv::Mat& img, YoloV8BoxVec& result) {
    cv::Mat mask = img.clone();
    for (int i = 0; i < result.size(); i++) {
        if(result[i].score < 0.25) continue;
        int left, top;
        left = result[i].x1;
        top = result[i].y1;
        int color_num = i;
        cv::Scalar color(colors[result[i].class_id % 25][0], colors[result[i].class_id % 25][1],
                         colors[result[i].class_id % 25][2]);
        cv::Rect bound = {result[i].x1, result[i].y1, result[i].x2 - result[i].x1, result[i].y2 - result[i].y1};

        rectangle(img, bound, color, 2);
        if (result[i].mask_img.rows && result[i].mask_img.cols > 0) {
            mask(bound).setTo(color, result[i].mask_img);
        }
        std::string label = std::string(m_class_names[result[i].class_id]) + std::to_string(result[i].score);
        putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    }
    addWeighted(img, 0.6, mask, 0.4, 0, img);  // add mask to src
}

void YoloV8::draw_bmcv(bm_handle_t& handle,
                       bm_image& frame,  // Draw the predicted bounding box
                       YoloV8BoxVec& result,
                       bool put_text_flag) {
    if (frame.image_format != FORMAT_RGB_PLANAR) {
        bm_image frame2;
        bm_image_create(handle, frame.height, frame.width, FORMAT_RGB_PLANAR, frame.data_type, &frame2);
        bmcv_image_storage_convert(handle, 1, &frame, &frame2);
        bm_image_destroy(frame);
        frame = frame2;
    }
    for (int i = 0; i < result.size(); i++) {
        if(result[i].score < 0.25) continue;
        int left, top, width, height;
        left = result[i].x1;
        top = result[i].y1;
        width = result[i].x2 - result[i].x1;
        height = result[i].y2 - result[i].y1;
        cv::Mat mask_img = result[i].mask_img;
        int color_num = i;
        cv::Scalar color(colors[result[i].class_id % 25][0], colors[result[i].class_id % 25][1],
                         colors[result[i].class_id % 25][2]);
        cv::Rect bound = {result[i].x1, result[i].y1, result[i].x2 - result[i].x1, result[i].y2 - result[i].y1};
        bm_image mask_bmimg;
        bm_image mask_bmimg_align;
        bm_image mask_bmimg_tmp;
        bm_image mask_bmimg_padding;
        auto ret =
            bm_image_create(handle, mask_img.rows, mask_img.cols, frame.image_format, frame.data_type, &mask_bmimg);
        ret =
            bm_image_create(handle, mask_img.rows, mask_img.cols, frame.image_format, frame.data_type, &mask_bmimg_tmp);
        int stride[8];
        stride[0] = FFALIGN(frame.width, 64);

        ret = bm_image_create(handle, frame.height, frame.width, frame.image_format, frame.data_type,
                              &mask_bmimg_padding, stride);
        cv::bmcv::toBMI(mask_img, &mask_bmimg_tmp, 1);
        bm_image_get_stride(mask_bmimg, stride);
        stride[0] = FFALIGN(stride[0], 64);
        ret = bm_image_create(handle, mask_img.rows, mask_img.cols, frame.image_format, frame.data_type,
                              &mask_bmimg_align, stride);
        bm_image_alloc_dev_mem_heap_mask(mask_bmimg_align, 6);
        bmcv_convert_to_attr_s bmcv_convert_to_attr = {color[0] / 255.0, 0, color[1] / 255.0, 0, color[2] / 255.0, 0};

        bmcv_image_storage_convert(handle, 1, &mask_bmimg_tmp, &mask_bmimg);
        bmcv_image_convert_to(handle, 1, bmcv_convert_to_attr, &mask_bmimg, &mask_bmimg);

        bmcv_width_align(handle, mask_bmimg, mask_bmimg_align);
        bmcv_copy_to_atrr_s attr = {left, top, 0, 0, 0, 0};

        bmcv_padding_atrr_t padding_atrr = {left, top, width, height, 0, 0, 0, 1};
        bmcv_rect_t rect = {0, 0, width, height};

        bmcv_image_vpp_convert_padding(handle, 1, mask_bmimg_align, &mask_bmimg_padding, &padding_atrr, &rect,
                                       BMCV_INTER_LINEAR);
        bmcv_image_bitwise_or(handle, frame, mask_bmimg_padding, frame);

        int colors_num = colors.size();
        // Draw a rectangle displaying the bounding box
        // bmcv_rect_t rect;
        rect.start_x = MIN(MAX(left, 0), frame.width);
        rect.start_y = MIN(MAX(top, 0), frame.height);
        rect.crop_w = MAX(MIN(width, frame.width - left), 0);
        rect.crop_h = MAX(MIN(height, frame.height - top), 0);
        bmcv_image_draw_rectangle(handle, frame, 1, &rect, 3, color[0], color[1], color[2]);
    }
    if (frame.image_format != 0) {
        bm_image frame2;
        bm_image_create(handle, frame.height, frame.width, FORMAT_YUV420P, frame.data_type, &frame2);
        bmcv_image_storage_convert(handle, 1, &frame, &frame2);
        bm_image_destroy(frame);
        frame = frame2;
    }
    if (put_text_flag) {
        for (int i = 0; i < result.size(); i++) {
            if(result[i].score < 0.25) continue;

            int left, top;
            left = result[i].x1;
            top = result[i].y1;
            int color_num = i;
            cv::Scalar color(colors[result[i].class_id % 25][0], colors[result[i].class_id % 25][1],
                             colors[result[i].class_id % 25][2]);
            std::string label = m_class_names[result[i].class_id] + ":" + cv::format("%.2f", result[i].score);
            bmcv_point_t org = {left, top};
            bmcv_color_t color2 = {color[0], color[1], color[2]};
            int thickness = 2;
            float fontScale = 2;
            if (BM_SUCCESS != bmcv_image_put_text(handle, frame, label.c_str(), org, color2, fontScale, thickness)) {
                std::cout << "bmcv put text error !!!" << std::endl;
            }
        }
    }
}
