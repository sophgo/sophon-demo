//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "FeatureExtractor.h"
extern int cnt;
FeatureExtractor::FeatureExtractor(std::shared_ptr<BMNNContext> context) : m_bmContext(context) {
    std::cout << "FeatureExtractor ctor .." << std::endl;
}

FeatureExtractor::~FeatureExtractor() {
    std::cout << "FeatureExtractor dtor ..." << std::endl;
    bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
    for (int i = 0; i < max_batch; i++) {
        bm_image_destroy(m_converto_imgs[i]);
        bm_image_destroy(m_resized_imgs[i]);
    }
}

void FeatureExtractor::Init() {
    // 1. get network
    m_bmNetwork = m_bmContext->network(0);

    // 2. get input
    max_batch = m_bmNetwork->maxBatch();
    auto tensor = m_bmNetwork->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];

    // 3. get output feature dim
    k_feature_dim = m_bmNetwork->outputTensor(0)->get_shape()->dims[1];
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

    // 5.converto, RGB mean std.
    float input_scale = tensor->get_scale();
    input_scale = input_scale * 1.0 / 255.f;
    converto_attr.alpha_0 = input_scale / 0.229;
    converto_attr.beta_0 = -0.485 / 0.229;
    converto_attr.alpha_1 = input_scale / 0.224;
    converto_attr.beta_1 = -0.456 / 0.224;
    converto_attr.alpha_2 = input_scale / 0.225;
    converto_attr.beta_2 = -0.406 / 0.225;
}
int FeatureExtractor::batch_size() {
    return max_batch;
};
int FeatureExtractor::pre_process(const bm_image& image, std::vector<bmcv_rect_t> crop_rects_batch) {
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
    int image_n = crop_rects_batch.size();
    // 1. crop and resize image
    int ret = 0;
    // std::cout<<image.width<<" "<<image.height<<std::endl;
    // std::cout<< crop_rects_batch[0].start_x <<" "<< crop_rects_batch[0].start_y <<" "<<crop_rects_batch[0].crop_w <<"
    // "<<crop_rects_batch[0].crop_h<<std::endl;
#if 0
    bm_image crops[crop_rects_batch.size()];
    bmcv_resize_t resize_attr[crop_rects_batch.size()];
    bmcv_resize_image resize_image[crop_rects_batch.size()];
    for (int i = 0; i < crop_rects_batch.size(); i++) {
        bm_image_create(m_bmContext->handle(), crop_rects_batch[i].crop_h, crop_rects_batch[i].crop_w,
                        image.image_format, image.data_type, &crops[i]);
        resize_attr[i] = {0,
                          0,
                          crop_rects_batch[i].crop_w,
                          crop_rects_batch[i].crop_h,
                          m_resized_imgs[i].width,
                          m_resized_imgs[i].height};
        resize_image[i] = {&resize_attr[i], 1, 1, 0, 0, 0, BMCV_INTER_LINEAR};
    }
    ret = bmcv_image_crop(m_bmContext->handle(), image_n, crop_rects_batch.data(), image, crops);
    ret = bmcv_image_resize(m_bmContext->handle(), image_n, resize_image, crops, m_resized_imgs.data());
    for (int i = 0; i < crop_rects_batch.size(); i++) {
        bm_image_destroy(crops[i]);
    }
#else
    ret = bmcv_image_vpp_convert(m_bmContext->handle(), image_n, image, m_resized_imgs.data(), crop_rects_batch.data());
#endif
    assert(BM_SUCCESS == ret);
#if 0
    for(int i =0; i < m_resized_imgs.size(); i++){
        cv::Mat resized_img;
        cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
        std::string fname = cv::format("results/resized_img_%d.jpg", cnt++);
        cv::imwrite(fname, resized_img);
    }
#endif
    // 2. converto
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
#if 0
    for(int i = 0; i<3*128*64; i++){
        if(*(input_tensor->get_cpu_data()+i)>100){
            exit(1);
        }
    }
#endif
    return 0;
}

bool FeatureExtractor::getRectsFeature(const bm_image& image, DETECTIONS& det) {
    std::vector<bmcv_rect_t> crop_rects_batch;

    // align bmimage stride.
    bm_image image1 = image;
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
    int start = 0;
    for (int i = 0; i < det.size(); i++) {
        bmcv_rect_t crop_rect;
        crop_rect.start_x = det[i].tlwh(0);
        crop_rect.start_y = det[i].tlwh(1);
        crop_rect.crop_w = det[i].tlwh(2);
        crop_rect.crop_h = det[i].tlwh(3);
        // crop_rect.start_x = MIN(MAX(int(dbox.tlwh(0)), 0), image.width);
        // crop_rect.start_y = MIN(MAX(int(dbox.tlwh(1)), 0), image.height);
        // crop_rect.crop_w = MAX(MIN(int(dbox.tlwh(2)), image.width - int(dbox.tlwh(0))), 0);
        // crop_rect.crop_h = MAX(MIN(int(dbox.tlwh(3)), image.height - int(dbox.tlwh(1))), 0);
        crop_rects_batch.push_back(crop_rect);
        if (crop_rects_batch.size() == batch_size() || (i == det.size() - 1 && !crop_rects_batch.empty())) {
            LOG_TS(m_ts, "extractor preprocess");
            CV_Assert(0 == pre_process(image_aligned, crop_rects_batch));
            LOG_TS(m_ts, "extractor preprocess");

            LOG_TS(m_ts, "extractor inference");
            CV_Assert(0 == m_bmNetwork->forward());
            LOG_TS(m_ts, "extractor inference");

            CV_Assert(0 == post_process(det, start, crop_rects_batch.size()));
            start += crop_rects_batch.size();
            crop_rects_batch.clear();
        }
    }
    if (need_copy)
        bm_image_destroy(image_aligned);  // very important, make sure you won't forget free bmimage.
    return true;
}

int FeatureExtractor::post_process(DETECTIONS& det, int start, int crop_size) {
    CV_Assert(start + crop_size <= det.size());
    // auto stream = m_bmNetwork->outputTensor(0)->get_cpu_data(); //why this code has bug?

    for (int i = start; i < start + crop_size; i++) {
        det[i].feature.resize(1, k_feature_dim);
        memcpy(det[i].feature.data(), m_bmNetwork->outputTensor(0)->get_cpu_data() + (i - start) * k_feature_dim,
               k_feature_dim * sizeof(float));
        // for (int j = 0; j < k_feature_dim; j++)
        // {
        //     // det[i].feature[j] = *(m_bmNetwork->outputTensor(0)->get_cpu_data() + (i - start) * k_feature_dim + j);
        //     //     det[i].feature[j] = stream[(i - start) * k_feature_dim + j];
        //     std::cout << det[i].feature[j] << " ";
        // }
    }
    // std::cout << std::endl;
    return 0;
}
