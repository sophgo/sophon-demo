//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov5.hpp"

#include <fstream>
#include <string>
#include <vector>

#include "runtime_c5.h"
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},    {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {170, 255, 0},  {85, 255, 0},    {0, 255, 0},     {0, 255, 85},
    {0, 255, 170},  {0, 255, 255},   {0, 170, 255},   {0, 85, 255},
    {0, 0, 255},    {85, 0, 255},    {170, 0, 255},   {255, 0, 255},
    {255, 0, 170},  {255, 0, 85},    {255, 0, 0},     {255, 0, 255},
    {255, 85, 255}, {255, 170, 255}, {255, 255, 255}, {170, 255, 255},
    {85, 255, 255}};

YoloV5::YoloV5(bm_handle_t h, std::string bmodel_file, int dev_id, bool cpu_opt) {
    tpuRtStatus_t ret = tpuRtErrFailure;
    ret = tpuRtSetDevice(dev_id);
    assert(ret == tpuRtSuccess);
    use_cpu_opt = cpu_opt;
    handle = h;
    stream = h->stream;
    ret = tpuRtCreateNetContext(&net_ctx);
    assert(ret == tpuRtSuccess);
    ret = tpuRtLoadNet(bmodel_file.c_str(), net_ctx, &net);
    assert(ret == tpuRtSuccess);
    char** tmp_net_names = nullptr;
    int net_num = tpuRtGetNetNames(net, &tmp_net_names);
    for (int i = 0; i < net_num; ++i) {
        network_names.push_back(tmp_net_names[i]);
    }
    tpuRtFreeNetNames(tmp_net_names);
    net_info = tpuRtGetNetInfo(net, network_names[0].c_str());
}

YoloV5::~YoloV5() {
    std::cout << "YoloV5 dtor ..." << std::endl;
    tpuRtStatus_t tpuret = tpuRtErrFailure;
    bmcv_status_t ret = bmcvErrFailure;
    tpuret = tpuRtUnloadNet(net);
    assert(tpuret == tpuRtSuccess);
    tpuret = tpuRtDestroyNetContext(net_ctx);
    assert(tpuret == tpuRtSuccess);
    ret = bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
    assert(ret == bmcvSuccess);
    ret = bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
    assert(ret == bmcvSuccess);
    for (int i = 0; i < max_batch; i++) {
        ret = bm_image_destroy(&m_converto_imgs[i]);
        assert(ret == bmcvSuccess);
        ret = bm_image_destroy(&m_resized_imgs[i]);
        assert(ret == bmcvSuccess);
    }
}

int YoloV5::Init(float confThresh, float nmsThresh,
                 const std::string& coco_names_file) {
    m_confThreshold = confThresh;
    m_nmsThreshold = nmsThresh;
    
    tpuRtStatus_t tpuret = tpuRtErrFailure;
    bmcv_status_t ret = bmcvErrFailure;
    std::ifstream ifs(coco_names_file);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            line = line.substr(0, line.length() - 1);
            m_class_names.push_back(line);
        }
    }

    // 1. get input
    input_num = net_info.input.num;
    inputTensors.resize(input_num);
    for (int i = 0; i < net_info.stage_num; i++) {
        int b = net_info.stages[i].input_shapes[0].dims[0];
        max_batch = max_batch > b ? max_batch : b;
        batches.push_back(b);
    }

    auto stage_info = net_info.stages[0];
    m_net_h = stage_info.input_shapes[0].dims[2];
    m_net_w = stage_info.input_shapes[0].dims[3];

    auto input_dtype = net_info.input.dtypes[0];
    auto input_scale = net_info.input.scales[0];

    // 2. get output
    output_num = net_info.output.num;
    outputTensors.resize(output_num);
    assert(output_num == 1 || output_num == 3);
    min_dim = stage_info.output_shapes[0].num_dims;

    // 3. initialize tensors
    for (int i = 0; i < input_num; ++i) {
        inputTensors[i] = std::make_shared<tpuRtTensor_t>();
        inputTensors[i]->shape = net_info.stages[0].input_shapes[i];
        inputTensors[i]->dtype = net_info.input.dtypes[i];
    }

    for (int i = 0; i < output_num; ++i) {
        outputTensors[i].reset(new tpuRtTensor_t(), [&tpuret](tpuRtTensor_t *p) {
            tpuret = tpuRtFree(&p->data, 0);
            assert(tpuret == tpuRtSuccess);
            delete p;
            p = nullptr;
        });
        outputTensors[i]->shape = net_info.stages[0].output_shapes[i];
        outputTensors[i]->dtype = net_info.output.dtypes[i];
        int size = getTensorBytes(*outputTensors[i]);
        tpuret = tpuRtMalloc(&(outputTensors[i]->data), size, 0);
        assert(tpuret == tpuRtSuccess);
    }

    // 4. initialize bmimages
    m_resized_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);
    // some API only accept bm_image whose stride is aligned to 64
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for (int i = 0; i < max_batch; ++i) {
        ret = bm_image_create(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR,
                            DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
        assert(bmcvSuccess == ret);
    }
    ret = bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
    assert(bmcvSuccess == ret);
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (input_dtype == TPU_INT8) {
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    for (int i = 0; i < max_batch; ++i) {
        ret = bm_image_create(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR,
                            img_dtype, &m_converto_imgs[i]);
        assert(bmcvSuccess == ret);
    }
    ret = bm_image_alloc_contiguous_mem(max_batch, m_converto_imgs.data());
    assert(bmcvSuccess == ret);

    // 5.converto
    input_scale = input_scale * 1.0 / 255.f;
    converto_attr.alpha_0 = input_scale;
    converto_attr.beta_0 = 0;
    converto_attr.alpha_1 = input_scale;
    converto_attr.beta_1 = 0;
    converto_attr.alpha_2 = input_scale;
    converto_attr.beta_2 = 0;

    return 0;
}

void YoloV5::enableProfile(TimeStamp* ts) { m_ts = ts; }

int YoloV5::batch_size() { return max_batch; };

int YoloV5::Detect(const std::vector<bm_image>& input_images,
                   std::vector<YoloV5BoxVec>& boxes) {
    int ret = 0;
    // 3. preprocess
    m_ts->save("yolov5 preprocess", input_images.size());
    ret = pre_process(input_images);
    CV_Assert(ret == 0);
    m_ts->save("yolov5 preprocess", input_images.size());

    // 4. forward
    m_ts->save("yolov5 inference", input_images.size());
    ret = forward(inputTensors, outputTensors);
    CV_Assert(ret == 0);
    m_ts->save("yolov5 inference", input_images.size());

    // 5. post process
    m_ts->save("yolov5 postprocess", input_images.size());
    if (use_cpu_opt)
        ret = post_process_cpu_opt(input_images, boxes);
    else
        ret = post_process(input_images, boxes);
    CV_Assert(ret == 0);
    m_ts->save("yolov5 postprocess", input_images.size());
    return ret;
}

int YoloV5::pre_process(const std::vector<bm_image>& images) {
    bmcv_status_t ret = bmcvErrFailure;
    auto input_tensor = inputTensors[0];
    int image_n = images.size();
    // 1. resize image
    for (int i = 0; i < image_n; ++i) {
        bm_image image1 = images[i];
        bm_image image_aligned;
        bool need_copy = image1.width & (64 - 1);
        if (need_copy) {
            int stride1[3], stride2[3];
            ret = bm_image_get_stride(image1, stride1);
            assert(bmcvSuccess == ret);
            stride2[0] = FFALIGN(stride1[0], 64);
            stride2[1] = FFALIGN(stride1[1], 64);
            stride2[2] = FFALIGN(stride1[2], 64);
            ret = bm_image_create(handle, image1.height, image1.width,
                            image1.image_format, image1.data_type,
                            &image_aligned, stride2);
            assert(bmcvSuccess == ret);

            ret = bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
            assert(bmcvSuccess == ret);
            bmcv_copy_to_atrr_t copyToAttr;
            memset(&copyToAttr, 0, sizeof(copyToAttr));
            copyToAttr.start_x = 0;
            copyToAttr.start_y = 0;
            copyToAttr.if_padding = 1;
            ret = bmcv_image_copy_to(handle, copyToAttr, image1, image_aligned);
            assert(bmcvSuccess == ret);
        } else {
            image_aligned = image1;
        }
#if USE_ASPECT_RATIO
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(images[i].width, images[i].height,
                                              m_net_w, m_net_h, &isAlignWidth);
        bmcv_padding_attr_t padding_attr;
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

        bmcv_rect_t crop_rect{0, 0, static_cast<unsigned int>(image1.width), static_cast<unsigned int>(image1.height)};
        ret = bmcv_image_vpp_convert_padding(
            handle, 1, image_aligned, &m_resized_imgs[i], &padding_attr,
            &crop_rect, BMCV_INTER_NEAREST);
        assert(bmcvSuccess == ret);
#else
        ret = bmcv_image_vpp_convert(handle, 1, images[i],
                                          &m_resized_imgs[i]);
        assert(bmcvSuccess == ret);                                  
#endif

#if DUMP_FILE
        cv::Mat resized_img;
        cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
        std::string fname = cv::format("resized_img_%d.jpg", i);
        cv::imwrite(fname, resized_img);
#endif
        if (need_copy) {
            ret = bm_image_destroy(&image_aligned);
            assert(bmcvSuccess == ret);
        }
    }

    // 2. converto
    ret = bmcv_image_convert_to(handle, image_n, converto_attr,
                                m_resized_imgs.data(), m_converto_imgs.data());
    CV_Assert(ret == bmcvSuccess);

    // 3. attach to tensor
    if (image_n != max_batch) image_n = get_nearest_batch(batches, image_n);    
    media_mem_t input_dev_mem;
    ret = bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(),
                                       &input_dev_mem);
    assert(bmcvSuccess == ret);
    input_tensor->data = reinterpret_cast<void*>(input_dev_mem.phy_addr);

    // input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
    return 0;
}

int YoloV5::forward(
    std::vector<std::shared_ptr<tpuRtTensor_t>>& input_tensors,
    std::vector<std::shared_ptr<tpuRtTensor_t>>& output_tensors) {
    tpuRtTensor_t tempInputTensors[net_info.input.num];
    for (int i = 0; i < net_info.input.num; ++i)
        tempInputTensors[i] = *input_tensors[i];
    tpuRtTensor_t tempOutputTensors[net_info.output.num];
    for (int i = 0; i < net_info.output.num; ++i)
        tempOutputTensors[i] = *output_tensors[i];
    auto ret = tpuRtLaunchNet(net, tempInputTensors, tempOutputTensors,
                              network_names[0].c_str(), stream);
    assert(ret == tpuRtSuccess);
    return 0;
}

float YoloV5::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w,
                                      int dst_h, bool* pIsAligWidth) {
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

int YoloV5::post_process(const std::vector<bm_image>& images,
                         std::vector<YoloV5BoxVec>& detected_boxes) {
    YoloV5BoxVec yolobox_vec;
  std::vector<cv::Rect> bbox_vec;
  for(int batch_idx = 0; batch_idx < images.size(); ++ batch_idx)
  {
    yolobox_vec.clear();
    auto& frame = images[batch_idx];
    int frame_width = frame.width;
    int frame_height = frame.height;

    int tx1 = 0, ty1 = 0;
    float ratiox = (float)m_net_w / frame.width, ratioy = (float)m_net_h / frame.height;
#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(frame.width, frame.height, m_net_w, m_net_h, &isAlignWidth);
    ratiox = ratioy = ratio;
    if (isAlignWidth) {
      ty1 = (int)((m_net_h - (int)(frame_height*ratio)) / 2);
    }else{
      tx1 = (int)((m_net_w - (int)(frame_width*ratio)) / 2);
    }
#else
    float ratio = 1;
#endif

    int min_idx = 0;
    int box_num = 0;
    for(int i=0; i<output_num; i++){
      auto output_shape = net_info.stages[0].output_shapes[i];
      auto output_dims = outputTensors[i]->shape.num_dims;
      assert(output_dims == 3 || output_dims == 5);
      if(output_dims == 5){
        box_num += output_shape.dims[1] * output_shape.dims[2] * output_shape.dims[3];
      }

      if(min_dim>output_dims){
        min_idx = i;
        min_dim = output_dims;
      }
    }

    auto out_tensor = outputTensors[min_idx];
    int nout = outputTensors[min_idx]->shape.dims[min_dim - 1];
    m_class_num = nout - 5;

    float* output_data = nullptr;
    std::vector<float> decoded_data;

    if(min_dim ==3 && output_num !=1){
      std::cout<<"--> WARNING: the current bmodel has redundant outputs"<<std::endl;
      std::cout<<"             you can remove the redundant outputs to improve performance"<< std::endl;
      std::cout<<std::endl;
    }
    char *cpu_data = nullptr;
    if(min_dim == 5){
      LOG_TS(m_ts, "post 1: get output and decode");
      // std::cout<<"--> Note: Decoding Boxes"<<std::endl;
      // std::cout<<"          you can put the process into model during trace"<<std::endl;
      // std::cout<<"          which can reduce post process time, but forward time increases 1ms"<<std::endl;
      // std::cout<<std::endl;
      const std::vector<std::vector<std::vector<int>>> anchors{
        {{10, 13}, {16, 30}, {33, 23}},
          {{30, 61}, {62, 45}, {59, 119}},
          {{116, 90}, {156, 198}, {373, 326}}};
      const int anchor_num = anchors[0].size();
      assert(output_num == (int)anchors.size());
      assert(box_num>0);
      if((int)decoded_data.size() != box_num*nout){
        decoded_data.resize(box_num*nout);
      }
      float *dst = decoded_data.data();
      for(int tidx = 0; tidx < output_num; ++tidx) {
        auto output_tensor = outputTensors[tidx];
        int feat_c = output_tensor->shape.dims[1];
        int feat_h = output_tensor->shape.dims[2];
        int feat_w = output_tensor->shape.dims[3];
        int area = feat_h * feat_w;
        assert(feat_c == anchor_num);
        int feature_size = feat_h*feat_w*nout;
        cpu_data = get_cpu_data(output_tensor, stream);
        float *tensor_data = reinterpret_cast<float*>(cpu_data) + batch_idx*feat_c*area*nout;
        for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++)
        {
          float *ptr = tensor_data + anchor_idx*feature_size;
          for (int i = 0; i < area; i++) {
            dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * m_net_w;
            dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * m_net_h;
            dst[2] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[tidx][anchor_idx][0];
            dst[3] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[tidx][anchor_idx][1];
            dst[4] = sigmoid(ptr[4]);
            float score = dst[4];
            if (score > m_confThreshold) {
              for(int d=5; d<nout; d++){
                dst[d] = sigmoid(ptr[d]);
              }
            }
            dst += nout;
            ptr += nout;
          }
        }
      }
      output_data = decoded_data.data();
      LOG_TS(m_ts, "post 1: get output and decode");
    } else {
      LOG_TS(m_ts, "post 1: get output");
      assert(box_num == 0 || box_num == out_tensor->shape.dims[1]);
      box_num = out_tensor->shape.dims[1];
      cpu_data = get_cpu_data(out_tensor, stream);
      float *output_data = reinterpret_cast<float*>(cpu_data) + batch_idx*box_num*nout;
      LOG_TS(m_ts, "post 1: get output");
    }

    LOG_TS(m_ts, "post 2: filter boxes");
    int max_wh = 7680;
    bool agnostic = false;
    for (int i = 0; i < box_num; i++) {
      float* ptr = output_data+i*nout;
      float score = ptr[4];
      if (score > m_confThreshold) {
#if USE_MULTICLASS_NMS
        for (int j = 0; j < m_class_num; j++) {
          float confidence = ptr[5 + j];
          int class_id = j;
          if (confidence * score > m_confThreshold)
          {
              float centerX = ptr[0];
              float centerY = ptr[1];
              float width = ptr[2];
              float height = ptr[3];

              YoloV5Box box;
              if (!agnostic)
                box.x = centerX - width / 2 + class_id * max_wh;
              else
                box.x = centerX - width / 2;
              if (box.x < 0) box.x = 0;
              if (!agnostic)
                box.y = centerY - height / 2 + class_id * max_wh;
              else
                box.y = centerY - height / 2;
              if (box.y < 0) box.y = 0;
              box.width = width;
              box.height = height;
              box.class_id = class_id;
              box.score = confidence * score;
              yolobox_vec.push_back(box);
          }
        }
#else
        int class_id = argmax(&ptr[5], m_class_num);
        float confidence = ptr[class_id + 5];
        if (confidence * score > m_confThreshold)
        {
            float centerX = ptr[0];
            float centerY = ptr[1];
            float width = ptr[2];
            float height = ptr[3];

            YoloV5Box box;
            if (!agnostic)
              box.x = centerX - width / 2 + class_id * max_wh;
            else
              box.x = centerX - width / 2;
            if (box.x < 0) box.x = 0;
            if (!agnostic)
              box.y = centerY - height / 2 + class_id * max_wh;
            else
              box.y = centerY - height / 2;
            if (box.y < 0) box.y = 0;
            box.width = width;
            box.height = height;
            box.class_id = class_id;
            box.score = confidence * score;
            yolobox_vec.push_back(box);
        }
#endif
      }
    }
    LOG_TS(m_ts, "post 2: filter boxes");

    LOG_TS(m_ts, "post 3: nms");
    NMS(yolobox_vec, m_nmsThreshold);
    for (auto& box : yolobox_vec){
      if (!agnostic){
        box.x -= box.class_id * max_wh;
        box.y -= box.class_id * max_wh;
      }
      box.x = (box.x - tx1) / ratiox;
      box.y = (box.y - ty1) / ratioy;
      box.width = (box.width) / ratiox;
      box.height = (box.height) / ratioy;
    }
    LOG_TS(m_ts, "post 3: nms");

    detected_boxes.push_back(yolobox_vec);
    auto tpu_ret = tpuRtFreeHost(cpu_data);
    assert(tpu_ret == tpuRtSuccess);
  }

  return 0;
}

int YoloV5::post_process_cpu_opt(const std::vector<bm_image>& images,
                                 std::vector<YoloV5BoxVec>& detected_boxes) {
    YoloV5BoxVec yolobox_vec;
    std::vector<cv::Rect> bbox_vec;

    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
        yolobox_vec.clear();
        auto& frame = images[batch_idx];
        int frame_width = frame.width;
        int frame_height = frame.height;

        int tx1 = 0, ty1 = 0;
        float ratiox = (float)m_net_w / frame.width,
              ratioy = (float)m_net_h / frame.height;
#if USE_ASPECT_RATIO
        bool is_align_width = false;
        float ratio = get_aspect_scaled_ratio(
            frame.width, frame.height, m_net_w, m_net_h, &is_align_width);
        ratiox = ratioy = ratio;
        if (is_align_width) {
            ty1 = (int)((m_net_h - (int)(frame_height * ratio)) / 2);
        } else {
            tx1 = (int)((m_net_w - (int)(frame_width * ratio)) / 2);
        }
#endif

        int min_idx = 0;
        int box_num = 0;
        for (int i = 0; i < output_num; i++) {
            auto output_shape = net_info.stages[0].output_shapes[i];
            auto output_dims = output_shape.num_dims;
            assert(output_dims == 3 || output_dims == 5);
            if (output_dims == 5) {
                box_num += output_shape.dims[1] * output_shape.dims[2] *
                           output_shape.dims[3];
            }

            if (min_dim > output_dims) {
                min_idx = i;
                min_dim = output_dims;
            }
        }

        auto out_tensor = outputTensors[min_idx];
        auto out_shape = net_info.stages[0].output_shapes[min_idx];
        int nout = out_shape.dims[min_dim - 1];
        m_class_num = nout - 5;
#if USE_MULTICLASS_NMS
        int out_nout = nout;
#else
        int out_nout = 7;
#endif
        float transformed_m_confThreshold = -std::log(1 / m_confThreshold - 1);

        float* output_data = nullptr;
        std::vector<float> decoded_data;

        if (min_dim == 3 && output_num != 1) {
            std::cout << "--> WARNING: the current bmodel has redundant outputs"
                      << std::endl;
            std::cout << "             you can remove the redundant outputs to "
                         "improve performance"
                      << std::endl;
            std::cout << std::endl;
        }
        char *cpu_data = nullptr;
        if (min_dim == 5) {
            LOG_TS(m_ts, "post 1: get output and decode");
            // std::cout<<"--> Note: Decoding Boxes"<<std::endl;
            // std::cout<<"          you can put the process into model during
            // trace"<<std::endl; std::cout<<"          which can reduce post
            // process time, but forward time increases 1ms"<<std::endl;
            // std::cout<<std::endl;
            const std::vector<std::vector<std::vector<int>>> anchors{
                {{10, 13}, {16, 30}, {33, 23}},
                {{30, 61}, {62, 45}, {59, 119}},
                {{116, 90}, {156, 198}, {373, 326}}};
            const int anchor_num = anchors[0].size();
            assert(output_num == (int)anchors.size());
            assert(box_num > 0);
            if ((int)decoded_data.size() != box_num * out_nout) {
                decoded_data.resize(box_num * out_nout);
            }
            float* dst = decoded_data.data();
            for (int tidx = 0; tidx < output_num; ++tidx) {
                auto output_tensor = outputTensors[tidx];
                int feat_c = output_tensor->shape.dims[1];
                int feat_h = output_tensor->shape.dims[2];
                int feat_w = output_tensor->shape.dims[3];
                int area = feat_h * feat_w;
                assert(feat_c == anchor_num);
                int feature_size = feat_h * feat_w * nout;
                cpu_data = get_cpu_data(out_tensor, stream);
                float *tensor_data = reinterpret_cast<float*>(cpu_data) + 
                                        batch_idx * feat_c * area * nout;
                for (int anchor_idx = 0; anchor_idx < anchor_num;
                     anchor_idx++) {
                    float* ptr = tensor_data + anchor_idx * feature_size;
                    for (int i = 0; i < area; i++) {
                        if (ptr[4] <= transformed_m_confThreshold) {
                            ptr += nout;
                            continue;
                        }
                        dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) /
                                 feat_w * m_net_w;
                        dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) /
                                 feat_h * m_net_h;
                        dst[2] = pow((sigmoid(ptr[2]) * 2), 2) *
                                 anchors[tidx][anchor_idx][0];
                        dst[3] = pow((sigmoid(ptr[3]) * 2), 2) *
                                 anchors[tidx][anchor_idx][1];
                        dst[4] = sigmoid(ptr[4]);
#if USE_MULTICLASS_NMS
                        for (int d = 5; d < nout; d++) dst[d] = ptr[d];
#else
                        dst[5] = ptr[5];
                        dst[6] = 5;
                        for (int d = 6; d < nout; d++) {
                            if (ptr[d] > dst[5]) {
                                dst[5] = ptr[d];
                                dst[6] = d;
                            }
                        }
                        dst[6] -= 5;
#endif
                        dst += out_nout;
                        ptr += nout;
                    }
                }
            }
            output_data = decoded_data.data();
            box_num = (dst - output_data) / out_nout;
            LOG_TS(m_ts, "post 1: get output and decode");
        } else {
            LOG_TS(m_ts, "post 1: get output");
            assert(box_num == 0 || box_num == out_tensor->shape.dims[1]);
            box_num = out_tensor->shape.dims[1];
            cpu_data = get_cpu_data(out_tensor, stream);
            output_data = reinterpret_cast<float*>(cpu_data) + 
                                                batch_idx * box_num * nout;
            LOG_TS(m_ts, "post 1: get output");
        }

        LOG_TS(m_ts, "post 2: filter boxes");
        int max_wh = 7680;
        bool agnostic = false;
        for (int i = 0; i < box_num; i++) {
            float* ptr = output_data + i * out_nout;
            float score = ptr[4];
            float box_transformed_m_confThreshold =
                -std::log(score / m_confThreshold - 1);
            if (min_dim != 5)
                box_transformed_m_confThreshold = m_confThreshold / score;
#if USE_MULTICLASS_NMS
            assert(min_dim == 5);
            float centerX = ptr[0];
            float centerY = ptr[1];
            float width = ptr[2];
            float height = ptr[3];
            for (int j = 0; j < m_class_num; j++) {
                float confidence = ptr[5 + j];
                int class_id = j;
                if (confidence > box_transformed_m_confThreshold) {
                    YoloV5Box box;
                    if (!agnostic)
                        box.x = centerX - width / 2 + class_id * max_wh;
                    else
                        box.x = centerX - width / 2;
                    if (box.x < 0) box.x = 0;
                    if (!agnostic)
                        box.y = centerY - height / 2 + class_id * max_wh;
                    else
                        box.y = centerY - height / 2;
                    if (box.y < 0) box.y = 0;
                    box.width = width;
                    box.height = height;
                    box.class_id = class_id;
                    box.score = sigmoid(confidence) * score;
                    yolobox_vec.push_back(box);
                }
            }
#else
            int class_id = ptr[6];
            float confidence = ptr[5];
            if (min_dim != 5) {
                ptr = output_data + i * nout;
                score = ptr[4];
                class_id = argmax(&ptr[5], m_class_num);
                confidence = ptr[class_id + 5];
            }
            if (confidence > box_transformed_m_confThreshold) {
                float centerX = ptr[0];
                float centerY = ptr[1];
                float width = ptr[2];
                float height = ptr[3];

                YoloV5Box box;
                if (!agnostic)
                    box.x = centerX - width / 2 + class_id * max_wh;
                else
                    box.x = centerX - width / 2;
                if (box.x < 0) box.x = 0;
                if (!agnostic)
                    box.y = centerY - height / 2 + class_id * max_wh;
                else
                    box.y = centerY - height / 2;
                if (box.y < 0) box.y = 0;
                box.width = width;
                box.height = height;
                box.class_id = class_id;
                if (min_dim == 5) confidence = sigmoid(confidence);
                box.score = confidence * score;
                yolobox_vec.push_back(box);
            }
#endif
        }
        LOG_TS(m_ts, "post 2: filter boxes");

        LOG_TS(m_ts, "post 3: nms");
        NMS(yolobox_vec, m_nmsThreshold);
        for (auto& box : yolobox_vec) {
            if (!agnostic) {
                box.x -= box.class_id * max_wh;
                box.y -= box.class_id * max_wh;
            }
            box.x = (box.x - tx1) / ratiox;
            box.y = (box.y - ty1) / ratioy;
            box.width = (box.width) / ratiox;
            box.height = (box.height) / ratioy;
        }
        LOG_TS(m_ts, "post 3: nms");

        detected_boxes.push_back(yolobox_vec);
        auto tpu_ret = tpuRtFreeHost(cpu_data);
        assert(tpu_ret == tpuRtSuccess);
    }

    return 0;
}

int YoloV5::argmax(float* data, int num) {
    float max_value = 0.0;
    int max_index = 0;
    for (int i = 0; i < num; ++i) {
        float value = data[i];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }

    return max_index;
}

float YoloV5::sigmoid(float x) { return 1.0 / (1 + expf(-x)); }

void YoloV5::NMS(YoloV5BoxVec& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(),
              [](const YoloV5Box& a, const YoloV5Box& b) {
                  return a.score < b.score;
              });

    std::vector<float> areas(length);
    for (int i = 0; i < length; i++) {
        areas[i] = dets[i].width * dets[i].height;
    }

    while (index > 0) {
        int i = 0;
        while (i < index) {
            float left = std::max(dets[index].x, dets[i].x);
            float top = std::max(dets[index].y, dets[i].y);
            float right = std::min(dets[index].x + dets[index].width,
                                   dets[i].x + dets[i].width);
            float bottom = std::min(dets[index].y + dets[index].height,
                                    dets[i].y + dets[i].height);
            float overlap = std::max(0.0f, right - left + 0.00001f) *
                            std::max(0.0f, bottom - top + 0.00001f);
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

void YoloV5::drawPred(int classId, float conf, int left, int top, int right,
                      int bottom,
                      cv::Mat& frame)  // Draw the predicted bounding box
{
    // Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom),
                  cv::Scalar(0, 0, 255), 3);

    // Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if ((int)m_class_names.size() >= m_class_num) {
        label = this->m_class_names[classId] + ":" + label;
    } else {
        label = std::to_string(classId) + ":" + label;
    }

    // Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize =
        getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX,
                0.75, cv::Scalar(0, 255, 0), 1);
}

void YoloV5::draw_bmcv(bm_handle_t handle, int classId, float conf, int left,
                       int top, int width, int height, bm_image& frame,
                       bool put_text_flag)  // Draw the predicted bounding box
{
    if (conf < 0.25) return;
    int colors_num = colors.size();
    // Draw a rectangle displaying the bounding box
    bmcv_rect_t rect;
    rect.start_x = MIN(MAX(left, 0), frame.width);
    rect.start_y = MIN(MAX(top, 0), frame.height);
    rect.crop_w = MAX(MIN(width, frame.width - rect.start_x), 0);
    rect.crop_h = MAX(MIN(height, frame.height - rect.start_y), 0);
    int thickness = 2;

    if (rect.crop_w <= thickness * 2 || rect.crop_h <= thickness * 2) {
        std::cout << "width or height too small, this rect will not be drawed: "
                  << "[" << rect.start_x << ", " << rect.start_y << ", "
                  << rect.crop_w << ", " << rect.crop_h << "]" << std::endl;
    } else {
        bmcv_image_draw_rectangle(
            handle, frame, 1, &rect, thickness, colors[classId % colors_num][0],
            colors[classId % colors_num][1], colors[classId % colors_num][2]);
    }
}
