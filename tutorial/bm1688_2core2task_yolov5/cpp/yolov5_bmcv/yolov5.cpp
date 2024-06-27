#include "yolov5.hpp"

#define USE_ASPECT_RATIO 1
#define USE_MULTICLASS_NMS 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

YOLOv5::YOLOv5(int dev_id,
               std::string bmodel_path,
               float conf_thresh,
               float nms_thresh,
               const std::string& coco_names_file)
    : m_conf_thresh(conf_thresh), m_nms_thresh(nms_thresh) {
    bm_dev_request(&m_handle, dev_id);
    m_bmrt = bmrt_create(m_handle);
    bmrt_load_bmodel(m_bmrt, bmodel_path.c_str());
    bmrt_get_network_names(m_bmrt, &m_net_names);
    m_net_info = bmrt_get_network_info(m_bmrt, m_net_names[0]);

    struct bm_misc_info misc_info;
    bm_get_misc_info(m_handle, &misc_info);
    can_mmap = misc_info.pcie_soc_mode == 1;

    // net info
    // 这个yolov5 bmodel只有一个stage
    m_batch_size = m_net_info->stages[0].input_shapes[0].dims[0];
    m_net_w = m_net_info->stages[0].input_shapes[0].dims[2];
    m_net_h = m_net_info->stages[0].input_shapes[0].dims[3];

    // 预处理相关
    m_input_num = m_net_info->input_num;
    m_input_shape = m_net_info->stages[0].input_shapes[0];
    m_input_dtype = m_net_info->input_dtypes[0];
    img_dtype = m_input_dtype == BM_INT8 ? DATA_TYPE_EXT_1N_BYTE_SIGNED : DATA_TYPE_EXT_FLOAT32;
    m_input_scale = m_net_info->input_scales[0] * 1.0 / 255.f;
    m_converto_attr.alpha_0 = m_input_scale;
    m_converto_attr.beta_0 = 0;
    m_converto_attr.alpha_1 = m_input_scale;
    m_converto_attr.beta_1 = 0;
    m_converto_attr.alpha_2 = m_input_scale;
    m_converto_attr.beta_2 = 0;

    // 预处理resize用的bmimage
    m_resized_bmimgs.resize(m_batch_size);
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for (int i = 0; i < m_batch_size; i++) {
        auto ret = bm_image_create(m_handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                   &m_resized_bmimgs[i], strides);
    }
    auto ret = bm_image_alloc_contiguous_mem(m_batch_size, m_resized_bmimgs.data());

    // 后处理相关
    m_output_num = m_net_info->output_num;
    for (int i = 0; i < m_output_num; i++) {
        m_output_scales.emplace_back(m_net_info->output_scales[i]);
        m_output_shapes.emplace_back(m_net_info->stages[0].output_shapes[i]);
        m_output_dtypes.emplace_back(m_net_info->output_dtypes[i]);
    }

    // coco
    std::ifstream ifs(coco_names_file);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            line = line.substr(0, line.length() - 1);
            m_class_names.push_back(line);
        }
    }
}

YOLOv5::~YOLOv5() {
    // pre
    bm_image_free_contiguous_mem(m_batch_size, m_resized_bmimgs.data());
    for (int i = 0; i < m_batch_size; i++) {
        bm_image_destroy(m_resized_bmimgs[i]);
    }

    // runtime
    bmrt_destroy(m_bmrt);
    bm_dev_free(m_handle);
}

int YOLOv5::get_batch_size() {
    return m_batch_size;
}

int YOLOv5::yolov5_detect(std::vector<cv::Mat>& dec_images, std::vector<YoloV5BoxVec>& boxes) {
    std::vector<bm_tensor_t> input_tensors(m_input_num);
    input_tensors[0].shape = m_input_shape;
    input_tensors[0].dtype = m_input_dtype;

    std::vector<bm_tensor_t> output_tensors(m_output_num);
    //   for (int i = 0; i < m_output_num; i++){
    //     output_tensors[i].dtype = m_output_dtypes[i];
    //     output_tensors[i].shape = m_output_shapes[i];
    //   }
    int ret = 0;
    ret = yolov5_preprocess(dec_images, input_tensors);
    assert(ret == 0);
    ret = yolov5_inference(input_tensors, output_tensors);
    assert(ret == 0);
    ret = yolov5_postprocess(dec_images, output_tensors, boxes);
    assert(ret == 0);
    return ret;
}

void YOLOv5::yolov5_draw(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame) {
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3);

    std::string label = cv::format("%.2f", conf);

    label = m_class_names[classId] + ":" + label;

    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
}

// 1b or 4b
int YOLOv5::yolov5_preprocess(std::vector<cv::Mat>& dec_images, std::vector<bm_tensor_t>& input_tensors) {
    // resize需要单图做，但convertto不需要
    for (int i = 0; i < dec_images.size(); i++) {
        auto dec_image = dec_images[i];

        bm_image bmimg;
        cv::bmcv::toBMI(dec_image, &bmimg);
        bm_image bmimg_aligned;
        bool need_copy = bmimg.width & (64 - 1);
        if (need_copy) {
            int stride1[3], stride2[3];
            bm_image_get_stride(bmimg, stride1);
            stride2[0] = FFALIGN(stride1[0], 64);
            stride2[1] = FFALIGN(stride1[1], 64);
            stride2[2] = FFALIGN(stride1[2], 64);
            assert(BM_SUCCESS == bm_image_create(m_handle, bmimg.height, bmimg.width, bmimg.image_format, bmimg.data_type, &bmimg_aligned,
                            stride2));

            assert(BM_SUCCESS == bm_image_alloc_dev_mem(bmimg_aligned, 1));
            bmcv_copy_to_atrr_t copyToAttr;
            memset(&copyToAttr, 0, sizeof(copyToAttr));
            copyToAttr.start_x = 0;
            copyToAttr.start_y = 0;
            copyToAttr.if_padding = 1;
            bmcv_image_copy_to(m_handle, copyToAttr, bmimg, bmimg_aligned);
        } else {
            bmimg_aligned = bmimg;
        }

        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(bmimg.width, bmimg.height, m_net_w, m_net_h, &isAlignWidth);
        bmcv_padding_atrr_t padding_attr;
        memset(&padding_attr, 0, sizeof(padding_attr));
        padding_attr.dst_crop_sty = 0;
        padding_attr.dst_crop_stx = 0;
        padding_attr.padding_b = 114;
        padding_attr.padding_g = 114;
        padding_attr.padding_r = 114;
        padding_attr.if_memset = 1;
        if (isAlignWidth) {
            padding_attr.dst_crop_h = bmimg.height * ratio;
            padding_attr.dst_crop_w = m_net_w;

            int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
            padding_attr.dst_crop_sty = ty1;
            padding_attr.dst_crop_stx = 0;
        } else {
            padding_attr.dst_crop_h = m_net_h;
            padding_attr.dst_crop_w = bmimg.width * ratio;

            int tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
            padding_attr.dst_crop_sty = 0;
            padding_attr.dst_crop_stx = tx1;
        }

        bmcv_rect_t crop_rect{0, 0, bmimg.width, bmimg.height};
        auto ret = bmcv_image_vpp_convert_padding(m_handle, 1, bmimg_aligned, &m_resized_bmimgs[i], &padding_attr,
                                                  &crop_rect, BMCV_INTER_NEAREST);

        assert(BM_SUCCESS == ret);

        if (need_copy) {
            bm_image_destroy(bmimg_aligned);
        }
        bm_image_destroy(bmimg);
    }

    // 这里先bmlib申请了连续batch_size的内存，做归一化的bmimage内存是attach的，
    // 因为后面tensor需要的是相同的dev mem，这里申请的在推理完成后会进行释放（推理函数中）
    std::vector<bm_image> converto_bmimgs(m_batch_size);
    for (int i = 0; i < m_batch_size; i++) {
        bm_image_create(m_handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, &converto_bmimgs[i]);
    }

    int size_byte = 0;
    bm_device_mem_t tensor_mem;
    bm_image_get_byte_size(converto_bmimgs[0], &size_byte);
    assert(BM_SUCCESS == bm_malloc_device_byte_heap(m_handle, &tensor_mem, 0, size_byte * m_batch_size));

    assert(BM_SUCCESS == bm_image_attach_contiguous_mem(m_batch_size, converto_bmimgs.data(), tensor_mem));

    assert(BM_SUCCESS == bmcv_image_convert_to(m_handle, m_batch_size, m_converto_attr, m_resized_bmimgs.data(), converto_bmimgs.data()));

    assert(BM_SUCCESS == bm_image_detach_contiguous_mem(m_batch_size, converto_bmimgs.data()));

    for (int i = 0; i < m_batch_size; i++) {
        bm_image_destroy(converto_bmimgs[i]);
    }

    input_tensors[0].device_mem = tensor_mem;
    return 0;
}

int YOLOv5::yolov5_inference(std::vector<bm_tensor_t>& input_tensors, std::vector<bm_tensor_t>& output_tensors) {
    bool ok = bmrt_launch_tensor_multi_cores(m_bmrt, m_net_names[0], input_tensors.data(), m_input_num, output_tensors.data(),
                                   m_output_num, false, false, &m_core_id, 1);
    if (!ok) {
        std::cout << "bmrt_launch_tensor_multi_cores() failed." << std::endl;
        return -1;
    }
    assert(BM_SUCCESS == bm_thread_sync_from_core(m_handle, m_core_id));

    for (auto& input_tensor : input_tensors) {
        bm_free_device(m_handle, input_tensor.device_mem);
    }
    return 0;
}

// 4b 时对3输出模型，每个tensor+偏移量对应第n张图片的tensor内容
int YOLOv5::yolov5_postprocess(std::vector<cv::Mat>& dec_images,
                               std::vector<bm_tensor_t>& output_tensors,
                               std::vector<YoloV5BoxVec>& box_data) {
    int image_nums = dec_images.size();
    YoloV5BoxVec yolobox_vec;
    std::vector<cv::Rect> bbox_vec;
    std::vector<float*> tensor_datas(m_output_num);
    for (int tidx = 0; tidx < m_output_num; tidx++) {
        tensor_datas[tidx] = get_cpu_data(output_tensors[tidx], tidx);
    }

    for (int batch_idx = 0; batch_idx < image_nums; ++batch_idx) {
        yolobox_vec.clear();

        auto frame = dec_images[batch_idx];
        int frame_width = frame.cols;
        int frame_height = frame.rows;

        int tx1 = 0, ty1 = 0;

        bool is_align_width = false;
        float ratio = get_aspect_scaled_ratio(frame_width, frame_height, m_net_w, m_net_h, &is_align_width);
        if (is_align_width) {
            ty1 = (int)((m_net_h - (int)(frame_height * ratio)) / 2);
        } else {
            tx1 = (int)((m_net_w - (int)(frame_width * ratio)) / 2);
        }

        int box_num = 0;

        auto output_shape = output_tensors[0].shape;
        auto output_dims = output_shape.num_dims;
        assert(output_dims == 3 || output_dims == 5);
        if (output_dims == 5) {
            box_num += output_shape.dims[1] * output_shape.dims[2] * output_shape.dims[3];
        }

        int min_dim = output_dims;

        int nout = output_tensors[0].shape.dims[output_dims - 1];
        m_class_num = nout - 5;
#if USE_MULTICLASS_NMS
        int out_nout = nout;
#else
        int out_nout = 7;
#endif
        float transformed_m_confThreshold = -std::log(1 / m_conf_thresh - 1);

        float* output_data = nullptr;
        std::vector<float> decoded_data;

        if (min_dim == 3 && m_output_num != 1) {
            std::cout << "--> WARNING: the current bmodel has redundant outputs" << std::endl;
            std::cout << "             you can remove the redundant outputs to improve performance" << std::endl;
            std::cout << std::endl;
        }

        if (min_dim == 5) {
            // std::cout<<"--> Note: Decoding Boxes"<<std::endl;
            // std::cout<<"          you can put the process into model during trace"<<std::endl;
            // std::cout<<"          which can reduce post process time, but forward time increases 1ms"<<std::endl;
            // std::cout<<std::endl;
            const std::vector<std::vector<std::vector<int>>> anchors{
                {{10, 13}, {16, 30}, {33, 23}}, {{30, 61}, {62, 45}, {59, 119}}, {{116, 90}, {156, 198}, {373, 326}}};
            const int anchor_num = anchors[0].size();
            assert(m_output_num == (int)anchors.size());
            assert(box_num > 0);
            if ((int)decoded_data.size() != box_num * out_nout) {
                decoded_data.resize(box_num * out_nout);
            }

            float* dst = decoded_data.data();
            for (int tidx = 0; tidx < m_output_num; ++tidx) {
                auto output_tensor = output_tensors[tidx];
                int feat_c = output_tensor.shape.dims[1];
                int feat_h = output_tensor.shape.dims[2];
                int feat_w = output_tensor.shape.dims[3];
                int area = feat_h * feat_w;
                assert(feat_c == anchor_num);
                int feature_size = feat_h * feat_w * nout;

                float* tensor_data = tensor_datas[tidx] + batch_idx * feat_c * area * nout;

                for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++) {
                    float* ptr = tensor_data + anchor_idx * feature_size;
                    for (int i = 0; i < area; i++) {
                        if (ptr[4] <= transformed_m_confThreshold) {
                            ptr += nout;
                            continue;
                        }
                        dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * m_net_w;
                        dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * m_net_h;
                        dst[2] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[tidx][anchor_idx][0];
                        dst[3] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[tidx][anchor_idx][1];
                        dst[4] = sigmoid(ptr[4]);
#if USE_MULTICLASS_NMS
                        for (int d = 5; d < nout; d++)
                            dst[d] = ptr[d];
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

        } else {
            assert(box_num == 0 || box_num == output_tensors[0].shape.dims[1]);
            box_num = output_tensors[0].shape.dims[1];
            output_data = tensor_datas[0] + batch_idx * box_num * nout;
        }

        int max_wh = 7680;
        bool agnostic = false;
        for (int i = 0; i < box_num; i++) {
            float* ptr = output_data + i * out_nout;
            float score = ptr[4];
            float box_transformed_m_confThreshold = -std::log(score / m_conf_thresh - 1);
            if (min_dim != 5)
                box_transformed_m_confThreshold = m_conf_thresh / score;
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
                    if (box.x < 0)
                        box.x = 0;
                    if (!agnostic)
                        box.y = centerY - height / 2 + class_id * max_wh;
                    else
                        box.y = centerY - height / 2;
                    if (box.y < 0)
                        box.y = 0;
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
                if (box.x < 0)
                    box.x = 0;
                if (!agnostic)
                    box.y = centerY - height / 2 + class_id * max_wh;
                else
                    box.y = centerY - height / 2;
                if (box.y < 0)
                    box.y = 0;
                box.width = width;
                box.height = height;
                box.class_id = class_id;
                if (min_dim == 5)
                    confidence = sigmoid(confidence);
                box.score = confidence * score;
                yolobox_vec.push_back(box);
            }
#endif
        }

        NMS(yolobox_vec, m_conf_thresh);
        if (!agnostic)
            for (auto& box : yolobox_vec) {
                box.x -= box.class_id * max_wh;
                box.y -= box.class_id * max_wh;
                box.x = (box.x - tx1) / ratio;
                if (box.x < 0)
                    box.x = 0;
                box.y = (box.y - ty1) / ratio;
                if (box.y < 0)
                    box.y = 0;
                box.width = (box.width) / ratio;
                if (box.x + box.width >= frame_width)
                    box.width = frame_width - box.x;
                box.height = (box.height) / ratio;
                if (box.y + box.height >= frame_height)
                    box.height = frame_height - box.y;
            }
        else
            for (auto& box : yolobox_vec) {
                box.x = (box.x - tx1) / ratio;
                if (box.x < 0)
                    box.x = 0;
                box.y = (box.y - ty1) / ratio;
                if (box.y < 0)
                    box.y = 0;
                box.width = (box.width) / ratio;
                if (box.x + box.width >= frame_width)
                    box.width = frame_width - box.x;
                box.height = (box.height) / ratio;
                if (box.y + box.height >= frame_height)
                    box.height = frame_height - box.y;
            }

        box_data.emplace_back(yolobox_vec);
    }

    // 释放output tensor内存
    for (int i = 0; i < m_output_num; i++) {
        bm_free_device(m_handle, output_tensors[i].device_mem);
    }

    for (int i = 0; i < m_output_num; i++) {
        if (can_mmap && BM_FLOAT32 == output_tensors[i].dtype) {
            int tensor_size = bm_mem_get_device_size(output_tensors[i].device_mem);
            bm_status_t ret = bm_mem_unmap_device_mem(m_handle, tensor_datas[i], tensor_size);
            assert(BM_SUCCESS == ret);
        } else {
            delete tensor_datas[i];
        }
    }
    return 0;
}

float* YOLOv5::get_cpu_data(bm_tensor_t& tensor, int out_idx) {
    float* cpu_data;
    bm_status_t ret;
    float* pFP32 = nullptr;
    int count = bmrt_shape_count(&tensor.shape);

    if (can_mmap) {
        if (tensor.dtype == BM_FLOAT32) {
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(m_handle, &tensor.device_mem, &addr);
            assert(BM_SUCCESS == ret);
            ret = bm_mem_invalidate_device_mem(m_handle, &tensor.device_mem);
            assert(BM_SUCCESS == ret);
            pFP32 = (float*)addr;
        } else if (BM_INT8 == tensor.dtype) {
            int8_t* pI8 = nullptr;
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(m_handle, &tensor.device_mem, &addr);
            assert(BM_SUCCESS == ret);
            ret = bm_mem_invalidate_device_mem(m_handle, &tensor.device_mem);
            assert(BM_SUCCESS == ret);
            pI8 = (int8_t*)addr;

            // dtype convert
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            for (int i = 0; i < count; ++i) {
                pFP32[i] = pI8[i] * m_output_scales[out_idx];
            }
            ret = bm_mem_unmap_device_mem(m_handle, pI8, bm_mem_get_device_size(tensor.device_mem));
            assert(BM_SUCCESS == ret);
        } else if (tensor.dtype == BM_INT32) {
            int32_t* pI32 = nullptr;
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(m_handle, &tensor.device_mem, &addr);
            assert(BM_SUCCESS == ret);
            ret = bm_mem_invalidate_device_mem(m_handle, &tensor.device_mem);
            assert(BM_SUCCESS == ret);
            pI32 = (int32_t*)addr;
            // dtype convert
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            for (int i = 0; i < count; ++i) {
                pFP32[i] = pI32[i] * m_output_scales[out_idx];
            }
            ret = bm_mem_unmap_device_mem(m_handle, pI32, bm_mem_get_device_size(tensor.device_mem));
            assert(BM_SUCCESS == ret);
        } else {
            std::cout << "NOT support dtype=" << tensor.dtype << std::endl;
        }
    } else {
        // the common method using d2s
        if (tensor.dtype == BM_FLOAT32) {
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(m_handle, pFP32, tensor.device_mem, count * sizeof(float));
            assert(BM_SUCCESS == ret);
        } else if (BM_INT8 == tensor.dtype) {
            int8_t* pI8 = nullptr;
            int tensor_size = bmrt_tensor_bytesize(&tensor);
            pI8 = new int8_t[tensor_size];
            assert(pI8 != nullptr);

            // dtype convert
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(m_handle, pI8, tensor.device_mem, tensor_size);
            assert(BM_SUCCESS == ret);
            for (int i = 0; i < count; ++i) {
                pFP32[i] = pI8[i] * m_output_scales[out_idx];
            }
            delete[] pI8;
        } else if (tensor.dtype == BM_INT32) {
            int32_t* pI32 = nullptr;
            int tensor_size = bmrt_tensor_bytesize(&tensor);
            pI32 = new int32_t[tensor_size];
            assert(pI32 != nullptr);

            // dtype convert
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(m_handle, pI32, tensor.device_mem, tensor_size);
            assert(BM_SUCCESS == ret);
            for (int i = 0; i < count; ++i) {
                pFP32[i] = pI32[i] * m_output_scales[out_idx];
            }
            delete[] pI32;

        } else {
            std::cout << "NOT support dtype=" << tensor.dtype << std::endl;
        }
    }

    cpu_data = pFP32;
    return cpu_data;
}

int YOLOv5::argmax(float* data, int num) {
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

float YOLOv5::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
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

float YOLOv5::sigmoid(float x) {
    return 1.0 / (1 + expf(-x));
}

void YOLOv5::NMS(YoloV5BoxVec& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const YoloV5Box& a, const YoloV5Box& b) { return a.score < b.score; });

    std::vector<float> areas(length);
    for (int i = 0; i < length; i++) {
        areas[i] = dets[i].width * dets[i].height;
    }

    while (index > 0) {
        int i = 0;
        while (i < index) {
            float left = std::max(dets[index].x, dets[i].x);
            float top = std::max(dets[index].y, dets[i].y);
            float right = std::min(dets[index].x + dets[index].width, dets[i].x + dets[i].width);
            float bottom = std::min(dets[index].y + dets[index].height, dets[i].y + dets[i].height);
            float overlap = std::max(0.0f, right - left + 0.00001f) * std::max(0.0f, bottom - top + 0.00001f);
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