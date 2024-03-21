// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#include "yolov5.h"

int Yolov5::init(int dev_id, std::string bmodel_file,std::string tpu_kernel_module_path){
    bm_dev_request(&handle_,dev_id);

    //load model
    ctx_ptr_ = std::make_shared<BModelContext>(dev_id, bmodel_file);
    net_ptr_ = ctx_ptr_->network(0);
    int batch = net_ptr_->batch();
    if(batch!=1){
        std::cout<<"Error! Only support 1 batch bmodel!"<<std::endl;
        return -1;
    }
    net_height_ = net_ptr_->get_input_tensor(0)->shape.dims[2];
    net_width_ = net_ptr_->get_input_tensor(0)->shape.dims[3];
    in_tensor_num_ = net_ptr_->m_netinfo->input_num;
    out_tensor_num_ = net_ptr_->m_netinfo->output_num;

    //preprocess init
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (net_ptr_->get_input_tensor(0)->dtype == BM_INT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }

    int aligned_net_w = FFALIGN(net_width_, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w}; 

    ret_ = bm_image_create(handle_, net_height_, net_width_, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &resized_img_, strides);
    assert(ret_==BM_SUCCESS);

    ret_ = bm_image_alloc_contiguous_mem(1, &resized_img_);
    assert(ret_==BM_SUCCESS);  

    input_mem_idx_=-1;
    output_mem_idx_=-1;
    int buffer_num=que_size_+2;
    converto_imgs_.resize(buffer_num);
    input_mem_buffer_.resize(buffer_num);
    for(int i=0;i<buffer_num;i++){
        ret_ = bm_image_create(handle_, net_height_, net_width_, FORMAT_RGB_PLANAR, img_dtype, &converto_imgs_[i]);
        assert(ret_==BM_SUCCESS);
        ret_ = bm_image_alloc_contiguous_mem(1, &converto_imgs_[i]);
        assert(ret_==BM_SUCCESS);
        bm_image_get_contiguous_device_mem(1, &converto_imgs_[i], &input_mem_buffer_[i]);
    }
    float input_scale = net_ptr_->m_netinfo->input_scales[0];
    input_scale = input_scale * 1.0 / 255.f;
    converto_attr_.alpha_0 = input_scale;
    converto_attr_.beta_0 = 0;
    converto_attr_.alpha_1 = input_scale;
    converto_attr_.beta_1 = 0;
    converto_attr_.alpha_2 = input_scale;
    converto_attr_.beta_2 = 0;

    //process init
    output_mem_buffer_.resize(buffer_num,std::vector<bm_device_mem_t>(out_tensor_num_));
    for(int i=0;i<buffer_num;i++){
        for(int j=0;j < out_tensor_num_; j++){
            bm_tensor_t* out_tensor = net_ptr_->get_output_tensor(j);
            uint64_t bytes_num = bmrt_shape_count(&(out_tensor->shape))*bmruntime::ByteSize(out_tensor->dtype);
            ret_ = bm_malloc_device_byte(handle_,&output_mem_buffer_[i][j],bytes_num);
            assert(ret_==BM_SUCCESS);
        }
    }
  
    
    //tpu kernal post process init

    tpu_kernel_module_t tpu_module;
    tpu_module = tpu_kernel_load_module_file(handle_, tpu_kernel_module_path.c_str()); 
    func_id = tpu_kernel_get_function(handle_, tpu_module, "tpu_kernel_api_yolov5_detect_out");
    

    int out_len_max = 25200*7;
    boxs_sysmem_ = std::vector<float>(out_len_max);
    detect_num_sysmem_ = 0;

    ret_ = bm_malloc_device_byte(handle_, &boxs_devmem_, out_len_max * sizeof(float));
    assert(BM_SUCCESS == ret_);
    ret_ = bm_malloc_device_byte(handle_, &detect_num_devmem_, 1*sizeof(int32_t));
    assert(BM_SUCCESS == ret_);

    /*initialize api for tpu_kernel_api_yolov5_out*/
    api_.top_addr = bm_mem_get_device_addr(boxs_devmem_);
    api_.detected_num_addr = bm_mem_get_device_addr(detect_num_devmem_);

    // config
    api_.input_num = out_tensor_num_;
    api_.batch_num = 1;
    for (int j = 0; j < out_tensor_num_; ++j) {
        api_.hw_shape[j][0] = net_ptr_->get_output_tensor(j)->shape.dims[2];
        api_.hw_shape[j][1] = net_ptr_->get_output_tensor(j)->shape.dims[3];
    }
    api_.num_classes = net_ptr_->get_output_tensor(0)->shape.dims[1] / anchors.size() - 5;;
    api_.num_boxes = anchors[0].size();
    api_.keep_top_k = 200;
    api_.nms_threshold = MAX(0.1, nmsThreshold_);
    api_.confidence_threshold = MAX(0.1, confThreshold_);
    auto it=api_.bias;
    for (const auto& subvector2 : anchors) {
        for (const auto& subvector1 : subvector2) {
            it = copy(subvector1.begin(), subvector1.end(), it);
        }
    }

    for (int j = 0; j < out_tensor_num_; j++) {
        api_.anchor_scale[j] = net_height_ / net_ptr_->get_output_tensor(j)->shape.dims[2];
    }
    api_.clip_box = 1;
}

int Yolov5::release(){

    //preprocess
    bm_image_free_contiguous_mem(1, &resized_img_);
    bm_image_destroy(resized_img_);

    for(int i=0;i<converto_imgs_.size();++i){
        bm_image_free_contiguous_mem(1, &converto_imgs_[i]);
        bm_image_destroy(converto_imgs_[i]);
    }

    //process
    for(int i=0;i<output_mem_buffer_.size();i++){
        for(int j=0;j<out_tensor_num_;j++){
            bm_free_device(handle_,output_mem_buffer_[i][j]);
        }
    }

    //post
    bm_free_device(handle_, boxs_devmem_);
    bm_free_device(handle_, detect_num_devmem_);

    bm_dev_free(handle_);
}

int Yolov5::preprocess(std::shared_ptr<bm_image> img){

    int idx = get_input_mem_id();
    bm_image* converto_img = &converto_imgs_[idx];
    bm_image* src_img = img.get();
    bm_image* resized_img = &resized_img_;

        //resize image
#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(src_img->width, src_img->height, net_width_, net_height_, &isAlignWidth);
    bmcv_padding_atrr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_sty = 0;
    padding_attr.dst_crop_stx = 0;
    padding_attr.padding_b = 114;
    padding_attr.padding_g = 114;
    padding_attr.padding_r = 114;
    padding_attr.if_memset = 1;
    if (isAlignWidth) {
        padding_attr.dst_crop_h = src_img->height * ratio;
        padding_attr.dst_crop_w = net_width_;

        int ty1 = (int)((net_height_ - padding_attr.dst_crop_h) / 2);
        padding_attr.dst_crop_sty = ty1;
        padding_attr.dst_crop_stx = 0;
    } 
    else {
        padding_attr.dst_crop_h = net_height_;
        padding_attr.dst_crop_w = src_img->width * ratio;

        int tx1 = (int)((net_width_ - padding_attr.dst_crop_w) / 2);
        padding_attr.dst_crop_sty = 0;
        padding_attr.dst_crop_stx = tx1;
    }

    bmcv_rect_t crop_rect{0, 0, src_img->width, src_img->height};
    ret_ = bmcv_image_vpp_convert_padding(handle_, 1, *src_img, resized_img,
                                        &padding_attr, &crop_rect);
#else
        ret_ = bmcv_image_vpp_convert(handle_, 1, *src_img, resized_img);
#endif
    assert(BM_SUCCESS == ret_);
        
    //2. converto
    ret_ = bmcv_image_convert_to(handle_, 1, converto_attr_, resized_img, converto_img);
    CV_Assert(ret_ == 0);

    pre_forward_que_.push_back(&input_mem_buffer_[idx]);
    return 0;
}

void Yolov5::forward_thread_dowork(){
    while(!stop_flag_){
        bm_device_mem_t* in_mem = pre_forward_que_.pop_front();
        bm_device_mem_t* out_mem = output_mem_buffer_[get_output_mem_id()].data();
        auto ret = net_ptr_->forward(in_mem,out_mem);
        assert(ret==0);
        forward_post_que_.push_back(out_mem);
    }
}

void Yolov5::post_thread_dowork(){
    while(!stop_flag_){
        bm_device_mem_t* dev_mem = forward_post_que_.pop_front();
        
        for (int j = 0; j < out_tensor_num_; j++) {
            api_.bottom_addr[j] = bm_mem_get_device_addr(dev_mem[j]);
        }
        tpu_kernel_launch(handle_, func_id, &api_, sizeof(api_));
        bm_thread_sync(handle_);

        bm_memcpy_d2s_partial_offset(handle_, (void*)(&detect_num_sysmem_), detect_num_devmem_, 1 * sizeof(int32_t), 0); // only support batchsize=1
        if(detect_num_sysmem_ > 0){
            bm_memcpy_d2s_partial_offset(handle_, (void*)(boxs_sysmem_.data()), boxs_devmem_, detect_num_sysmem_ * 7 * sizeof(float), 0);  // 25200*7
        }

        std::unique_lock<std::mutex> lock(global_frame_que_mtx_);
        std::shared_ptr<FrameInfoDetect> data_ptr = global_frame_que_.front();
        global_frame_que_.pop_front();
        lock.unlock();
        
        int tx1 = 0, ty1 = 0;
#if USE_ASPECT_RATIO
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(data_ptr->image_ptr->width, data_ptr->image_ptr->height, net_width_, net_height_, &isAlignWidth);
        if (isAlignWidth) {
            ty1 = (int)((net_height_ - (int)(data_ptr->image_ptr->height * ratio)) / 2);   
        } 
        else {
            tx1 = (int)((net_width_ - (int)(data_ptr->image_ptr->width * ratio)) / 2);
        }
#endif
        for (int bid = 0; bid < detect_num_sysmem_; bid++) {
            YoloV5Box temp_bbox;
            float* out_tensor = boxs_sysmem_.data();
            temp_bbox.class_id = *(out_tensor + 7 * bid + 1);
            if (temp_bbox.class_id <= -1) {
                continue;
            }
            temp_bbox.score = *(out_tensor + 7 * bid + 2);
            float centerX = (*(out_tensor + 7 * bid + 3) + 1 - tx1) / ratio - 1;
            float centerY = (*(out_tensor + 7 * bid + 4) + 1 - ty1) / ratio - 1;
            temp_bbox.width = (*(out_tensor + 7 * bid + 5) + 0.5) / ratio;
            temp_bbox.height = (*(out_tensor + 7 * bid + 6) + 0.5) / ratio;

            temp_bbox.x = MAX(int(centerX - temp_bbox.width / 2), 0);
            temp_bbox.y = MAX(int(centerY - temp_bbox.height / 2), 0);
            data_ptr->boxs_vec.push_back(temp_bbox);  // 0
        }
        //push
        out_que_.push_back(data_ptr);
    
    }
}
