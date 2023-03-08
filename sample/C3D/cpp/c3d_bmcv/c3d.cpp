//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "c3d.hpp"


template<class ForwardIterator>
size_t argmin(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::min_element(first, last));
}
 
template<class ForwardIterator>
size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}


//public:
C3D::C3D(std::shared_ptr<BMNNContext> context, int step_len, int dev_id):
        m_bmContext(context), m_step(step_len), m_dev_id(dev_id){
    std::cout << "C3D create bm_context" << std::endl;
}

C3D::~C3D(){
    std::cout << "C3D delete bm_context" << std::endl;   
    bm_image_free_contiguous_mem(max_batch * m_clip_len, m_resized_input.data());
    bm_image_free_contiguous_mem(max_batch * m_clip_len, m_croped_input.data());
    bm_image_free_contiguous_mem(max_batch * m_clip_len, m_converto_input.data());
    for(int i = 0; i < max_batch * m_clip_len; i++){
        bm_image_destroy(m_converto_input[i]);
        bm_image_destroy(m_croped_input[i]);
        bm_image_destroy(m_resized_input[i]);
    }
}

void C3D::Init(){
    //1. Get network.
    m_bmNetwork = m_bmContext->network(0);
    //2. Get input.
    m_input_tensor = m_bmNetwork->inputTensor(0);

    //3. Set parameters.
    max_batch = m_bmNetwork->maxBatch();
    m_num_channels = m_input_tensor->get_shape()->dims[1] 
                   * m_input_tensor->get_shape()->dims[2]; //(3*16,112,112)
    m_clip_len = m_input_tensor->get_shape()->dims[2];
    m_net_h = m_input_tensor->get_shape()->dims[3];
    m_net_w = m_input_tensor->get_shape()->dims[4];

    //4. Initialize bm_images.
    m_decoded_input.resize(max_batch * m_clip_len);
    m_resized_input.resize(max_batch * m_clip_len);
    m_croped_input.resize(max_batch * m_clip_len);
    m_converto_input.resize(max_batch * m_clip_len);
    int aligned_171_w = FFALIGN(171, 64);
    int resize_strides[3] = {aligned_171_w, aligned_171_w, aligned_171_w};
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    
    auto img_dtype = DATA_TYPE_EXT_FLOAT32;
    // auto img_dtype = DATA_TYPE_EXT_1N_BYTE;//for debug
    
    if(m_input_tensor->get_dtype() == BM_INT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }

#if 0
    for(int i = 0; i < max_batch * m_clip_len; i++){
        ret = bm_image_create(m_bmContext->handle(),
                                    128, 171,               //h, w. But in opencv is w, h.(171,128)
                                    FORMAT_BGR_PLANAR, 
                                    DATA_TYPE_EXT_1N_BYTE, 
                                    &m_resized_input[i], 
                                    resize_strides);
        assert(BM_SUCCESS == ret);
        ret = bm_image_create(m_bmContext->handle(),
                                    m_net_h, m_net_w,
                                    FORMAT_BGR_PLANAR, 
                                    DATA_TYPE_EXT_1N_BYTE, 
                                    &m_croped_input[i], 
                                    strides);
        assert(BM_SUCCESS == ret);
    }
    ret = bm_image_alloc_contiguous_mem(max_batch * m_clip_len, m_resized_input.data());
    assert(BM_SUCCESS == ret);
    ret = bm_image_alloc_contiguous_mem(max_batch * m_clip_len, m_croped_input.data());
    assert(BM_SUCCESS == ret);
#else

    ret = bm_image_create_batch(m_bmContext->handle(), 
                                    128, 171, 
                                    FORMAT_BGR_PLANAR, 
                                    DATA_TYPE_EXT_1N_BYTE, 
                                    m_resized_input.data(),
                                    max_batch * m_clip_len,
                                    resize_strides);
    assert(BM_SUCCESS == ret);
    ret = bm_image_create_batch(m_bmContext->handle(), 
                                    m_net_h, m_net_w, 
                                    FORMAT_BGR_PLANAR, 
                                    DATA_TYPE_EXT_1N_BYTE, 
                                    m_croped_input.data(),
                                    max_batch * m_clip_len,
                                    strides);
    assert(BM_SUCCESS == ret);
#endif       

    ret = bm_image_create_batch(m_bmContext->handle(), 
                                m_net_h, m_net_w, 
                                FORMAT_BGR_PLANAR, 
                                img_dtype, 
                                m_converto_input.data(),
                                max_batch * m_clip_len);
    assert(BM_SUCCESS == ret);
    
    // 5.Set vpp converto params.
    float input_scale = m_input_tensor->get_scale();
    m_converto_attr.alpha_0 = input_scale;
    m_converto_attr.beta_0 = -104.0; //ImageNet channel B mean
    m_converto_attr.alpha_1 = input_scale;
    m_converto_attr.beta_1 = -117.0; //ImageNet channel G mean
    m_converto_attr.alpha_2 = input_scale;
    m_converto_attr.beta_2 = -123.0; //ImageNet channel R mean
}

int C3D::batch_size(){
    return max_batch;
}

int C3D::detect(const std::vector<std::string> &batch_videos, std::vector<int> &preds){
    int ret = 0;
    
    //0. Decode video and get bm_image list
    m_ts->save("C3D decode_time");
    for(int i = 0; i < max_batch; i++){
        if(i < batch_videos.size())
            decode_video(batch_videos[i], m_decoded_input, i);
        else
            decode_video(batch_videos[0], m_decoded_input, i);
    }
    m_ts->save("C3D decode_time");
    //1. Preprocess, convert raw images to format which fits C3D network.
    ret = pre_process(m_decoded_input);

    CV_Assert(ret == 0);
    //2. Run C3D inference.
    m_ts->save("C3D inference");
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    m_ts->save("C3D inference");
    
    std::shared_ptr<BMNNTensor> outputTensor = m_bmNetwork->outputTensor(0);
    auto output_shape = outputTensor->get_shape();
    auto output_dims = output_shape->num_dims;
#if DEBUG
    std::cout<<"Output dims:"<<output_dims<<std::endl;
    std::cout<<"Output shape infos: "<<output_shape->dims[0]<<" "
             <<output_shape->dims[1]<<std::endl;
#endif
    assert(m_bmNetwork->outputTensorNum() == 1); 
    int class_num = output_shape->dims[1];
    auto output_data = outputTensor->get_cpu_data();
    m_ts->save("C3D postprocess_time");
    for(int i = 0; i < batch_videos.size(); i++){
        auto max_index = argmax(output_data + i * class_num, 
                                output_data + (i + 1) * class_num);
        preds.push_back(max_index);

        /*CODE-FOR-DEBUG*/
        // for(int j=i*class_num;j<(i+1)*class_num;j++){
        //     std::cout<<*(output_data+j)<<" ";
        // }
        // std::cout<<std::endl;
        
    }

    m_ts->save("C3D postprocess_time");
    return 0;
}

void C3D::enableProfile(TimeStamp *ts){
    m_ts = ts;
}

void C3D::decode_video(const std::string video_path, std::vector<bm_image> &decoded_frames, int video_id){
    int channel_base = video_id * m_clip_len;
    auto handle = m_bmContext->handle();
    bmcv_copy_to_atrr_t copy_to_attr={0, 0, 0, 0, 0, 0};

    /*FFmepg do not support c3d dataset, use opencv first.*/
    cv::VideoCapture cap(video_path, cv::CAP_ANY, m_dev_id);
    if(!cap.isOpened()){
        std::cout << "open video stream failed!" << std::endl;
        exit(1);
    }
    int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
#if DEBUG
    std::cout << "Frame num: " << frame_num << std::endl;
    std::cout << "resolution of input stream: " << h << ", " << w << std::endl;
#endif
    // VideoDecFFM decoder;
    // decoder.openDec(&handle, video_path.c_str());

    if(video_id == 0){
        ret = bm_image_create_batch(m_bmContext->handle(), 
                                    h, w, 
                                    FORMAT_BGR_PACKED, 
                                    DATA_TYPE_EXT_1N_BYTE, 
                                    m_decoded_input.data(),
                                    max_batch * m_clip_len);
    }

    /* Convert the input frame to the format of the network. */
    int frame_count = 0;

    for(int i = 0; i < frame_num; i++){ //if i < step * 16
    // for(int i = 0; i < m_step * m_clip_len; i++){ //if i < step * 16
        cv::Mat img(h, w, CV_8UC3, cv::SophonDevice(m_dev_id));
        cap.read(img);
        if(img.empty()) continue;
        
        // bm_image *frame = decoder.grab();
        // if(!frame) break;

        if(frame_count >= m_clip_len)
            break;
        if(i % m_step != 0)
            continue;

        bm_image tmp;
        assert(0 == cv::bmcv::toBMI((cv::Mat&)img, &tmp, true));
        ret = bmcv_image_vpp_convert(m_bmContext->handle(), 
                                        1, 
                                        tmp, 
                                        &decoded_frames[channel_base + frame_count]);
        assert(BM_SUCCESS == ret);
        bm_image_destroy(tmp);
        frame_count++;
    }
    while(frame_count < m_clip_len){
        // ret = bm_image_create(m_bmContext->handle(),
        //                             h, w,
        //                             decoded_frames[0].image_format, 
        //                             decoded_frames[0].data_type, 
        //                             &decoded_frames[channel_base + frame_count]);
        // assert(BM_SUCCESS == ret);
        bmcv_image_copy_to(handle, 
                            copy_to_attr, 
                            decoded_frames[channel_base + frame_count - 1], 
                            decoded_frames[channel_base + frame_count]);
        frame_count++;
    }
}


void print_array(float array[], int i0, int i1, int i2, int i3){
    for(int i = 0; i < 112; i++){
        std::cout<<array[i3*112 + i2 * 112 * 112 + 
                        i1 * 16 * 112 * 112 + i0 * 3 * 16 * 112 * 112 + i]<<" ";
    }
    std::cout<<std::endl;
}
int C3D::pre_process(const std::vector<bm_image> &decoded_frames){
    //Safety check.
    //1. Preprocess input videos in host memory.
    m_ts->save("C3D preprocess_time");
    auto handle = m_bmContext->handle();
    for(int i = 0; i < decoded_frames.size(); i++){ 
        //64-bytes align will enhance data process rate.    
        bm_image image1 = decoded_frames[i]; 
        bm_image image_aligned;
        bool need_copy = image1.width & (64-1);
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
        } else {
            image_aligned = image1;
        }
    
    #if 1
        //Resize, then crop. 
        ret = bmcv_image_vpp_convert(m_bmContext->handle(), 
                                            1, 
                                            image_aligned, 
                                            &m_resized_input[i]);
        assert(BM_SUCCESS == ret);
        bmcv_rect_t crop_rect = {30, 8, 112, 112}; //here is w, h
        ret = bmcv_image_crop(m_bmContext->handle(),
                                    1,
                                    &crop_rect,
                                    m_resized_input[i],
                                    &m_croped_input[i]
                                    );
    #else
        //Crop, then Resize.
        bmcv_rect_t crop_rect = {74, 56, 171, 128}; //here is w, h
        ret = bmcv_image_vpp_convert(m_bmContext->handle(),
                                    1,
                                    image_aligned,
                                    &m_croped_input[channel_base + frame_count],
                                    &crop_rect
                                    );
    #endif
        /*Do not forget destroy bm_image ! ! !*/
        bm_image_destroy(image1); //decoded_frames
        if(need_copy) bm_image_destroy(image_aligned);
        /**************************************/
    }

    //Apply normalization, aka converto.    
    ret = bmcv_image_convert_to(m_bmContext->handle(), 
                                        max_batch * m_clip_len, 
                                        m_converto_attr, 
                                        m_croped_input.data(), 
                                        m_converto_input.data());
    assert(BM_SUCCESS == ret);
    m_ts->save("C3D preprocess_time");
    
    /*CODE-FOR-DEBUG*/
    // for(int i = 0; i < max_batch * m_clip_len; i++)
    // {
    //     std::string img_file = "debug/croped/test_croped_b" + std::to_string(max_batch) + "_" + std::to_string(i) +".jpg";
    //     std::cout<<"saving: "<<img_file<<std::endl;
    //     void* jpeg_data = NULL;
    //     size_t out_size = 0;
    //     bm_image test;
    //     bm_image_create(m_bmContext->handle(), 
    //                         112, 112, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, &test);
    //     bmcv_image_storage_convert(m_bmContext->handle(), 1, &m_croped_input[i], &test);    
    //     ret = bmcv_image_jpeg_enc(m_bmContext->handle(), 
    //                                             1, 
    //                                             &test, 
    //                                             &jpeg_data, 
    //                                             &out_size);   
    //     if (ret == BM_SUCCESS) {
    //     FILE *fp = fopen(img_file.c_str(), "wb");
    //     fwrite(jpeg_data, out_size, 1, fp);
    //     fclose(fp);
    //     }
    //     free(jpeg_data);
    // }
    // for(int i = 0; i < max_batch * m_clip_len; i++)
    // {
    //     std::string img_file = "debug/converted/test_converted_b" + std::to_string(max_batch) + "_" + std::to_string(i) +".jpg";
    //     std::cout<<"saving: "<<img_file<<std::endl;
    //     void* jpeg_data = NULL;
    //     size_t out_size = 0;
    //     bm_image test;
    //     bm_image_create(m_bmContext->handle(), 
    //                         112, 112, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, &test);
    //     bmcv_image_storage_convert(m_bmContext->handle(), 1, &m_converto_input[i], &test);    
    //     ret = bmcv_image_jpeg_enc(m_bmContext->handle(), 
    //                                             1, 
    //                                             &test, 
    //                                             &jpeg_data, 
    //                                             &out_size);   
    //     if (ret == BM_SUCCESS) {
    //     FILE *fp = fopen(img_file.c_str(), "wb");
    //     fwrite(jpeg_data, out_size, 1, fp);
    //     fclose(fp);
    //     }
    //     free(jpeg_data);
    // }
    // exit(1);
    
    m_ts->save("C3D realloc_time");
    //2. Attach to input tensor.
    bm_device_mem_t bgrbgrbgr_dev_mem;
    bm_image_get_contiguous_device_mem(max_batch * m_clip_len, 
                                        m_converto_input.data(),
                                        &bgrbgrbgr_dev_mem);
    bm_device_mem_t bbbgggrrr_dev_mem;
    bm_malloc_device_byte(m_bmContext->handle(), &bbbgggrrr_dev_mem, bgrbgrbgr_dev_mem.size);
    int planar_size = bgrbgrbgr_dev_mem.size / (3 * max_batch * m_clip_len);
    
    //realloc device memory, from b g r / b g r / b g r to b b b / g g g / r r r.
    for(int i = 0; i < max_batch; i++){
        for(int j = 0; j < m_clip_len; j++){
            for(int k = 0; k < 3; k++){
                bm_memcpy_d2d_byte(m_bmContext->handle(), 
                                bbbgggrrr_dev_mem,
                                (i * 3 * m_clip_len + k * m_clip_len + j) * planar_size,
                                bgrbgrbgr_dev_mem,
                                (i * 3 * m_clip_len + j * 3 + k) * planar_size,
                                planar_size); 
            }            
        }
    }

    /*CODE-FOR-DEBUG*/
    // m_input_tensor->set_device_mem(&bgrbgrbgr_dev_mem);return 0;
    // std::cout << "size:" << bgrbgrbgr_dev_mem.size << std::endl;
    // print_array(m_input_tensor->get_cpu_data(), 0, 1, 0 ,0);
    // print_array(m_input_tensor->get_cpu_data(), 0, 2, 0 ,0);
    // std::cout << "==============" << std::endl;
    // std::cout<<bgrbgrbgr_dev_mem.u.device.device_addr<<std::endl;
    // std::cout<<bbbgggrrr_dev_mem.u.device.device_addr<<std::endl;
    // std::cout<<m_input_tensor->get_dtype()<<std::endl;
    // float* input_data = new float[max_batch*m_clip_len*3*112*112];
    // ret = bm_memcpy_d2s(m_bmContext->handle(),input_data, bbbgggrrr_dev_mem);
    // assert(BM_SUCCESS == ret);// for(int i=0;i<m_clip_len*3*112*112;i++){
    //     std::cout<<input_data[i]<<" ";
    // }
    // std::cout<<std::endl;

    m_input_tensor->set_device_mem(&bbbgggrrr_dev_mem);
    bm_free_device(m_bmContext->handle(), bbbgggrrr_dev_mem);

    m_ts->save("C3D realloc_time");
    return 0;
}
