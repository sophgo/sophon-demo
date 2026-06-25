#include "unet.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

#define DUMP_FILE 1
#define USE_ASPECT_RATIO 1


UNet::UNet(std::shared_ptr<BMNNContext> context):m_bmContext(context)
{
    std::cout << "UNet ctor .." << std::endl;
}

UNet::~UNet()
{
    std::cout << "UNet dtor .." << std::endl;
    bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
    for(int i = 0;i<max_batch;++i)
    {
        bm_image_destroy(m_resized_imgs[i]);
        bm_image_destroy(m_converto_imgs[i]);
    }
}

int UNet::Init(float out_threshold, int nclasses)
{
    m_outThreshold = out_threshold;
    m_nclasses = nclasses;
    
    m_bmNetwork = m_bmContext->network(0);

    max_batch = m_bmNetwork->maxBatch();
    auto tensor = m_bmNetwork->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];

    output_num = m_bmNetwork->outputTensorNum();
    assert(output_num > 0);
    min_dim = m_bmNetwork->outputTensor(0)->get_shape()->num_dims;

    m_resized_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for(int i=0; i<max_batch; i++)
    {
        auto ret= bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
        if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    }
    bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (tensor->get_dtype() == BM_INT8)
    {
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_converto_imgs.data(), max_batch);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }

    float input_scale = tensor->get_scale();

    input_scale = input_scale * 1.0 / 255.f;
    input_converto_attr.alpha_0 = input_scale;
    input_converto_attr.beta_0 = 0;
    input_converto_attr.alpha_1 = input_scale;
    input_converto_attr.beta_1 = 0;
    input_converto_attr.alpha_2 = input_scale;
    input_converto_attr.beta_2 = 0;

    float output_scale = 1.0 / input_scale;
    output_converto_attr.alpha_0 = output_scale;
    output_converto_attr.beta_0 = 0;
    output_converto_attr.alpha_1 = output_scale;
    output_converto_attr.beta_1 = 0;
    output_converto_attr.alpha_2 = output_scale;
    output_converto_attr.beta_2 = 0;

    return 0;
}

void UNet::enableProfile(TimeStamp * ts)
{
    m_ts = ts;
}

int UNet::batch_size()
{
    return max_batch;
}

int UNet::Segment(const std::vector<bm_image> & images, std::vector<bm_image> & masks)
{
    int ret = 0;
    LOG_TS(m_ts, "unet preprocess");
    ret = pre_process(images);
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "unet preprocess");

    LOG_TS(m_ts, "unet inference");
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "unet inference");

    LOG_TS(m_ts, "unet postprocess");
    ret = post_process(images, masks);
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "unet postprocess");
    return ret;
}

int UNet::pre_process(const std::vector<bm_image> & images)
{
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
    int image_n = images.size();
    int ret = 0;
    bmcv_resize_image resize_attr;
    for(int i = 0;i < image_n; ++ i)
    {
        bm_image image1 = images[i];
        bm_image image_aligned;
        bool need_copy = image1.width & (64-1);
        if (need_copy)
        {
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
        }
        else
        {
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
        if (isAlignWidth) 
        {
            padding_attr.dst_crop_h = images[i].height*ratio;
            padding_attr.dst_crop_w = m_net_w;
            int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
            padding_attr.dst_crop_sty = ty1;
            padding_attr.dst_crop_stx = 0;
        }
        else
        {
            padding_attr.dst_crop_h = m_net_h;
            padding_attr.dst_crop_w = images[i].width*ratio;
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
        if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    
#if DUMP_FILE
        cv::Mat resized_img;
        cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
        std::string fname = cv::format("resized_img_%d.jpg", i);
        cv::imwrite(fname, resized_img);
#endif
        if(need_copy) bm_image_destroy(image_aligned);
    }
    ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, input_converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
    CV_Assert(ret == 0);

    if(image_n != max_batch) image_n = m_bmNetwork->get_nearest_batch(image_n); 
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
    input_tensor->set_device_mem(&input_dev_mem);
    input_tensor->set_shape_by_dim(0, image_n);
    return 0;
}

float UNet::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
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

int UNet::post_process(const std::vector<bm_image> & images, std::vector<bm_image> & masks)
{
    
    std::shared_ptr<BMNNTensor> outputTensor = m_bmNetwork->outputTensor(0);
    bmcv_resize_image bmcv_resize_attr;
    bmcv_resize_t resize_img_attr;
    bmcv_resize_attr.roi_num = 1;
    bmcv_resize_attr.stretch_fit = 0;
    bmcv_resize_attr.padding_b = 0;
    bmcv_resize_attr.padding_g = 0;
    bmcv_resize_attr.padding_r = 0;
    bmcv_resize_attr.interpolation = BMCV_INTER_NEAREST;

    for(int batch_idx = 0;batch_idx < images.size();++batch_idx)
    {
        bm_image result;
        auto & frame = images[batch_idx];
        auto output_shape = m_bmNetwork->outputTensor(0)->get_shape();
        // [batch_size, nc, net_h, net_w]
        int feat_c = outputTensor->get_shape()->dims[1];
        int feat_h = outputTensor->get_shape()->dims[2];
        int feat_w = outputTensor->get_shape()->dims[3];

        int feature_size = feat_w * feat_h;
        uchar * decoded_data = new uchar[feature_size];
    
        float * tensor_data = (float*)outputTensor->get_cpu_data() + batch_idx * feat_c * feature_size;
        // decode: tensor_data->bmimg
        if(feat_c == 1)
        {
            // for nclasses == 1, use sigmoid and outThreshold
            for(int i = 0;i<feature_size;++i)
                decoded_data[i] = sigmoid(tensor_data[i]) > m_outThreshold ? 255 : 0;
        }
        else if (feat_c == 2)
        {
            // for nclasses == 2, convert to 0 or 255
            for(int i = 0;i<feature_size;++i)
            {
                float pro0 = tensor_data[i];
                float pro1 = tensor_data[i + feature_size];
                decoded_data[i] = (pro0 > pro1) ? 0 : 255;
            }
        }
        else
        {
            // for multi-classes, convert to the index of classes
            for(int i = 0;i<feature_size; ++i)
                decoded_data[i] = argmax(tensor_data, i, m_nclasses, feature_size);
        }
        
        bm_image_create(m_bmContext->handle(), feat_h, feat_w,
        FORMAT_GRAY, images[batch_idx].data_type, &result);
        bm_image_copy_host_to_device(result, (void**)&decoded_data);

        bm_image result_resized;
        auto ret= bm_image_create(m_bmContext->handle(), images[batch_idx].height, images[batch_idx].width, FORMAT_GRAY, images[batch_idx].data_type, &result_resized);
        if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
        bm_image_alloc_dev_mem(result_resized);

        resize_img_attr.start_x = 0;
        resize_img_attr.start_y = 0;
        resize_img_attr.in_width = feat_w;
        resize_img_attr.in_height = feat_h;
        resize_img_attr.out_height = images[batch_idx].height;
        resize_img_attr.out_width = images[batch_idx].width;

        bmcv_resize_attr.resize_img_attr = & resize_img_attr;

        ret = bmcv_image_resize(m_bmContext->handle(), 1, &bmcv_resize_attr, &result, &result_resized);
        if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
        masks.push_back(result_resized);
        bm_image_destroy(result);
        delete []decoded_data;
    }
    return 0;
}

float UNet::sigmoid(float x)
{
  return 1.0 / (1 + expf(-x));
}

int UNet::argmax(float* data, int idx, int nclasses, int feature_size)
{
    int ans = -1;
    int max_score = -1;
    for(int i = 0; i< nclasses; ++i)
    {
        float temp = data[idx + i * feature_size];
        if(temp > max_score)
        {
            ans = i;
            max_score = temp;
        } 
    }
    return ans;
}