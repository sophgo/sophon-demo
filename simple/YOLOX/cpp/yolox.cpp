#include "yolox.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

int readAllPictures(const char *dir_name,vector<string>& v)
{
    DIR *dirp = NULL;
    struct dirent *dp=NULL;
    dirp = opendir(dir_name);
    int re_value = 0;
    if(dirp == NULL){
        printf("Can not find path: %s\n",dir_name);
        return re_value;
    }
    while ((dp = readdir(dirp)) != NULL) {
        if(dp->d_type != 4)
        {
            char dir_temp[512]={};
            if(dir_name[strlen(dir_name)-1] == '/')
                snprintf(dir_temp,512,"%s%s",dir_name,dp->d_name);
            else
                snprintf(dir_temp,512,"%s/%s",dir_name,dp->d_name);
            if(strcmp(&dir_temp[strlen(dir_temp)-3],"png")==0)
            {
                v.push_back(std::string(dir_temp));
                re_value++;
            }else if(strcmp(&dir_temp[strlen(dir_temp)-3],"jpg")==0){
                v.push_back(std::string(dir_temp));
                re_value++;
            }else if(strcmp(&dir_temp[strlen(dir_temp)-3],"bmp")==0){
                v.push_back(std::string(dir_temp));
                re_value++;
            }
        }
    }
    (void) closedir(dirp);
    return re_value;
}

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

float box_intersection_FM(ObjRect a, ObjRect b)
{
	float w = overlap_FM(a.left, a.width, b.left, b.width);
	float h = overlap_FM(a.top, a.height, b.top, b.height);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

float box_union_FM(ObjRect a, ObjRect b)
{
	float i = box_intersection_FM(a, b);
	float u = a.width*a.height + b.width*b.height - i;
	return u;
}

float box_iou_FM(ObjRect a, ObjRect b)
{
	return box_intersection_FM(a, b) / box_union_FM(a, b);
}

static bool sort_ObjRect(ObjRect a, ObjRect b)
{
    return a.score > b.score;
}

static void nms_sorted_bboxes(const std::vector<ObjRect>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = objects.size();

    for (int i = 0; i < n; i++)    {
        const ObjRect& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const ObjRect& b = objects[picked[j]];

            float iou = box_iou_FM(a, b);
            if (iou > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

YoloXDete::YoloXDete(std::string bmodel_path, int device_id)
{
    engie = new sail::Engine(bmodel_path,device_id, sail::IOMode::SYSO);
    if(!engie){
        printf("Create Engine filed!\n");
        exit(1) ;
    }
    handle = engie->get_handle();
    graph_name = engie->get_graph_names().at(0);
    input_tensor_name = engie->get_input_names(graph_name).at(0);
    output_tensor_name = engie->get_output_names(graph_name).at(0);

    input_scale = engie->get_input_scale(graph_name,input_tensor_name);
    bm_data_type_t input_dtype = engie->get_input_dtype(graph_name,input_tensor_name);
    switch (input_dtype)
    {
    case BM_FLOAT32:
        img_dtype = DATA_TYPE_EXT_FLOAT32;
        break;
    default:
        img_dtype = DATA_TYPE_EXT_1N_BYTE;
        break;
    }
    std::vector<int> input_shape = engie->get_input_shape(graph_name,input_tensor_name);
    batch_size = input_shape.at(0);
    net_input_w = input_shape.at(2);
    net_input_h = input_shape.at(3);

    printf("######################################\n");
    printf("%d,%d,%d,%d\n",input_shape.at(0),input_shape.at(1),input_shape.at(2),input_shape.at(3));
    printf("######################################\n");

    bm_data_type_t output_dtype = engie->get_output_dtype(graph_name,output_tensor_name);
    vector<int> output_shape = engie->get_output_shape(graph_name,output_tensor_name);

    input_tensor = new sail::Tensor(handle,input_shape,input_dtype,false,true);
    output_tensor = new sail::Tensor(handle,output_shape,output_dtype,true,true);

    input_tensormap.insert(std::pair<std::string,sail::Tensor*>(input_tensor_name,input_tensor));
    output_tensormap.insert(std::pair<std::string,sail::Tensor*>(output_tensor_name,output_tensor));
}

YoloXDete::~YoloXDete()
{
    if(engie){
        delete engie;
        engie = NULL;
    }
    if(input_tensor){
        delete input_tensor;
        input_tensor = NULL;
    }
    if(output_tensor){
        delete output_tensor;
        output_tensor = NULL;
    }
}

int YoloXDete::get_batchsize()
{
    return batch_size;
}

int YoloXDete::get_net_input_w()
{
    return net_input_w;
}
int YoloXDete::get_net_input_h()
{
    return net_input_h;
}

float YoloXDete::get_input_scale()
{
    return input_scale;
}

sail::Handle YoloXDete::get_handle()
{
    return handle;
}

bm_image_data_format_ext YoloXDete::get_img_dtype()
{
    return img_dtype;
}

void YoloXDete::Inference()
{
    double start_time = sail::get_current_time_us();
    engie->process(graph_name,input_tensormap,output_tensormap);
    double end_time = sail::get_current_time_us();
    printf("Inference time use:%.2f ms, Batch size:%d, avg fps:%.1f\n",
        (end_time-start_time)/1000, batch_size,batch_size*1000*1000/(end_time-start_time));
}

void* YoloXDete::get_output_data_prt()
{
    return output_tensor->sys_data();
}

vector<int> YoloXDete::get_output_shape()
{
    return engie->get_output_shape(graph_name,output_tensor_name);
}


VideoProcess::VideoProcess(std::string video_name,
        sail::Handle& handle, 
        int batch_size_, 
        int device_id,
        int resize_width,
        int resize_height,
        bm_image_data_format_ext output_dtype):
        batch_size(batch_size_),
        output_width(resize_width),
        output_height(resize_height),
        frame_skip(1){
    decoder = new sail::Decoder(video_name,true,device_id);
    if (!decoder){
        printf("Video[%s] read failed!\n",video_name.c_str());
        exit(1) ;
    }
    if(!decoder->is_opened()){
        printf("Video[%s] read failed!\n",video_name.c_str());
        exit(1) ;
    }
    vector<int> video_shape = decoder->get_frame_shape();
    image_height = video_shape[2];
    image_width = video_shape[3];
    printf("Video Width[%d], Height[%d]!\n",image_width,image_height);
    if(batch_size == 1){
        input_image = sail::BMImage(handle,image_height,image_width,FORMAT_BGR_PLANAR,DATA_TYPE_EXT_1N_BYTE);
        resize_image = sail::BMImage(handle,resize_height,resize_width,FORMAT_BGR_PLANAR,DATA_TYPE_EXT_1N_BYTE);
        output_image = sail::BMImage(handle,resize_height,resize_width,FORMAT_BGR_PLANAR,output_dtype);
    }else if(batch_size == 4){
        resize_image_array = sail::BMImageArray<4>(handle,resize_height,resize_width,FORMAT_BGR_PLANAR,DATA_TYPE_EXT_1N_BYTE);
        output_image_array = sail::BMImageArray<4>(handle,resize_height,resize_width,FORMAT_BGR_PLANAR,output_dtype);
    }else{
        printf("Error Batch Size: %d\n",batch_size);
        exit(1) ;
    }
    calc_scale_padatt();
}

VideoProcess::~VideoProcess()
{
    if (decoder){
        delete decoder;
        decoder= NULL;
    }
}

void VideoProcess::calc_scale_padatt()
{
    scale_w = float(output_width) / image_width;
    scale_h = float(output_height) / image_height;

    int pad_w = output_width;
    int pad_h = output_height;

    scale_min = scale_h;

    if (scale_w < scale_h){
        pad_h = image_height*scale_w;
        scale_min = scale_w;
    }else{
        pad_w = image_width*scale_h;
    }
    paddingatt.set_stx(0);
    paddingatt.set_sty(0);
    paddingatt.set_w(pad_w);
    paddingatt.set_h(pad_h);
    paddingatt.set_r(114);
    paddingatt.set_g(114);
    paddingatt.set_b(114);
}

vector<float> VideoProcess::get_resize_scalemin(){
    vector<float> min_scale(batch_size,scale_min);
    return std::move(min_scale);
}

int VideoProcess::getBMImage(sail::Handle& handle)
{
    int ret_value = 0;
    for (size_t j = 0; j < frame_skip; j++)     {       //skip frame_skip-1 frames
        decoder->read(handle,input_image);
        if(ret_value != 0){
            break;
        }
    }
    return ret_value;
}

int VideoProcess::getBMImageArray(sail::Handle& handle)
{
    int ret_value = 0;
    for(int i=0;i<batch_size;++i){
        for (size_t j = 0; j < frame_skip; j++)     {       //skip frame_skip-1 frames
            ret_value = decoder->read_(handle,input_image_array[i]);
            if(ret_value != 0){
                break;
            }
        }
    }
    return ret_value;
}

int VideoProcess::process_padding_BMImage(sail::Bmcv& bmcv)
{
    return bmcv.vpp_crop_and_resize_padding(
        input_image,
        resize_image,
        0,0,image_width,image_height,
        output_width,output_height,
        paddingatt);
}

int VideoProcess::process_padding_BMImageArray(sail::Bmcv& bmcv)
{
    return bmcv.vpp_crop_and_resize_padding(
        input_image_array,
        resize_image_array,
        0,0,image_width,image_height,
        output_width,output_height,
        paddingatt);
}

int VideoProcess::getTensor(sail::Handle& handle,
    sail::Bmcv& bmcv,
    sail::Tensor& output_tensor,
    const std::tuple<std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>> &alpha_beta)
{
    int ret_value = 0;
    if (batch_size == 1){
        ret_value = getBMImage(handle);
        if(ret_value != 0){
            printf("getBMImage Fialed!\n");
            return ret_value;
        }
        ret_value = process_padding_BMImage(bmcv);
        if(ret_value != 0){
            printf("process_padding_BMImage Fialed!\n");
            return ret_value;
        }
        ret_value = bmcv.convert_to(resize_image, output_image, alpha_beta);
        if(ret_value != 0){
            printf("convert_to Fialed!\n");
            return ret_value;
        }
        bmcv.bm_image_to_tensor(output_image,output_tensor);
    }else if(batch_size == 4){
        ret_value = getBMImageArray(handle);
        if(ret_value != 0){
            printf("getBMImage Fialed!\n");
            return ret_value;
        }
        ret_value = process_padding_BMImageArray(bmcv);
        if(ret_value != 0){
            printf("process_padding_BMImageArray Fialed!\n");
            return ret_value;
        }
        ret_value = bmcv.convert_to(resize_image_array, output_image_array, alpha_beta);
        if(ret_value != 0){
            printf("convert_to Fialed!\n");
            return ret_value;
        }
        bmcv.bm_image_to_tensor(output_image_array,output_tensor);
    }else{
        return 1;
    }
    return 0;
}

PictureProcess::PictureProcess(std::string pic_path,
        sail::Handle& handle, 
        int batch_size_, 
        int device_id_,
        int resize_width,
        int resize_height,
        bm_image_data_format_ext output_dtype):
        batch_size(batch_size_),
        output_width(resize_width),
        output_height(resize_height),
        current_idx(0),
        device_id(device_id_) {
    imagename_list.clear();
    int pic_num = readAllPictures(pic_path.c_str(),imagename_list);
    if(pic_num <= 0){
        printf("Can not find any picture in path: %s\n",pic_path.c_str());
        exit(1);
    }
    if(pic_num % batch_size != 0){
        int add_count = batch_size - (pic_num%batch_size);
        for (size_t i = 0; i < add_count; i++)    {
            imagename_list.push_back(imagename_list[0]);
        }
    }
    if(batch_size == 1){
        output_image = sail::BMImage(handle,resize_height,resize_width,FORMAT_BGR_PLANAR,output_dtype);
    }else if(batch_size == 4){
        resize_image_array = sail::BMImageArray<4>(handle,resize_height,resize_width,FORMAT_BGR_PLANAR,DATA_TYPE_EXT_1N_BYTE);
        output_image_array = sail::BMImageArray<4>(handle,resize_height,resize_width,FORMAT_BGR_PLANAR,output_dtype);
    }else{
        printf("Error Batch Size: %d\n",batch_size);
        exit(1) ;
    }
}

PictureProcess::~PictureProcess()
{

}

vector<float> PictureProcess::get_resize_scalemin()
{
    return scale_min_v;
}

vector<string> PictureProcess::get_curr_imagename()
{
    return input_name_list;
}


void PictureProcess::calc_scale_padatt(int iput_w, int input_h, int output_w, int output_h)
{
    float scale_w = float(output_w) / iput_w;
    float scale_h = float(output_h) / input_h;

    int pad_w = output_w;
    int pad_h = output_h;

    scale_min = scale_h;

    if (scale_w < scale_h){
        pad_h = input_h*scale_w;
        scale_min = scale_w;
    }else{
        pad_w = iput_w*scale_h;
    }
    paddingatt.set_stx(0);
    paddingatt.set_sty(0);
    paddingatt.set_w(pad_w);
    paddingatt.set_h(pad_h);
    paddingatt.set_r(114);
    paddingatt.set_g(114);
    paddingatt.set_b(114);

    scale_min_v.push_back(scale_min);
}

int PictureProcess::getBMImage(sail::Handle& handle, sail::Bmcv& bmcv)
{
    if (current_idx >= imagename_list.size()){
        printf("Read the end!\n");
        return 1;
    }
    sail::Decoder decoder(imagename_list.at(current_idx),true, device_id);
    sail::BMImage ost_image = decoder.read(handle);
    input_image = bmcv.convert_format(ost_image);
    image_width = input_image.width();
    image_height = input_image.height();

    input_name_list.push_back(imagename_list.at(current_idx));
    calc_scale_padatt(image_width, image_height, output_width, output_height);

    resize_image = bmcv.crop_and_resize_padding(
        input_image,
        0,0,image_width,image_height,
        output_width,output_height,
        paddingatt);

    current_idx++;
    return 0;
}

int PictureProcess::getBMImageArray(sail::Handle& handle,sail::Bmcv& bmcv)
{
    if (current_idx >= imagename_list.size()){
        printf("Read the end!\n");
        return 1;
    }
    for (int i=0;i<batch_size;++i){
        sail::Decoder decoder(imagename_list.at(current_idx),true, device_id);
        sail::BMImage ost_image = decoder.read(handle);
        BMImage input_image_temp = bmcv.convert_format(ost_image);
        image_width = input_image_temp.width();
        image_height = input_image_temp.height();
        input_name_list.push_back(imagename_list.at(current_idx));
        calc_scale_padatt(image_width, image_height, output_width, output_height);
        
        current_idx++;
        BMImage resize_image_temp = bmcv.crop_and_resize_padding(
            input_image_temp,
            0,0,image_width,image_height,
            output_width,output_height,
            paddingatt);
        input_image_array.push_back(std::move(input_image_temp));
        int ret_value = resize_image_array.copy_from(i,resize_image_temp);
        if(ret_value != 0){
            return ret_value;
        }
    }
    return 0;
}

int PictureProcess::getTensor(sail::Handle& handle,sail::Bmcv& bmcv,
        sail::Tensor& output_tensor,
        const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>> &alpha_beta){
    input_name_list.clear();
    scale_min_v.clear();
    int ret_value = 0;
    if (batch_size == 1){
        ret_value = getBMImage(handle,bmcv);
        if(ret_value != 0){
            return ret_value;
        }
        ret_value = bmcv.convert_to(resize_image, output_image, alpha_beta);
        if(ret_value != 0){
            printf("convert_to Fialed!\n");
            return ret_value;
        }
        bmcv.bm_image_to_tensor(output_image,output_tensor);
    }else if(batch_size == 4){
        input_image_array.clear();
        ret_value = getBMImageArray(handle,bmcv);
        if(ret_value != 0){
            return ret_value;
        }
        ret_value = bmcv.convert_to(resize_image_array, output_image_array, alpha_beta);
        if(ret_value != 0){
            printf("convert_to Fialed!\n");
            return ret_value;
        }
        bmcv.bm_image_to_tensor(output_image_array,output_tensor);
    }else{
        return 1;
    }
    return ret_value;
}


YoloX_PostForward::YoloX_PostForward(int net_w, int net_h, std::vector<int> strides):network_width(net_w),network_height(net_h)
{
  outlen_diml = 0;
  for (int i=0;i<strides.size();++i)  {
    int layer_w = net_w/strides[i];
    int layer_h = net_h/strides[i];
    outlen_diml += layer_h*layer_w;
  }
  grids_x_ = new int[outlen_diml];
  grids_y_ = new int[outlen_diml];
  expanded_strides_ = new int[outlen_diml];

  int channel_len = 0;
  for (int i=0;i<strides.size();++i)  {
    int layer_w = net_w/strides[i];
    int layer_h = net_h/strides[i];
    for (int m = 0; m < layer_h; ++m)   {
      for (int n = 0; n < layer_w; ++n)    {
          grids_x_[channel_len+m*layer_w+n] = n;
          grids_y_[channel_len+m*layer_w+n] = m;
          expanded_strides_[channel_len+m*layer_w+n] = strides[i];
      }
    }
    channel_len += layer_w * layer_h;
  }
}

YoloX_PostForward::~YoloX_PostForward()
{
  delete grids_x_;
  grids_x_ = NULL;
  delete grids_y_;
  grids_y_ = NULL;
  delete expanded_strides_;
  expanded_strides_ = NULL;
}

void YoloX_PostForward::process(float* data_ptr,std::vector<int> output_shape, std::vector<std::pair<int,int>> ost_size, 
  float threshold, float nms_threshold, std::vector<std::vector<ObjRect>> &detections)
{
  int size_one_batch = output_shape[1]*output_shape[2];
  int channels_resu_ = output_shape[2];
  int classes_ = channels_resu_ - 5;

  for (int batch_idx=0; batch_idx<ost_size.size();batch_idx++){
    int batch_start_ptr = size_one_batch * batch_idx;
    std::vector<ObjRect> dect_temp;
    dect_temp.clear();
    float scale_x = (float)ost_size[batch_idx].first/network_width;
    float scale_y = (float)ost_size[batch_idx].second/network_height;
    for (size_t i = 0; i < outlen_diml; i++)    {
        int ptr_start=i*channels_resu_;
        float box_objectness = data_ptr[batch_start_ptr + ptr_start+4];
        if(data_ptr[batch_start_ptr + ptr_start+4] >= threshold){
            float center_x = (data_ptr[batch_start_ptr +ptr_start+0] + grids_x_[i]) * expanded_strides_[i];
            float center_y = (data_ptr[batch_start_ptr +ptr_start+1] + grids_y_[i]) * expanded_strides_[i];
            float w_temp = exp(data_ptr[batch_start_ptr +ptr_start+2])*expanded_strides_[i];
            float h_temp = exp(data_ptr[batch_start_ptr +ptr_start+3])*expanded_strides_[i];
            float score = data_ptr[batch_start_ptr +ptr_start+4];
            center_x *= scale_x;
            center_y *= scale_y;
            w_temp *= scale_x;
            h_temp *= scale_y;
            float left = center_x - w_temp/2;
            float top = center_y - h_temp/2;
            float right = center_x + w_temp/2;
            float bottom = center_y + h_temp/2;

            // printf("[%.2f,%.2f,%.2f,%.2f]::::%f,%f\n",center_x,center_y,w_temp,h_temp,scale_x,scale_y);

            for (int class_idx = 0; class_idx < classes_; class_idx++)       {
                float box_cls_score = data_ptr[batch_start_ptr +ptr_start + 5 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if (box_prob > threshold)         {
                    ObjRect obj_temp;
                    obj_temp.width = w_temp;
                    obj_temp.height = h_temp;
                    obj_temp.left = left;
                    obj_temp.top = top;
                    obj_temp.right = right;
                    obj_temp.bottom = bottom;
                    obj_temp.score = box_prob;
                    obj_temp.class_id = class_idx;
                    dect_temp.push_back(obj_temp);
                }
            }
        }
    }

    std::sort(dect_temp.begin(),dect_temp.end(),sort_ObjRect);

    std::vector<ObjRect> dect_temp_batch;
    std::vector<int> picked;
    dect_temp_batch.clear();
    nms_sorted_bboxes(dect_temp, picked, nms_threshold);

    for (size_t i = 0; i < picked.size(); i++)    {
        dect_temp_batch.push_back(dect_temp[picked[i]]);
    }
    
    detections.push_back(dect_temp_batch);
  }
}

void YoloX_PostForward::process(float* data_ptr,std::vector<int> output_shape, std::vector<float> resize_scale, 
  float threshold, float nms_threshold, std::vector<std::vector<ObjRect>> &detections)
{
  int size_one_batch = output_shape[1]*output_shape[2];
  int channels_resu_ = output_shape[2];
  int classes_ = channels_resu_ - 5;


  for (int batch_idx=0; batch_idx<resize_scale.size();batch_idx++){
    int batch_start_ptr = size_one_batch * batch_idx;
    std::vector<ObjRect> dect_temp;
    dect_temp.clear();
    float scale_x = 1.0/resize_scale[batch_idx];
    float scale_y = 1.0/resize_scale[batch_idx];
    
    // float scale_x = (float)network_width/resize_scale[batch_idx];
    // float scale_y = (float)network_height/resize_scale[batch_idx];

    for (size_t i = 0; i < outlen_diml; i++)    {
        int ptr_start=i*channels_resu_;
        float box_objectness = data_ptr[batch_start_ptr + ptr_start+4];
        if(data_ptr[batch_start_ptr + ptr_start+4] >= threshold){
            float center_x = (data_ptr[batch_start_ptr +ptr_start+0] + grids_x_[i]) * expanded_strides_[i];
            float center_y = (data_ptr[batch_start_ptr +ptr_start+1] + grids_y_[i]) * expanded_strides_[i];
            float w_temp = exp(data_ptr[batch_start_ptr +ptr_start+2])*expanded_strides_[i];
            float h_temp = exp(data_ptr[batch_start_ptr +ptr_start+3])*expanded_strides_[i];
            float score = data_ptr[batch_start_ptr +ptr_start+4];
            center_x *= scale_x;
            center_y *= scale_y;
            w_temp *= scale_x;
            h_temp *= scale_y;
            float left = center_x - w_temp/2;
            float top = center_y - h_temp/2;
            float right = center_x + w_temp/2;
            float bottom = center_y + h_temp/2;

            // printf("[%.2f,%.2f,%.2f,%.2f]::::%f,%f\n",center_x,center_y,w_temp,h_temp,scale_x,scale_y);

            for (int class_idx = 0; class_idx < classes_; class_idx++)       {
                float box_cls_score = data_ptr[batch_start_ptr +ptr_start + 5 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if (box_prob > threshold)         {
                    ObjRect obj_temp;
                    obj_temp.width = w_temp;
                    obj_temp.height = h_temp;
                    obj_temp.left = left;
                    obj_temp.top = top;
                    obj_temp.right = right;
                    obj_temp.bottom = bottom;
                    obj_temp.score = box_prob;
                    obj_temp.class_id = class_idx;
                    dect_temp.push_back(obj_temp);
                }
            }
        }
    }

    std::sort(dect_temp.begin(),dect_temp.end(),sort_ObjRect);

    std::vector<ObjRect> dect_temp_batch;
    std::vector<int> picked;
    dect_temp_batch.clear();
    nms_sorted_bboxes(dect_temp, picked, nms_threshold);

    for (size_t i = 0; i < picked.size(); i++)    {
        dect_temp_batch.push_back(dect_temp[picked[i]]);
    }
    
    detections.push_back(dect_temp_batch);
  }
}

void YoloX_PostForward::process(float* data_ptr,std::vector<int> output_shape, std::pair<int,int> ost_size, 
  float threshold, float nms_threshold, std::vector<ObjRect> &detections)
{
  int size_one_batch = output_shape[1]*output_shape[2];
  int channels_resu_ = output_shape[2];
  int classes_ = channels_resu_ - 5;

  int batch_start_ptr = 0;
  std::vector<ObjRect> dect_temp;
  dect_temp.clear();
  float scale_x = (float)ost_size.first/network_width;
  float scale_y = (float)ost_size.second/network_height;
  for (size_t i = 0; i < outlen_diml; i++)    {
    int ptr_start=i*channels_resu_;
    float box_objectness = data_ptr[batch_start_ptr + ptr_start+4];
    if(data_ptr[batch_start_ptr + ptr_start+4] >= threshold){
      float center_x = (data_ptr[batch_start_ptr +ptr_start+0] + grids_x_[i]) * expanded_strides_[i];
      float center_y = (data_ptr[batch_start_ptr +ptr_start+1] + grids_y_[i]) * expanded_strides_[i];
      float w_temp = exp(data_ptr[batch_start_ptr +ptr_start+2])*expanded_strides_[i];
      float h_temp = exp(data_ptr[batch_start_ptr +ptr_start+3])*expanded_strides_[i];
      float score = data_ptr[batch_start_ptr +ptr_start+4];
      center_x *= scale_x;
      center_y *= scale_y;
      w_temp *= scale_x;
      h_temp *= scale_y;
      float left = center_x - w_temp/2;
      float top = center_y - h_temp/2;
      float right = center_x + w_temp/2;
      float bottom = center_y + h_temp/2;

      // printf("[%.2f,%.2f,%.2f,%.2f]::::%f,%f\n",center_x,center_y,w_temp,h_temp,scale_x,scale_y);

      for (int class_idx = 0; class_idx < classes_; class_idx++)       {
          float box_cls_score = data_ptr[batch_start_ptr +ptr_start + 5 + class_idx];
          float box_prob = box_objectness * box_cls_score;
          if (box_prob > threshold)         {
              ObjRect obj_temp;
              obj_temp.width = w_temp;
              obj_temp.height = h_temp;
              obj_temp.left = left;
              obj_temp.top = top;
              obj_temp.right = right;
              obj_temp.bottom = bottom;
              obj_temp.score = box_prob;
              obj_temp.class_id = class_idx;
              dect_temp.push_back(obj_temp);
          }
      }
    }
  }

  std::sort(dect_temp.begin(),dect_temp.end(),sort_ObjRect);

  std::vector<int> picked;
  detections.clear();
  nms_sorted_bboxes(dect_temp, picked, nms_threshold);

  for (size_t i = 0; i < picked.size(); i++)    {
      detections.push_back(dect_temp[picked[i]]);
  }
}