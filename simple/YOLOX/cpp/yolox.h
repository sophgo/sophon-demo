#ifndef __INCLUDE_YOLOX_H_
#define __INCLUDE_YOLOX_H_
#define USE_FFMPEG  1
#define USE_OPENCV  1
#define USE_BMCV    1
#include <sail/cvwrapper.h>
#include <sail/engine.h>
#include <sail/tensor.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace sail;

struct ObjRect
{
    unsigned int class_id;
    float score;
    float left;
    float top;
    float right;
    float bottom;
    float width;
    float height;
};


class YoloXDete
{ 
public:
    YoloXDete(std::string bmodel_path, int device_id);
    ~YoloXDete();

    void Inference();
    int get_batchsize();
    int get_net_input_w();
    int get_net_input_h();
    float get_input_scale();
    bm_image_data_format_ext get_img_dtype();
    sail::Handle get_handle();
    void* get_output_data_prt();
    std::vector<int> get_output_shape();

    sail::Tensor* input_tensor = NULL;
    sail::Tensor* output_tensor = NULL;

private:
    sail::Engine *engie = NULL;

    std::map<std::string,sail::Tensor*> input_tensormap;
    std::map<std::string,sail::Tensor*> output_tensormap;

    string graph_name;
    string input_tensor_name;
    string output_tensor_name;

    float input_scale;
    bm_image_data_format_ext img_dtype;

    int batch_size;
    int net_input_w;
    int net_input_h;

    sail::Handle handle;
};

//
class VideoProcess
{
public:
    VideoProcess(std::string video_name,
        sail::Handle& handle, 
        int batch_size, 
        int device_id,
        int resize_width,
        int resize_height,
        bm_image_data_format_ext output_dtype);

    ~VideoProcess();

    int getTensor(sail::Handle& handle,sail::Bmcv& bmcv,
        sail::Tensor& output_tensor,
        const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>> &alpha_beta);

    std::vector<float> get_resize_scalemin();

    sail::BMImage input_image;
    sail::BMImage resize_image;
    sail::BMImage output_image;
    sail::BMImageArray<4> input_image_array;
    sail::BMImageArray<4> resize_image_array;
    sail::BMImageArray<4> output_image_array;

private:
    sail::Decoder *decoder = NULL;
    int frame_skip;
    int image_width;
    int image_height;
    int output_width;
    int output_height;

    float scale_w;
    float scale_h;
    float scale_min; 
    int batch_size;
    sail::PaddingAtrr paddingatt;

    void calc_scale_padatt();
    int getBMImage(sail::Handle& handle);
    int getBMImageArray(sail::Handle& handle);
    int process_padding_BMImage(sail::Bmcv& bmcv);
    int process_padding_BMImageArray(sail::Bmcv& bmcv);

};

class PictureProcess
{
public:
    PictureProcess(std::string pic_path,
        sail::Handle& handle, 
        int batch_size, 
        int device_id,
        int resize_width,
        int resize_height,
        bm_image_data_format_ext output_dtype);

    ~PictureProcess();

    int getTensor(sail::Handle& handle,sail::Bmcv& bmcv,
        sail::Tensor& output_tensor,
        const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>> &alpha_beta);

    std::vector<float> get_resize_scalemin();
    std::vector<string> get_curr_imagename();

    sail::BMImage input_image;
    sail::BMImage resize_image;
    sail::BMImage output_image;

    std::vector<sail::BMImage> input_image_array;
    sail::BMImageArray<4> resize_image_array;
    sail::BMImageArray<4> output_image_array;


private:
    int image_width;
    int image_height;
    int output_width;
    int output_height;

    std::vector<float> scale_min_v; 
    float scale_min;
    int batch_size;
    sail::PaddingAtrr paddingatt;

    std::vector<string> imagename_list;      // all images name
    std::vector<string> input_name_list;     //current images name

    int current_idx;
    int device_id;

    void calc_scale_padatt(int iput_w, int input_h, int output_w, int output_h);
    int getBMImage(sail::Handle& handle,sail::Bmcv& bmcv);
    int getBMImageArray(sail::Handle& handle,sail::Bmcv& bmcv);
    int process_padding_BMImageArray(sail::Bmcv& bmcv);
};

class YoloX_PostForward
{
private:
  /* data */
public:
  YoloX_PostForward(int net_w, int net_h, std::vector<int> strides);
  ~YoloX_PostForward();
  void process(float* data_ptr,std::vector<int> output_shape, std::vector<std::pair<int,int>> ost_size, 
    float threshold, float nms_threshold, std::vector<std::vector<ObjRect>> &detections);
  
  void process(float* data_ptr,std::vector<int> output_shape, std::pair<int,int> ost_size, 
    float threshold, float nms_threshold, std::vector<ObjRect> &detections);

  void process(float* data_ptr,std::vector<int> output_shape, std::vector<float> resize_scale, 
    float threshold, float nms_threshold, std::vector<std::vector<ObjRect>> &detections);

private:
  int outlen_diml ;
  int* grids_x_;
  int* grids_y_;
  int* expanded_strides_;
  int network_width;
  int network_height;
};

float box_iou_FM(ObjRect a, ObjRect b);




#endif