#include <fstream>
#include "lprnet.hpp"
#include "utils.hpp"

using namespace std;

static char const *arr_chars[] = {"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",\
      "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", \
      "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B",\
      "C", "D", "E", "F", "G", "H", "J", "K", "L","M", "N", "P", "Q", "R", "S", "T", "U", "V",\
      "W", "X", "Y", "Z", "I", "O", "-"};
      
//const char *model_name = "lprnet";
string get_res(int pred_num[], int len_char, int clas_char);

LPRNET::LPRNET(const string bmodel, int dev_id){
  // init device id
  dev_id_ = dev_id;

  //create device handle
  bm_status_t ret = bm_dev_request(&bm_handle_, dev_id_);
  if (BM_SUCCESS != ret) {
      std::cout << "ERROR: bm_dev_request err=" << ret << std::endl;
      exit(-1);
  }

  // init bmruntime contxt
  p_bmrt_ = bmrt_create(bm_handle_);
  if (NULL == p_bmrt_) {
    cout << "ERROR: get handle failed!" << endl;
    exit(1);
  }

  // load bmodel by file
  bool flag = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
  if (!flag) {
    cout << "ERROR: Load bmodel[" << bmodel << "] failed" << endl;
    exit(-1);
  }

  bmrt_get_network_names(p_bmrt_, &net_names_);
  cout << "> Load model " << net_names_[0] << " successfully" << endl;

  // get model info by model name
  auto net_info = bmrt_get_network_info(p_bmrt_, net_names_[0]);
  if (NULL == net_info) {
    cout << "ERROR: get net-info failed!" << endl;
    exit(1);
  }
  cout << "input scale:" << net_info->input_scales[0] << endl;
  cout << "output scale:" << net_info->output_scales[0] << endl;
  //cout << "input number:" << net_info->input_num << endl;
  //cout << "output number:" << net_info->output_num << endl;

  /* get fp32/int8 type, the thresholds may be different */
  if (BM_FLOAT32 == net_info->input_dtypes[0]) {
    int8_flag_ = false;
    cout <<  "fp32 input" << endl;
  } else {
    int8_flag_ = true;
    cout <<  "int8 input" << endl;
  }
  if (BM_FLOAT32 == net_info->output_dtypes[0]) {
    int8_output_flag = false;
    cout <<  "fp32 output" << endl;
  } else {
    int8_output_flag = true;
    cout <<  "int8 output" << endl;
  }

#ifdef DEBUG
  bmrt_print_network_info(net_info);
#endif
  
  bm_shape_t input_shape = net_info->stages[0].input_shapes[0];
  /* attach device_memory for inference input data */
  bmrt_tensor(&input_tensor_, p_bmrt_, net_info->input_dtypes[0], input_shape);
  /* malloc input and output system memory for preprocess data */
  int input_count = bmrt_shape_count(&input_shape);
  cout << "input count:" << input_count << endl;
  if (int8_flag_) {
    input_int8 = new int8_t[input_count];
  } else {
    input_f32 = new float[input_count];
  }

  bm_shape_t output_shape = net_info->stages[0].output_shapes[0];
  bmrt_tensor(&output_tensor_, p_bmrt_, net_info->output_dtypes[0], output_shape);
  int output_count = bmrt_shape_count(&output_shape);
  //cout << "** output count:" << count << endl;
  if (int8_output_flag) {
    output_int8 = new int8_t[output_count];
  } else {
    output_f32 = new float[output_count];
  }
  //output_ = new float[count];


  //input_shape contain dims value(n,c,h,w)
  net_h_ = input_shape.dims[2];
  net_w_ = input_shape.dims[3];
  batch_size_ = input_shape.dims[0];
  num_channels_ = input_shape.dims[1];
  len_char = net_info->stages[0].output_shapes[0].dims[2];
  clas_char = net_info->stages[0].output_shapes[0].dims[1];
  output_scale = net_info->output_scales[0];
  count_per_img = output_count/batch_size_;

  input_scale = 0.0078125 * net_info->input_scales[0];
  vector<float> mean_values;
  mean_values.push_back(127.5);
  mean_values.push_back(127.5);
  mean_values.push_back(127.5);
  setMean(mean_values);

  ts_ = nullptr;
}

LPRNET::~LPRNET() {
  if (int8_flag_) {
    delete []input_int8;
  } else {
    delete []input_f32;
  }
  if (int8_output_flag) {
    delete []output_int8;
  } else {
    delete []output_f32;
  }
  bm_free_device(bm_handle_, input_tensor_.device_mem);
  bm_free_device(bm_handle_, output_tensor_.device_mem);
  free(net_names_);
  bmrt_destroy(p_bmrt_);
  bm_dev_free(bm_handle_);
}

void LPRNET::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}

void LPRNET::preForward(const vector<cv::Mat> &images) {
  LOG_TS(ts_, "lprnet pre-process")
  //cout << "input image size:" << image.size() << endl;
  for (int i = 0; i < batch_size_; i++){
    vector<cv::Mat> input_channels;
    wrapInputLayer(&input_channels, i);
    //cout << "input_channels.size = " << input_channels.size() << endl;
    preprocess(images[i], &input_channels);
  }
  LOG_TS(ts_, "lprnet pre-process")
}

void LPRNET::forward() {
  if (int8_flag_) {
    bm_memcpy_s2d(bm_handle_, input_tensor_.device_mem, ((void *)input_int8));
  } else {
    bm_memcpy_s2d(bm_handle_, input_tensor_.device_mem, ((void *)input_f32));
  }
  LOG_TS(ts_, "lprnet inference")
  bool ret = bmrt_launch_tensor_ex(p_bmrt_, net_names_[0],
                                  &input_tensor_, 1, &output_tensor_, 1, true, false);
  if (!ret) {
    cout << "ERROR: Failed to launch network" << net_names_[0] << "inference" << endl;
  }

  // sync, wait for finishing inference
  bm_thread_sync(bm_handle_);
  LOG_TS(ts_, "lprnet inference")
  
  size_t size = bmrt_tensor_bytesize(&output_tensor_);
  if (int8_output_flag) {
    bm_memcpy_d2s_partial(bm_handle_, output_int8, output_tensor_.device_mem, size);
  } else {
    bm_memcpy_d2s_partial(bm_handle_, output_f32, output_tensor_.device_mem, size);
  }
}

static bool comp(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

void LPRNET::postForward (vector<string> &detections) {
  // auto net_info = bmrt_get_network_info(p_bmrt_, net_names_[0]);
  // int stage_num = net_info->stage_num;
  // bm_shape_t output_shape;
  // for (int i = 0; i < stage_num; i++) {
  //   if (net_info->stages[i].input_shapes[0].dims[0] == 1) {
  //     output_shape = net_info->stages[i].output_shapes[0];
  //     break;
  //   }
  //   if ( i == (stage_num - 1)) {
  //     cout << "ERROR: output not match stages" << endl;
  //     return;
  //   }
  // }

  LOG_TS(ts_, "lprnet post-process")
  // int output_count = bmrt_shape_count(&output_shape);

  detections.clear();

  int N = 1;
  //cout << "len_char = " << len_char << endl;
  //cout << "output_int8 = " << output_int8 << endl;
  //cout << "output_scales=" << net_info->output_scales[0] << endl;
  
  //cout << image_output[0] <<endl;
  vector<pair<float , int>> pairs;
  //vector<string> res;
  for (int i = 0; i < batch_size_; i++) {
    //res.clear();
    int pred_num[len_char]={1000};
    for (int j = 0; j < len_char; j++){
      pairs.clear();
      for (int k = 0; k < clas_char; k++){
        if (int8_output_flag) {
          pairs.push_back(make_pair(output_int8[i * count_per_img + k * len_char + j] * output_scale, k));
        }else{
          pairs.push_back(make_pair(output_f32[i * count_per_img + k * len_char + j], k));
        }
      }
      partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), comp);
      //cout << pairs[0].second << " : " << pairs[0].first << endl;
      pred_num[j] = pairs[0].second;
    }
    string res = get_res(pred_num, len_char, clas_char);
#ifdef DEBUG
    cout << "res = " << res << endl;
#endif
    detections.push_back(res);
  }
  LOG_TS(ts_, "lprnet post-process")
}

void LPRNET::setMean(vector<float> &values) {
    vector<cv::Mat> channels;

    for (int i = 0; i < num_channels_; i++) {
      /* Extract an individual channel. */
      if (int8_flag_) {
        cv::Mat channel(net_h_, net_w_, CV_8SC1,cv::Scalar(0), cv::SophonDevice(this->dev_id_));
        channels.push_back(channel);
      } else {
        cv::Mat channel(net_h_, net_w_, CV_32FC1,cv::Scalar(0), cv::SophonDevice(this->dev_id_));
        channels.push_back(channel); 
      }
    }
    //init mat mean_
    vector<cv::Mat> channels_;
    for (int i = 0; i < num_channels_; i++) {
      /* Extract an individual channel. */
      cv::Mat channel_(net_h_, net_w_, CV_32FC1, cv::Scalar((float)values[i]), cv::SophonDevice(this->dev_id_));
      channels_.push_back(channel_);
    }
    if (int8_flag_) {
        mean_.create(net_h_, net_w_, CV_8SC3, dev_id_);
    }else{
        mean_.create(net_h_, net_w_, CV_32FC3, dev_id_);
    }

    cv::merge(channels_, mean_);
}

void LPRNET::wrapInputLayer(std::vector<cv::Mat>* input_channels, int batch_id) {
  int h = net_h_;
  int w = net_w_;

  //init input_channels
  if (int8_flag_) {
    int8_t *channel_base = input_int8;
    channel_base += h * w * num_channels_ * batch_id;
    for (int i = 0; i < num_channels_; i++) {
      cv::Mat channel(h, w, CV_8SC1, channel_base);
      input_channels->push_back(channel);
      channel_base += h * w;
    }
  } else {
    float *channel_base = input_f32;
    channel_base += h * w * num_channels_ * batch_id;
    for (int i = 0; i < num_channels_; i++) {
      cv::Mat channel(h, w, CV_32FC1, channel_base);
      input_channels->push_back(channel);
      channel_base += h * w;
    }
  }
}

void LPRNET::preprocess (const cv::Mat& img, vector<cv::Mat>* input_channels) {
   /* Convert the input image to the input image format of the network. */
  cv::Mat sample = img;
  cv::Mat sample_resized(net_h_, net_w_, CV_8UC3, cv::SophonDevice(dev_id_));
  if (sample.size() != cv::Size(net_w_, net_h_)) {
    cv::resize(sample, sample_resized, cv::Size(net_w_, net_h_));
  }
  else {
    sample_resized = sample;
  }

  cv::Mat sample_float(cv::SophonDevice(this->dev_id_));
  sample_resized.convertTo(sample_float, CV_32FC3);
  
  cv::Mat sample_normalized(cv::SophonDevice(this->dev_id_));
  cv::subtract(sample_float, mean_, sample_normalized);
  
  /*note: int8 in convert need mul input_scale*/
  if (int8_flag_) {
    //cout << "** int8 ** input_scale=" << input_scale << endl;
    cv::Mat sample_int8(cv::SophonDevice(this->dev_id_));
    sample_normalized.convertTo(sample_int8, CV_8SC1, input_scale); 
    cv::split(sample_int8, *input_channels);
  } else {
    //cout << "** f32" << "input_scale:" << input_scale  << endl;
    cv::Mat sample_fp32(cv::SophonDevice(this->dev_id_));
    sample_normalized.convertTo(sample_fp32, CV_32FC3, input_scale);
    //cout << sample_fp32 << endl;
    cv::split(sample_fp32, *input_channels);
  }
}

int LPRNET::batch_size() {
  return batch_size_;
};

string get_res(int pred_num[], int len_char, int clas_char){
  int no_repeat_blank[20];
  //int num_chars = sizeof(CHARS) / sizeof(CHARS[0]);
  int cn_no_repeat_blank = 0;
  int pre_c = pred_num[0];
  if (pre_c != clas_char - 1) {
      no_repeat_blank[0] = pre_c;
      cn_no_repeat_blank++;
  }
  for (int i = 0; i < len_char; i++){
      if (pred_num[i] == pre_c) continue;
      if (pred_num[i] == clas_char - 1){
          pre_c = pred_num[i];
          continue;
      }
      no_repeat_blank[cn_no_repeat_blank] = pred_num[i];
      pre_c = pred_num[i];
      cn_no_repeat_blank++;
  }

  //static char res[10];
  string res="";
  for (int j = 0; j < cn_no_repeat_blank; j++){
    res = res + arr_chars[no_repeat_blank[j]];
    //cout << arr_chars[no_repeat_blank[j]] << endl;
    //strcat(res, arr_chars[no_repeat_blank[j]]);  
  }
  //cout << temp << endl;
  //strcpy(res, temp);
  return res;
}