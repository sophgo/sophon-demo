#include <fstream>
#include "lprnet.hpp"

using namespace std;

static char const *arr_chars[] = {"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",\
      "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", \
      "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B",\
      "C", "D", "E", "F", "G", "H", "J", "K", "L","M", "N", "P", "Q", "R", "S", "T", "U", "V",\
      "W", "X", "Y", "Z", "I", "O", "-"};

//const char *model_name = "lprnet";
string get_res(int pred_num[], int len_char, int clas_char);

LPRNET::LPRNET(bm_handle_t bm_handle, const string bmodel):p_bmrt_(nullptr) {

  bool ret;

  // get device handle
  bm_handle_ = bm_handle;

  // init bmruntime contxt
  p_bmrt_ = bmrt_create(bm_handle_);
  if (NULL == p_bmrt_) {
    cout << "ERROR: get handle failed!" << endl;
    exit(1);
  }

  // load bmodel from file
  ret = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
  if (!ret) {
    cout << "ERROR: Load bmodel[" << bmodel << "] failed" << endl;
    exit(1);
  }

  bmrt_get_network_names(p_bmrt_, &net_names_);

  // get model info by model name
  net_info_ = bmrt_get_network_info(p_bmrt_, net_names_[0]);
  if (NULL == net_info_) {
    cout << "ERROR: get net-info failed!" << endl;
    exit(1);
  }

  // get data type
  if (NULL == net_info_->input_dtypes) {
    cout << "ERROR: get net input type failed!" << endl;
    exit(1);
  }

  if (BM_FLOAT32 == net_info_->input_dtypes[0])
    input_is_int8_ = false;
  else
    input_is_int8_ = true;

  if (BM_FLOAT32 == net_info_->output_dtypes[0])
    output_is_int8_ = false;
  else
    output_is_int8_ = true;

  // allocate output buffer
  if (output_is_int8_) {
    output_int8 = new int8_t[BUFFER_SIZE];
  }else{
    output_fp32 = new float[BUFFER_SIZE];
  }
  
  // init bm images for storing results of combined operation of resize & crop & split
  bm_status_t bm_ret = bm_image_create_batch(bm_handle_,
                              INPUT_HEIGHT,
                              INPUT_WIDTH,
                              FORMAT_BGR_PLANAR,
                              DATA_TYPE_EXT_1N_BYTE,
                              resize_bmcv_,
                              MAX_BATCH);
  if (BM_SUCCESS != bm_ret) {
    cout << "ERROR: bm_image_create_batch failed" << endl;
    exit(1);
  }

  // bm images for storing inference inputs
  bm_image_data_format_ext data_type;
  if (input_is_int8_) { // INT8
    data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  } else { // FP32
    data_type = DATA_TYPE_EXT_FLOAT32;
  }
  bm_ret = bm_image_create_batch (bm_handle_,
                               INPUT_HEIGHT,
                               INPUT_WIDTH,
                               FORMAT_BGR_PLANAR,
                               data_type,
                               linear_trans_bmcv_,
                               MAX_BATCH);

  if (BM_SUCCESS != bm_ret) {
    cout << "ERROR: bm_image_create_batch failed" << endl;
    exit(1);
  }

  // initialize linear transform parameter
  // - mean value
  // - scale value (mainly for INT8 calibration)
  output_scale = net_info_->output_scales[0];
  input_scale = net_info_->input_scales[0] * 0.0078125;
  //cout << "scale: "<< input_scale << endl;
  linear_trans_param_.alpha_0 = input_scale;
  linear_trans_param_.beta_0 = -127.5 * input_scale;
  linear_trans_param_.alpha_1 = input_scale;
  linear_trans_param_.beta_1 = -127.5 * input_scale;
  linear_trans_param_.alpha_2 = input_scale;
  linear_trans_param_.beta_2 = -127.5 * input_scale;

  bm_shape_t output_shape = net_info_->stages[0].output_shapes[0];
  batch_size_ = net_info_->stages[0].output_shapes[0].dims[0];
  len_char = net_info_->stages[0].output_shapes[0].dims[2];
  clas_char = net_info_->stages[0].output_shapes[0].dims[1];
  int output_count = bmrt_shape_count(&output_shape);
  count_per_img = output_count/batch_size_;
  

}

LPRNET::~LPRNET() {
  // deinit bm images
  bm_image_destroy_batch (resize_bmcv_, MAX_BATCH);
  bm_image_destroy_batch (linear_trans_bmcv_, MAX_BATCH);

  // free output buffer
  if (output_is_int8_) {
    delete []output_int8;
  } else {
    delete []output_fp32;
  }

  // deinit contxt handle
  bmrt_destroy(p_bmrt_);
  free(net_names_);
}

void LPRNET::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}

void LPRNET::preForward(vector<bm_image> &input) {
  LOG_TS(ts_, "lprnet pre-process")
  preprocess_bmcv (input);
  LOG_TS(ts_, "lprnet pre-process")
}

void LPRNET::forward() {
  //memset(output_, 0, sizeof(float) * BUFFER_SIZE);
  LOG_TS(ts_, "lprnet inference")
  bool res;
  if (output_is_int8_) {
    res = bm_inference(p_bmrt_, linear_trans_bmcv_, (int8_t*)output_int8, input_shape_, net_names_[0]);
  }else{
    res = bm_inference(p_bmrt_, linear_trans_bmcv_, (float*)output_fp32, input_shape_, net_names_[0]);
  }

  LOG_TS(ts_, "lprnet inference")
  if (!res) {
    cout << "ERROR : inference failed!!"<< endl;
    exit(1);
  }
}

static bool comp(const pair<float, int>& lhs,
                        const pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

void LPRNET::postForward(vector<bm_image> &input, vector<string> &detections) {
  // int stage_num = net_info_->stage_num;
  // bm_shape_t output_shape;
  // for (int i = 0; i < stage_num; i++) {
  //   if (net_info_->stages[i].input_shapes[0].dims[0] == (int)input.size()) {
  //     output_shape = net_info_->stages[i].output_shapes[0];
  //     break;
  //   }
  //   if ( i == (stage_num - 1)) {
  //     cout << "ERROR: output not match stages" << endl;
  //     return;
  //   }
  // }

  LOG_TS(ts_, "lprnet post-process")
  // int output_count = bmrt_shape_count(&output_shape);
  // int img_size = input.size();
  // int count_per_img = output_count/img_size;
  //cout << "img_size = " << img_size << endl;
  //cout << "count_per_img = " << count_per_img << endl;
  detections.clear();

  int N = 1;
  // int len_char = net_info_->stages[0].output_shapes[0].dims[2];
  // int clas_char = net_info_->stages[0].output_shapes[0].dims[1];
  //cout << "len_char = " << len_char << endl;
  //cout << "clas_char = " << clas_char << endl;
  
  vector<pair<float , int>> pairs;
  //vector<string> res;
  for (int i = 0; i < batch_size_; i++) {
    //res.clear();
    int pred_num[len_char]={1000};
    for (int j = 0; j < len_char; j++){
      pairs.clear();
      for (int k = 0; k < clas_char; k++){
        if (output_is_int8_){
          pairs.push_back(make_pair(output_int8[i * count_per_img + k * len_char + j] * output_scale, k));
        }else{
          pairs.push_back(make_pair(output_fp32[i * count_per_img + k * len_char + j], k));
        }
        
      }
      partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), comp);
      //cout << pairs[0].second << " : " << pairs[0].first << endl;
      pred_num[j] = pairs[0].second;
    }
    string res = get_res(pred_num, len_char, clas_char);
#ifdef DEBUG_RESULT
    cout << "res = " << res << endl;
#endif
    detections.push_back(res);
  }
  LOG_TS(ts_, "lprnet post-process")
}

void LPRNET::preprocess_bmcv (vector<bm_image> &input) {
  if (input.empty()) {
    cout << "mul-batch bmcv input empty!!!" << endl;
    return ;
  }

  if (!((1 == input.size()) || (4 == input.size()))) {
    cout << "mul-batch bmcv input error!!!" << endl;
    return ;
  }

  // set input shape according to input bm images
  input_shape_ = {4, {(int)input.size(), 3, INPUT_HEIGHT, INPUT_WIDTH}};

  // do not crop
  crop_rect_ = {0, 0, input[0].width, input[0].height};

  // resize && split by bmcv
  for (size_t i = 0; i < input.size(); i++) {
    LOG_TS(ts_, "lprnet pre-process-vpp")
    bmcv_image_vpp_convert (bm_handle_, 1, input[i], &resize_bmcv_[i], &crop_rect_);
    LOG_TS(ts_, "lprnet pre-process-vpp")
  }
  //cout << resize_bmcv_[0].width << " " << resize_bmcv_[0].height <<endl;
  // do linear transform
  LOG_TS(ts_, "linear transform")
  bmcv_image_convert_to(bm_handle_, input.size(), linear_trans_param_, resize_bmcv_, linear_trans_bmcv_);
  LOG_TS(ts_, "linear transform")
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