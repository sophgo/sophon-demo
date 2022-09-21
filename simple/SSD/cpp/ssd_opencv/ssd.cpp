#include "ssd.hpp"
#include "utils.hpp"

#define IS_END_OF_DATA(entry) (entry[1] == 0 && entry[2] == 0 &&\
                               entry[3] == 0 && entry[4] == 0 &&\
                               entry[5] == 0 && entry[6] == 0)

using namespace std;

string net_name_;
//const char *model_name = "VGG_VOC0712_SSD_300x300_deploy";
SSD::SSD(const std::string bmodel, int dev_id) {
  // init device id
  dev_id_ = dev_id;
  //create device handle
  bm_status_t ret = bm_dev_request(&bm_handle_, dev_id_);
  if (BM_SUCCESS != ret) {
      std::cout << "ERROR: bm_dev_request err=" << ret << std::endl;
      exit(-1);
  }

  //create inference runtime handle
  p_bmrt_ = bmrt_create(bm_handle_);


#ifdef SOC_MODE
  set_bmrt_mmap(p_bmrt_, true);
#endif

  // load bmodel by file
  bool flag = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
  if (!flag) {
    std::cout << "ERROR: Load bmodel[" << bmodel << "] failed" << std::endl;
    exit(-1);
  }

  const char **net_names;
  bmrt_get_network_names(p_bmrt_, &net_names);
  net_name_ = net_names[0];
  free(net_names);
  // as of a simple example, assume the model file contains one SSD network only
  std::cout << "> Load model " << net_name_.c_str() << " successfully" << std::endl;

  //more info pelase see bm_net_info_t in bmdef.h
  auto net_info = bmrt_get_network_info(p_bmrt_, net_name_.c_str());
  std::cout << "** input scale:" << net_info->input_scales[0] << std::endl;
  std::cout << "** output scale:" << net_info->output_scales[0] << std::endl;

  //int8 out score different f32.reduce threshold to make result same
  if (BM_FLOAT32 == net_info->input_dtypes[0]) {
    threshold_ = 0.6;
    flag_int8 = false;
  } else {
    threshold_ = 0.52;
    flag_int8 = true;
  }

  //for int8 input data need mul input_scale
  if (flag_int8) input_scale = net_info->input_scales[0];

  // only one input shape supported in the pre-built model
  //you can get stage_num from net_info
  int stage_num = net_info->stage_num;
  bm_shape_t input_shape;
  bm_shape_t output_shape;
  for (int i = 0; i < stage_num; i++) {
    if(net_info->stages[i].input_shapes[0].dims[0] == 1) {
      output_shape = net_info->stages[i].output_shapes[0];
      input_shape = net_info->stages[i].input_shapes[0];
      break;
    }

    if ( i == (stage_num - 1)) {
      std::cout << "ERROR: output not match stages" << std::endl;
      return;
    }
  }

  //malloc device_memory for inference input and output data
  bmrt_tensor(&input_tensor_, p_bmrt_, net_info->input_dtypes[0], input_shape);
  bmrt_tensor(&output_tensor_, p_bmrt_, net_info->output_dtypes[0], output_shape);

  int count;
  count = bmrt_shape_count(&input_shape);
  std::cout << "** input count:" << count << std::endl;
  //malloc system memory for preprocess data
  if (flag_int8) {
    input_int8 = new int8_t[count];
  } else {
    input_f32 = new float[count];
  }
  count = bmrt_shape_count(&output_shape);
  std::cout << "** output count:" << count << std::endl;
  output_ = new float[count];

  //input_shape contain dims value(n,c,h,w)
  batch_size_ = input_shape.dims[0];
  num_channels_ = input_shape.dims[1];
  input_geometry_.height = input_shape.dims[2];
  input_geometry_.width = input_shape.dims[3];

  std::vector<float> mean_values;
  mean_values.push_back(123);
  mean_values.push_back(117);
  mean_values.push_back(104);
  setMean(mean_values);

  ts_ = nullptr;

}

SSD::~SSD() {
  if (flag_int8) {
    delete []input_int8;
  } else {
    delete []input_f32;
  }
  delete []output_;
  bm_free_device(bm_handle_, input_tensor_.device_mem);
  bm_free_device(bm_handle_, output_tensor_.device_mem);
  bmrt_destroy(p_bmrt_);
  bm_dev_free(bm_handle_);
}

void SSD::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}

void SSD::preForward(const cv::Mat &image) {
  LOG_TS(ts_, "ssd pre-process")
  std::cout << "input image size:" << image.size() << std::endl;
  std::vector<cv::Mat> input_channels;
  wrapInputLayer(&input_channels);
  preprocess(image, &input_channels);
  LOG_TS(ts_, "ssd pre-process")
}

void SSD::forward() {
#ifdef DEBUG_INT8
  FILE * pFile;
  pFile = fopen ("input_data.bin", "wb");
  fwrite (input_int8, input_tensor_.device_mem.size, 1, pFile);
  fclose (pFile);
#endif
  //copy system memory data to device memory for inference
  if (flag_int8) {
    bm_memcpy_s2d(bm_handle_, input_tensor_.device_mem, ((void *)input_int8));
  } else {
    bm_memcpy_s2d(bm_handle_, input_tensor_.device_mem, ((void *)input_f32));
  }
  LOG_TS(ts_, "ssd inference")
  bool ret = bmrt_launch_tensor_ex(p_bmrt_, net_name_.c_str(),
                                  &input_tensor_, 1, &output_tensor_, 1, true, false);
  if (!ret) {
    std::cout << "ERROR: Failed to launch network" << net_name_.c_str() << "inference" << std::endl;
  }

  // sync, wait for finishing inference
  bm_thread_sync(bm_handle_);
  LOG_TS(ts_, "ssd inference")

  size_t size = bmrt_tensor_bytesize(&output_tensor_);
  bm_memcpy_d2s_partial(bm_handle_, output_, output_tensor_.device_mem, size);
#ifdef DEBUG_INT8
  pFile = fopen ("output_data.bin", "wb");
  fwrite (output_int8, size, 1, pFile);
  fclose (pFile);
#endif
}

void SSD::postForward(const cv::Mat &image, std::vector<ObjRect> &detections) {
  auto net_info = bmrt_get_network_info(p_bmrt_, net_name_.c_str());
  int stage_num = net_info->stage_num;
  bm_shape_t output_shape;
  for (int i = 0; i < stage_num; i++) {
    if(net_info->stages[i].input_shapes[0].dims[0] == 1) {
      output_shape = net_info->stages[i].output_shapes[0];
      break;
    }

    if ( i == (stage_num - 1)) {
      std::cout << "ERROR: output not match stages" << std::endl;
      return;
    }
  }

  LOG_TS(ts_, "ssd post-process")
  int output_count = bmrt_shape_count(&output_shape);
  for (int i = 0; i < output_count; i += 7) {
    ObjRect detection;
    float *proposal = &output_[i];

    if (IS_END_OF_DATA(proposal))
      break;

    /*
     * Detection_Output format:
     *   [image_id, label, score, xmin, ymin, xmax, ymax]
     */
    //std::cout << "** score: " << proposal[2] << std::endl;
    if (proposal[2] < threshold_ || proposal[2] >= 1.0 )
      continue;

    detection.class_id = proposal[1];
    detection.score = proposal[2];
    detection.x1 = proposal[3] * image.cols;
    detection.y1 = proposal[4] * image.rows;
    detection.x2 = proposal[5] * image.cols;
    detection.y2 = proposal[6] * image.rows;
#define DEBUG_SSD
#ifdef DEBUG_SSD
    std::cout << "class id: " << std::setw(2) << detection.class_id
            << std::fixed << std::setprecision(5)
            << " upper-left: (" << std::setw(10) << detection.x1 << ", "
            << std::setw(10) << detection.y1 << ") "
            << " object-size: (" << std::setw(10) << detection.x2 - detection.x1 + 1 << ", "
            << std::setw(10)  << detection.y2 - detection.y1 + 1 << ")"
            << std::endl;
#endif

    detections.push_back(detection);
  }
  LOG_TS(ts_, "ssd post-process")
}

void SSD::setMean(std::vector<float> &values) {
    std::vector<cv::Mat> channels;

    for (int i = 0; i < num_channels_; i++) {
      /* Extract an individual channel. */
      if (flag_int8) {
        cv::Mat channel(input_geometry_.height, input_geometry_.width,
                   CV_8SC1,cv::Scalar(0), cv::SophonDevice(this->dev_id_));
        channels.push_back(channel);
      } else {
        cv::Mat channel(input_geometry_.height, input_geometry_.width,
                   CV_32FC1,cv::Scalar(0), cv::SophonDevice(this->dev_id_));
        channels.push_back(channel); 
      }
    }
    //init mat mean_
    std::vector<cv::Mat> channels_;
    for (int i = 0; i < num_channels_; i++) {
      /* Extract an individual channel. */
      cv::Mat channel_(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar((float)values[i]), cv::SophonDevice(this->dev_id_));
      channels_.push_back(channel_);
    }
    if (flag_int8) {
        mean_.create(input_geometry_.height, input_geometry_.width, CV_8SC3, dev_id_);
    }else{
        mean_.create(input_geometry_.height, input_geometry_.width, CV_32FC3, dev_id_);
    }

    cv::merge(channels_, mean_);
}

void SSD::wrapInputLayer(std::vector<cv::Mat>* input_channels) {
  int h = input_geometry_.height;
  int w = input_geometry_.width;

  //init input_channels
  if (flag_int8) {
    int8_t *channel_base = input_int8;
    for (int i = 0; i < num_channels_; i++) {
      cv::Mat channel(h, w, CV_8SC1, channel_base);
      input_channels->push_back(channel);
      channel_base += h * w;
    }
  } else {
    float *channel_base = input_f32;
    for (int i = 0; i < num_channels_; i++) {
      cv::Mat channel(h, w, CV_32FC1, channel_base);
      input_channels->push_back(channel);
      channel_base += h * w;
    }
  }
}

void SSD::preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample = img;
  cv::Mat sample_resized(input_geometry_.height, input_geometry_.width, CV_8UC3, cv::SophonDevice(dev_id_));
  if (sample.size() != input_geometry_) {
    cv::resize(sample, sample_resized, input_geometry_);
  }
  else {
    sample_resized = sample;
  }

  cv::Mat sample_float(cv::SophonDevice(this->dev_id_));
  sample_resized.convertTo(sample_float, CV_32FC3);

  cv::Mat sample_normalized(cv::SophonDevice(this->dev_id_));
  cv::subtract(sample_float, mean_, sample_normalized);
  
  /*note: int8 in convert need mul input_scale*/
  if (flag_int8) {
    std::cout << "** int8" << std::endl;
    cv::Mat sample_int8(cv::SophonDevice(this->dev_id_));
    sample_normalized.convertTo(sample_int8, CV_8SC1, input_scale);
    cv::split(sample_int8, *input_channels);
  } else {
    std::cout << "** f32" << std::endl;
    cv::split(sample_normalized, *input_channels);
  }
}

bool SSD::getPrecision() {
  return flag_int8;
}
