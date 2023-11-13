
#include "segformer.hpp"
#include <fstream>
#include <vector>
#include <string>

#define USE_ASPECT_RATIO 1
#define RESIZE_STRATEGY BMCV_INTER_NEAREST
#define USE_BMCV_VPP_CONVERT 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1

std::vector<std::string> cityscapes_classes();
std::vector<std::vector<int>> cityscapes_palette();
std::vector<std::vector<int>> get_palette(std::string dataset);
std::vector<std::vector<std::vector<int>>> palette_bmcv(std::vector<std::vector<int>> result, std::vector<std::vector<int>> palette);

SegFormer::SegFormer(int dev_id, std::string bmodel_file) : engine()
{
  engine = std::make_shared<sail::Engine>(dev_id);
  if (!engine->load(bmodel_file))
  {
    std::cout << "Engine load bmodel " << bmodel_file << "failed" << endl;
    exit(0);
  }
  std::cout << "SegFormer construct success !!!" << std::endl;
}

SegFormer::~SegFormer()
{
  std::cout << "SegFormer deconstruct done !!!" << std::endl;
}

int SegFormer::Init()
{
  std::cout << "===============================" << std::endl;
  // step1:initialize bmcv
  sail::Handle handle(engine->get_device_id());
  bmcv = std::make_shared<sail::Bmcv>(handle);

  // graph names of network
  graph_names = engine->get_graph_names();
  std::string gh_info;
  for_each(graph_names.begin(), graph_names.end(), [&](std::string &s)
           { gh_info += "0:" + s + ";"; });
  std::cout << "graph_names:" << gh_info << std::endl;
  if (graph_names.size() > 1)
  {
    std::cout << "NetworkNumError:this net only accept one network!" << std::endl;
    exit(1);
  }
  // input names of network

  input_names = engine->get_input_names(graph_names[0]);
  assert(input_names.size() > 0);
  std::string input_tensor_names;
  for_each(input_names.begin(), input_names.end(), [&](std::string &s)
           { input_tensor_names += "0:" + s + ";"; });
  std::cout << "net input name -> " << input_tensor_names << std::endl;
  if (input_names.size() > 1)
  {
    std::cout << "InputNumError: this net only accept one input!" << std::endl;
    exit(1);
  }
  // input shape of network 0
  input_shape = engine->get_input_shape(graph_names[0], input_names[0]);
  std::string input_tensor_shape;
  for_each(input_shape.begin(), input_shape.end(), [&](int s)
           { input_tensor_shape += std::to_string(s) + " "; });
  std::cout << "input tensor shape -> " << input_tensor_shape << std::endl;
  // data type of network input.
  input_dtype = engine->get_input_dtype(graph_names[0], input_names[0]);
  std::cout << "input dtype -> " << input_dtype << ", is fp32=" << ((input_dtype == BM_FLOAT32) ? "true" : "false")
            << std::endl;

  // output names of network
  output_names = engine->get_output_names(graph_names[0]);
  assert(output_names.size() > 0);
  std::string output_tensors_names;
  for_each(output_names.begin(), output_names.end(), [&](std::string &s)
           { output_tensors_names += "0:" + s + ";"; });
  std::cout << "net output name -> " << output_tensors_names << std::endl;
  // output shapes of network 0

  output_shape.resize(output_names.size());
  for (int i = 0; i < output_names.size(); i++)
  {
    output_shape[i] = engine->get_output_shape(graph_names[0], output_names[i]);
    std::string output_tensor_shape;
    for_each(output_shape[i].begin(), output_shape[i].end(),
             [&](int s)
             { output_tensor_shape += std::to_string(s) + " "; });
    std::cout << "output tensor " << i << " shape -> " << output_tensor_shape << std::endl;
  }
  // data type of network output.
  output_dtype = engine->get_output_dtype(graph_names[0], output_names[0]);
  std::cout << "output dtype -> " << output_dtype << ", is fp32=" << ((output_dtype == BM_FLOAT32) ? "true" : "false")
            << "\n";
  std::cout << "===============================" << std::endl;

  // step3:Initialize Network IO
  input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, false, false);
  input_tensors[input_names[0]] = input_tensor.get();
  output_tensor.resize(output_names.size());
  for (int i = 0; i < output_names.size(); i++)
  {
    output_tensor[i] = std::make_shared<sail::Tensor>(handle, output_shape[i], output_dtype, true, true);
    output_tensors[output_names[i]] = output_tensor[i].get();
  }
  engine->set_io_mode(graph_names[0], sail::SYSO);

  // Initialize net utils
  max_batch = input_shape[0];
  m_net_h = input_shape[2];
  m_net_w = input_shape[3];
  min_dim = output_shape[0].size();
  float input_scale = engine->get_input_scale(graph_names[0], input_names[0]);

  const std::vector<float> std = {58.395, 57.12, 57.375};
  const std::vector<float> mean = {123.675, 116.28, 103.53};

  ab[0] = 1 / (std[0]) * input_scale;
  ab[1] = (-mean[0] / std[0]) * input_scale;
  ab[2] = 1 / (std[1]) * input_scale;
  ab[3] = (-mean[1] / std[1]) * input_scale;
  ab[4] = 1 / (std[2]) * input_scale;
  ab[5] = (-mean[2] / std[2]) * input_scale;
  return 0;
}

void SegFormer::enableProfile(TimeStamp *ts)
{
  m_ts = ts;
}

int SegFormer::batch_size()
{
  return max_batch;
};

int SegFormer::Detect(std::vector<sail::BMImage> &input_images, std::vector<std::vector<std::vector<int32_t>>> &results_data)
{
  int ret = 0;
  // 1. preprocess
  LOG_TS(m_ts, "segformer preprocess");
  if (input_images.size() == 4 && max_batch == 4)
  {
    ret = pre_process<4>(input_images);
  }
  else if (input_images.size() == 1 && max_batch == 1)
  {
    ret = pre_process(input_images[0]);
  }
  else
  {
    std::cout << "unsupport batch size!" << std::endl;
    exit(1);
  }
  CV_Assert(ret == 0);
  LOG_TS(m_ts, "segformer preprocess");
  auto bmimg = bmcv->tensor_to_bm_image(*input_tensors[input_names[0]]);
  // 2. forward
  LOG_TS(m_ts, "segformer inference");
  engine->process(graph_names[0], input_tensors, output_tensors);
  LOG_TS(m_ts, "segformer inference");

  // 3. post process
  LOG_TS(m_ts, "segformer postprocess");
  ret = post_process(input_images, results_data);
  CV_Assert(ret == 0);
  LOG_TS(m_ts, "segformer postprocess");
  return ret;
}

int SegFormer::pre_process(sail::BMImage &input)
{
  int stride1[3], stride2[3];
  bm_image_get_stride(input.data(), stride1); // bmcv api
  stride2[0] = FFALIGN(stride1[0], 64);
  stride2[1] = FFALIGN(stride1[1], 64);
  stride2[2] = FFALIGN(stride1[2], 64);
  sail::BMImage rgb_img(engine->get_handle(), input.height(), input.width(), FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                        stride2);
  bmcv->convert_format(input, rgb_img);
  sail::BMImage convert_img(engine->get_handle(), input_shape[2], input_shape[3], FORMAT_RGB_PLANAR,
                            bmcv->get_bm_image_data_format(input_dtype));

#if USE_ASPECT_RATIO
  bool isAlignWidth = false;
  float ratio = get_aspect_scaled_ratio(input.width(), input.height(), m_net_w, m_net_h, &isAlignWidth);
  sail::PaddingAtrr pad = sail::PaddingAtrr();
  pad.set_r(114);
  pad.set_g(114);
  pad.set_b(114);
  if (isAlignWidth)
  {
    unsigned int th = input.height() * ratio;
    pad.set_h(th);
    pad.set_w(m_net_w);
    int ty1 = (int)((m_net_h - th) / 2);
    pad.set_sty(ty1);
    pad.set_stx(0);
  }
  else
  {
    pad.set_h(m_net_h);
    unsigned int tw = input.width() * ratio;
    pad.set_w(tw);

    int tx1 = (int)((m_net_w - tw) / 2);
    pad.set_sty(0);
    pad.set_stx(tx1);
  }
#if USE_BMCV_VPP_CONVERT
  // Using BMCV api, align with yolov5_bmcv.
  sail::BMImage resized_img(engine->get_handle(), input_shape[2], input_shape[3], FORMAT_RGB_PLANAR,
                            DATA_TYPE_EXT_1N_BYTE);
  bmcv_rect_t rect;
  rect.start_x = 0;
  rect.start_y = 0;
  rect.crop_w = input.width();
  rect.crop_h = input.height();
  bmcv_padding_atrr_t padding;
  padding.dst_crop_stx = pad.dst_crop_stx;
  padding.dst_crop_sty = pad.dst_crop_sty;
  padding.dst_crop_w = pad.dst_crop_w;
  padding.dst_crop_h = pad.dst_crop_h;
  padding.if_memset = 1;
  padding.padding_r = pad.padding_r;
  padding.padding_g = pad.padding_g;
  padding.padding_b = pad.padding_b;
  auto ret = bmcv_image_vpp_convert_padding(engine->get_handle().data(), 1, rgb_img.data(), &resized_img.data(),
                                            &padding, &rect);
  assert(ret == 0);

#else
  sail::BMImage resized_img =
      bmcv->vpp_crop_and_resize_padding(rgb_img, 0, 0, rgb_img.width(), rgb_img.height(), m_net_w, m_net_h, pad);
#endif
#else
  sail::BMImage resized_img =
      bmcv->crop_and_resize(rgb_img, 0, 0, rgb_img.width(), rgb_img.height(), m_net_w, m_net_h, RESIZE_STRATEGY);
#endif

  cv::Mat resized_mat;
  cv::bmcv::toMAT(&resized_img.data(), resized_mat);
  stored_imgs.push_back(resized_mat);

  bmcv->convert_to(
      resized_img, convert_img,
      std::make_tuple(std::make_pair(ab[0], ab[1]), std::make_pair(ab[2], ab[3]), std::make_pair(ab[4], ab[5])));
  bmcv->bm_image_to_tensor(convert_img, *input_tensor.get());

  // cout<<"input tensor "<<*input_tensor.get()<<endl;
#if DUMP_FILE
  cv::Mat stored_img;
  int i = 0;
  cv::bmcv::toMAT(&resized_img.data(), stored_img);
  std::string fname = cv::format("stored_img%d.jpg", i);
  cv::imwrite(fname, stored_img);

#endif

  return 0;
}

template <std::size_t N>
int SegFormer::pre_process(std::vector<sail::BMImage> &input)
{
  if (input.size() != N)
  {
    std::cout << "Unsupport batch size!" << std::endl;
    exit(1);
  }
  std::shared_ptr<sail::BMImage> resized_imgs_vec[N];
  sail::BMImageArray<N> resized_imgs;
  sail::BMImageArray<N> convert_imgs(engine->get_handle(), input_shape[2], input_shape[3], FORMAT_RGB_PLANAR,
                                     bmcv->get_bm_image_data_format(input_dtype));

  for (size_t i = 0; i < input.size(); ++i)
  {
    int stride1[3], stride2[3];
    bm_image_get_stride(input[i].data(), stride1); // bmcv api
    stride2[0] = FFALIGN(stride1[0], 64);
    stride2[1] = FFALIGN(stride1[1], 64);
    stride2[2] = FFALIGN(stride1[2], 64);
    sail::BMImage rgb_img(engine->get_handle(), input[i].height(), input[i].width(), FORMAT_RGB_PLANAR,
                          DATA_TYPE_EXT_1N_BYTE, stride2);
    bmcv->convert_format(input[i], rgb_img);

#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(input[i].width(), input[i].height(), m_net_w, m_net_h, &isAlignWidth);
    sail::PaddingAtrr pad = sail::PaddingAtrr();
    pad.set_r(114);
    pad.set_g(114);
    pad.set_b(114);
    if (isAlignWidth)
    {
      unsigned int th = input[i].height() * ratio;
      pad.set_h(th);
      pad.set_w(m_net_w);
      int ty1 = (int)((m_net_h - th) / 2);
      pad.set_sty(ty1);
      pad.set_stx(0);
    }
    else
    {
      pad.set_h(m_net_h);
      unsigned int tw = input[i].width() * ratio;
      pad.set_w(tw);
      int tx1 = (int)((m_net_w - tw) / 2);
      pad.set_sty(0);
      pad.set_stx(tx1);
    }
    resized_imgs_vec[i] = std::make_shared<sail::BMImage>(engine->get_handle(), input_shape[2], input_shape[3],
                                                          FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE);
#if USE_BMCV_VPP_CONVERT
    // Using BMCV api, align with yolov5_bmcv.
    bmcv_rect_t rect;
    rect.start_x = 0;
    rect.start_y = 0;
    rect.crop_w = input[i].width();
    rect.crop_h = input[i].height();
    bmcv_padding_atrr_t padding;
    padding.dst_crop_stx = pad.dst_crop_stx;
    padding.dst_crop_sty = pad.dst_crop_sty;
    padding.dst_crop_w = pad.dst_crop_w;
    padding.dst_crop_h = pad.dst_crop_h;
    padding.if_memset = 1;
    padding.padding_r = pad.padding_r;
    padding.padding_g = pad.padding_g;
    padding.padding_b = pad.padding_b;
    auto ret = bmcv_image_vpp_convert_padding(engine->get_handle().data(), 1, rgb_img.data(),
                                              &resized_imgs_vec[i].get()->data(), &padding, &rect);
    assert(ret == 0);
#else
    bmcv->vpp_crop_and_resize_padding(&rgb_img.data(), &resized_imgs_vec[i].get()->data(), 0, 0, rgb_img.width(),
                                      rgb_img.height(), m_net_w, m_net_h, pad);
#endif
    resized_imgs.attach_from(i, *resized_imgs_vec[i].get());
#else
    sail::BMImage resized_img =
        bmcv->crop_and_resize(rgb_img, 0, 0, rgb_img.width(), rgb_img.height(), m_net_w, m_net_h, RESIZE_STRATEGY);
    resized_imgs.CopyFrom(i, resized_img);
#endif
    cv::Mat resized_mat;
    cv::bmcv::toMAT(&resized_imgs_vec[i].get()->data(), resized_mat);
    stored_imgs.push_back(resized_mat);
  }
  bmcv->convert_to(
      resized_imgs, convert_imgs,
      std::make_tuple(std::make_pair(ab[0], ab[1]), std::make_pair(ab[2], ab[3]), std::make_pair(ab[4], ab[5])));
  bmcv->bm_image_to_tensor(convert_imgs, *input_tensor.get());
  return 0;
}

int SegFormer::post_process(std::vector<sail::BMImage> &images, std::vector<std::vector<std::vector<int32_t>>> &results_data)
{
  // auto inputTensors=input_tensor;
  for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx)
  {

    int feat_c = output_tensor[batch_idx]->shape()[1];
    int raw_height = output_tensor[batch_idx]->shape()[2];
    int col_width = output_tensor[batch_idx]->shape()[3];

    bool own_sys_data = output_tensor[batch_idx]->own_sys_data();
    bool own_dev_data = output_tensor[batch_idx]->own_dev_data();

    bm_data_type_t dtype = output_tensor[batch_idx]->dtype();
    const void *output_data = output_tensor[batch_idx]->sys_data();

    std::vector<std::vector<int32_t>> result_data(raw_height, std::vector<int32_t>(col_width, 0));
    // 将数据复制到 result_data
    for (int row = 0; row < raw_height; row++)
    {
      for (int col = 0; col < col_width; col++)
      {
        if (dtype == BM_INT32)
        {
          result_data[row][col] = static_cast<int32_t>(*reinterpret_cast<const int32_t *>(output_data));
          output_data = reinterpret_cast<const int32_t *>(output_data) + 1;
        }
        else
        {
          result_data[row][col] = static_cast<int32_t>(*reinterpret_cast<const float *>(output_data));
          output_data = reinterpret_cast<const float *>(output_data) + 1;
        }
      }
    }

    std::vector<std::vector<int>> palette_out = get_palette(palette);
    std::vector<std::vector<std::vector<int>>> color_seg = palette_bmcv(result_data, palette_out);

    // 先将 color_seg 转换为 cv::Mat
    cv::Mat color_seg_mat(color_seg.size(), color_seg[0].size(), CV_8UC3);
    //  color_seg_mat
    for (int i = 0; i < color_seg.size(); ++i)
    {
      for (int j = 0; j < color_seg[i].size(); ++j)
      {
        const std::vector<int> &color1 = color_seg[i][j];
        cv::Vec3b &pixel = color_seg_mat.at<cv::Vec3b>(i, j);
        pixel[0] = (color1[0]); // R
        pixel[1] = (color1[1]); // G
        pixel[2] = (color1[2]); // B
      }
    }

    // cv::imwrite("color_seg_mat.png",color_seg_mat);
    cv::Mat img = stored_imgs[batch_idx];
    cv::Mat seg_result(color_seg.size(), color_seg[0].size(), CV_8UC3);
    for (int i = 0; i < color_seg.size(); ++i)
    {
      for (int j = 0; j < color_seg[i].size(); ++j)
      {
        cv::Vec3b color1 = color_seg_mat.at<cv::Vec3b>(i, j);
        cv::Vec3b color2 = img.at<cv::Vec3b>(i, j);

        cv::Vec3b &pixel = seg_result.at<cv::Vec3b>(i, j);
        pixel[0] = color1[0] * 0.6 + color2[0] * 0.4; // R
        pixel[1] = color1[1] * 0.6 + color2[1] * 0.4; // G
        pixel[2] = color1[2] * 0.6 + color2[2] * 0.4; // B
      }
    }

    // cv::imwrite("color_seg.png",seg_result);
    int ret = cv::bmcv::toBMI(seg_result, &images[batch_idx].data());
    if (ret != BM_SUCCESS)
    {
      spdlog::error("cv::bmcv::toBMI() err {},{}", __FILE__, __LINE__);
    }
    results_data.push_back(result_data);
  }
  stored_imgs.clear();
  return 0;
}

float SegFormer::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
  float ratio;
  float r_w = (float)dst_w / src_w;
  float r_h = (float)dst_h / src_h;
  if (r_h > r_w)
  {
    *pIsAligWidth = true;
    ratio = r_w;
  }
  else
  {
    *pIsAligWidth = false;
    ratio = r_h;
  }
  return ratio;
}

std::vector<std::string> cityscapes_classes()
{
  return {"road", "sidewalk", "building", "wall", "fence", "pole",
          "traffic light", "traffic sign", "vegetation", "terrain", "sky",
          "person", "rider", "car", "truck", "bus", "train", "motorcycle",
          "bicycle"};
}

std::vector<std::vector<int>> cityscapes_palette()
{
  return {{128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156}, {190, 153, 153}, {153, 153, 153}, {250, 170, 30}, {220, 220, 0}, {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60}, {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100}, {0, 80, 100}, {0, 0, 230}, {119, 11, 32}};
}

std::vector<std::vector<int>> get_palette(std::string dataset)
{
  std::unordered_map<std::string, std::vector<std::string>> dataset_aliases = {
      {"cityscapes", {"cityscapes"}},
      {"ade", {"ade", "ade20k"}},
      {"voc", {"voc", "pascal_voc", "voc12", "voc12aug"}}};

  std::unordered_map<std::string, std::string> alias2name;
  for (const auto &entry : dataset_aliases)
  {
    const std::string &name = entry.first;
    const std::vector<std::string> &aliases = entry.second;
    for (const auto &alias : aliases)
    {
      alias2name[alias] = name;
    }
  }

  if (alias2name.find(dataset) != alias2name.end())
  {
    const std::string &name = alias2name[dataset];
    if (name == "cityscapes")
    {
      return cityscapes_palette();
    }
    else
    {
      throw std::runtime_error("Unrecognized dataset: " + dataset);
    }
  }
  else
  {
    throw std::runtime_error("Unrecognized dataset: " + dataset);
  }
}

std::vector<std::vector<int>> PALETTE = {
    {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156}, {190, 153, 153}, {153, 153, 153}, {250, 170, 30}, {220, 220, 0}, {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60}, {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100}, {0, 80, 100}, {0, 0, 230}, {119, 11, 32}};

std::vector<std::string> CLASSES = {
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"};

std::vector<std::vector<std::vector<int>>> palette_bmcv(std::vector<std::vector<int>> result, std::vector<std::vector<int>> palette)
{
  std::vector<std::vector<int>> seg = result;
  if (palette.empty())
  {
    if (PALETTE.empty())
    {
      palette.resize(CLASSES.size());
      srand(time(0)); // Seed the random number generator
      for (int i = 0; i < CLASSES.size(); ++i)
      {
        palette[i].resize(3);

        for (int j = 0; j < 3; ++j)
        {
          palette[i][j] = rand() % 256;
        }
      }
    }
    else
    {
      palette = PALETTE;
    }
  }
  std::vector<std::vector<std::vector<int>>> color_seg(seg.size(), std::vector<std::vector<int>>(seg[0].size(), std::vector<int>(3, 0)));

  for (int i = 0; i < seg.size(); ++i)
  {
    for (int j = 0; j < seg[0].size(); ++j)
    {
      int label = seg[i][j];
      std::vector<int> color = palette[label];
      color_seg[i][j] = color;
    }
  }

  // Convert to BGR
  for (int i = 0; i < color_seg.size(); ++i)
  {
    for (int j = 0; j < color_seg[i].size(); ++j)
    {
      std::swap(color_seg[i][j][0], color_seg[i][j][2]);
    }
  }
  return color_seg;
}