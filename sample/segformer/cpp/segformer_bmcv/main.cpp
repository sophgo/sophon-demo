//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include "ff_decode.hpp"
#include "segformer.hpp"
#include <iostream>

using json = nlohmann::json;
using namespace std;
#define USE_OPENCV_DECODE 0

using namespace std;
void readFilesRecursive(const std::string &directory, std::vector<std::string> &files_vector);
std::string getRelativePath(const std::string &filepath, const std::string &basepath);
std::string replaceSuffix(const std::string &relative_path, const std::string &img_suffix, const std::string &seg_map_suffix);
std::string removePrefix(const std::string &path, const std::string &prefix);

std::unordered_map<std::string, std::string> data = {
    {"data_root", "datasets/cityscapes_small"},
    {"img_dir", "leftImg8bit/val"},
    {"ann_dir", "gtFine/val"},
    {"img_suffix", "_leftImg8bit.png"},
    {"seg_map_suffix", "_gtFine_labelTrainIds.png"}};

int main(int argc, char *argv[])
{

  cout.setf(ios::fixed);
  cout << "===========================+++++++++++++++++++++++++++" << endl;
  cout << "hello,i am going to get params...." << endl;
  // get params
  // segformer_fp32.b0.512x1024.city.160k.bmodel
  const char *keys =
      "{bmodel | ../../models/BM1684/segformer.b0.512x1024.city.160k_fp32_1b.bmodel | bmodel file path}"
      "{dev_id | 0 | TPU device id}"
      "{input | ../../datasets/cityscapes_small | input path, images direction or video file path}"
      "{palette | cityscapes | Color palette used for segmentation map}"
      "{help | 0 | print help information.}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help"))
  {
    parser.printMessage();
    return 0;
  }

  // parms parse
  string bmodel_file = parser.get<string>("bmodel");
  string input = parser.get<string>("input");
  string palette = parser.get<string>("palette");
  int dev_id = parser.get<int>("dev_id");

  // check params
  struct stat info;
  cout << "nihao" << endl;
  if (stat(bmodel_file.c_str(), &info) != 0)
  {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }
  if (stat(input.c_str(), &info) != 0)
  {
    cout << "Cannot find input path." << endl;
    exit(1);
  }
  // creat handle
  BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
  cout << "set device id: " << dev_id << endl;
  bm_handle_t h = handle->handle();

  // load bmodel
  shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());

  // initialize net
  SegFormer segformer(bm_ctx);
  CV_Assert(0 == segformer.Init());

  segformer.palette = palette;
  // profiling
  TimeStamp segformer_ts;
  TimeStamp *ts = &segformer_ts;
  segformer.enableProfile(&segformer_ts);

  // get batch_size
  int batch_size = segformer.batch_size();

  // create save path
  if (access("results", 0) != F_OK)
    mkdir("results", S_IRWXU);

  // test images
  if (info.st_mode & S_IFDIR)
  {
    // 设置 data
    data["data_root"] = input;
    // 构建 img_dir 和 ann_dir
    string img_dir = data["data_root"] + "/" + data["img_dir"];
    string ann_dir = data["data_root"] + "/" + data["ann_dir"];

    if (access("results/images", 0) != F_OK)
      mkdir("results/images", S_IRWXU);

    std::string res_dir = "../../datasets/result_cl";
    if (access(res_dir.c_str(), 0) != F_OK)
      mkdir(res_dir.c_str(), S_IRWXU);

    // get files
    std::vector<std::string> files_vector;
    readFilesRecursive(img_dir, files_vector);

    // 初始化batch参数
    vector<bm_image> batch_imgs;
    vector<string> batch_names;
    int cn = files_vector.size();
    int id = 0;
    json results_json;
    results_json["data_root"] = removePrefix(data["data_root"], "../../");
    results_json["img_dir"] = removePrefix(img_dir, "../../");
    results_json["ann_dir"] = removePrefix(ann_dir, "../../");

    for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++){
      string img_file = *iter;
      cout << id << "/" << cn << ", img_file: " << img_file << endl;
      ts->save("decode time");
      bm_image bmimg;
      picDec(h, img_file.c_str(), bmimg);
      ts->save("decode time");
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(bmimg);
      batch_names.push_back(img_name);
      
      // 判断是不是最后一张
      iter++;
      bool end_flag = (iter == files_vector.end());
      iter--;

     if((batch_imgs.size() == batch_size || end_flag) &&!batch_imgs.empty()){

        // If the number of images is less than the batch size, fill the batch with the last image
        int recode_size=batch_imgs.size();
        while (batch_imgs.size() < batch_size) {
          batch_imgs.push_back(batch_imgs.back());
          batch_names.push_back(batch_names.back());
        }

        std::vector<std::vector<std::vector<int32_t>>> results_data;
        // predict
        CV_Assert(0 == segformer.Detect(batch_imgs, results_data));
        for (int i = 0; i < recode_size; i++){
          if (batch_imgs[i].image_format != 0)
          {
            id++;
            bm_image frame;
            bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width, FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
            bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
            bm_image_destroy(batch_imgs[i]);
            batch_imgs[i] = frame;
          }

          std::vector<std::vector<int32_t>> result_data = results_data[i];
          // 转换为灰度图像
          cv::Mat gray_image(result_data.size(), result_data[0].size(), CV_8UC1);
          for (int i = 0; i < gray_image.rows; i++){
            for (int j = 0; j < gray_image.cols; j++)
            {
              gray_image.at<uchar>(i, j) = result_data[i][j];
            }
          }

          // 文件路径对齐
          vector<string> res_file_paths;
          json results;

          // 获取没有扩展名的文件名
          size_t dot_pos = batch_names[i].find_last_of(".");
          std::string filename_no_ext = batch_names[i].substr(0, dot_pos);

          // 构造保存的文件名
          std::string res_file = filename_no_ext + ".png";
          std::string res_file_path = res_dir + "/" + res_file;
          res_file_paths.push_back(res_file_path);

          // 计算相对路径
          std::string relative_path = getRelativePath(img_file, img_dir);
          results["filename"] = relative_path;

          // 替换后的路径
          std::string seg_map = replaceSuffix(relative_path, data["img_suffix"], data["seg_map_suffix"]);
          json seg;
          seg["seg_map"] = seg_map;
          results["ann"] = seg;
  
          cv::imwrite(res_file_paths[i], gray_image);
          results["res"] = removePrefix(res_file_paths[i], "../../");
          cout << "write gray image" << endl;

          results_json["img_info"].push_back(results);

          // save image
          void *jpeg_data = NULL;
          size_t out_size = 0;
          int ret = bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data, &out_size);
          if (ret == BM_SUCCESS)
          {
            string img_file = "results/images/" + batch_names[i];
            FILE *fp = fopen(img_file.c_str(), "wb");
            fwrite(jpeg_data, out_size, 1, fp);
            fclose(fp);
          }
          free(jpeg_data);
          bm_image_destroy(batch_imgs[i]);
        }

        batch_imgs.clear();
        batch_names.clear();
      }
    }
    // save results
    size_t index = input.rfind("/");
    if (index == input.length() - 1)
    {
      input = input.substr(0, input.length() - 1);
      index = input.rfind("/");
    }
    string dataset_name = input.substr(index + 1);
    index = bmodel_file.rfind("/");
    string model_name = bmodel_file.substr(index + 1);
    string json_file = "results/" + model_name + "_" + dataset_name + "_bmcv_cpp" + "_result.json";
    cout << "================" << endl;
    cout << "result saved in " << json_file << endl;
    ofstream(json_file) << std::setw(4) << results_json;
  }
  // test video
  else
  {
    if (access("results/video", 0) != F_OK)
      mkdir("results/video", S_IRWXU);
    VideoDecFFM decoder;
    decoder.openDec(&h, input.c_str());
    std::vector<bm_image> batch_imgs;
    std::vector<std::string> batch_names;
    int id = 0;
    bool endFlag = false;
    while (!endFlag) {
      ts->save("decode time");
      bm_image* img = decoder.grab();
      ts->save("decode time");
      if (!img) {
        endFlag = true;
      } else {
        batch_imgs.push_back(*img);
        delete img;
        img = nullptr;
      }
      if (batch_imgs.size() == batch_size || (endFlag && batch_imgs.size())) {

        // If the number of images is less than the batch size, fill the batch with the last image
        int recode_size=batch_imgs.size();
        while (batch_imgs.size() < batch_size) {
          batch_imgs.push_back(batch_imgs.back());
          batch_names.push_back(batch_names.back());
        }

        std::vector<std::vector<std::vector<int32_t>>> result_datas;
        CV_Assert(0 == segformer.Detect(batch_imgs, result_datas));
        for (int i = 0; i < recode_size; i++)
        {
          id++;
          batch_names.push_back(std::to_string(id) + ".jpg");
          if (batch_imgs[i].image_format != 0)
          {
            bm_image frame;
            bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width, FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
            bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
            bm_image_destroy(batch_imgs[i]);
            batch_imgs[i] = frame;
          }

          string img_file = "results/video/" + to_string(id) + ".jpg";
          void *jpeg_data = NULL;
          size_t out_size = 0;
          int ret = bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data, &out_size);
          if (ret == BM_SUCCESS)
          {
            FILE *fp = fopen(img_file.c_str(), "wb");
            fwrite(jpeg_data, out_size, 1, fp);
            fclose(fp);
          }
          free(jpeg_data);
          bm_image_destroy(batch_imgs[i]);
        }
        batch_imgs.clear();
        batch_names.clear();
      }
    }
  }
  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  segformer_ts.calbr_basetime(base_time);
  segformer_ts.build_timeline("segformer test");
  segformer_ts.show_summary("segformer test");
  segformer_ts.clear();

  return 0;
}

std::string removePrefix(const std::string &path, const std::string &prefix)
{
  if (path.compare(0, prefix.length(), prefix) == 0)
  {
    return path.substr(prefix.length());
  }
  else
  {
    return path;
  }
}

void readFilesRecursive(const std::string &directory, std::vector<std::string> &files_vector)
{
  DIR *dir;
  struct dirent *entry;
  struct stat fileStat;

  dir = opendir(directory.c_str());
  if (dir == nullptr)
  {
    std::cerr << "Failed to open directory: " << directory << std::endl;
    return;
  }

  while ((entry = readdir(dir)) != nullptr)
  {
    std::string entryName = entry->d_name;
    std::string fullPath = directory + "/" + entryName;

    if (entryName != "." && entryName != "..")
    {
      if (stat(fullPath.c_str(), &fileStat) == 0)
      {
        if (S_ISDIR(fileStat.st_mode))
        {
          // 如果是子目录，递归读取
          readFilesRecursive(fullPath, files_vector);
        }
        else
        {
          // 如果是文件，检查文件扩展名为图片格式
          std::string extension = entryName.substr(entryName.find_last_of(".") + 1);
          if (extension == "jpg" || extension == "png" || extension == "bmp")
          {
            files_vector.push_back(fullPath);
          }
        }
      }
      else
      {
        std::cerr << "Failed to get file stat for: " << fullPath << std::endl;
      }
    }
  }

  closedir(dir);
}

std::string getRelativePath(const std::string &filepath, const std::string &basepath)
{
  size_t base_len = basepath.length();

  // 检查 basepath 结尾是否有斜杠
  if (base_len > 0 && (basepath[base_len - 1] == '/' || basepath[base_len - 1] == '\\'))
  {
    base_len--;
  }

  // 检查 filepath 是否以 basepath 开头
  if (filepath.compare(0, base_len, basepath) == 0)
  {
    // 返回 filepath 中去掉 basepath 部分的相对路径，并删除开头的斜杠
    return filepath.substr(base_len + 1);
  }
  else
  {
    // basepath 不是 filepath 的前缀，返回原始 filepath
    return filepath;
  }
}

std::string replaceSuffix(const std::string &relative_path, const std::string &img_suffix, const std::string &seg_map_suffix)
{
  std::string replaced_path = relative_path;

  // 找到最后一个出现的 img_suffix
  size_t suffix_pos = replaced_path.rfind(img_suffix);
  if (suffix_pos != std::string::npos)
  {
    // 替换 img_suffix 为 seg_map_suffix
    replaced_path.replace(suffix_pos, img_suffix.length(), seg_map_suffix);
  }

  return replaced_path;
}