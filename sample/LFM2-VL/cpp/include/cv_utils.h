/*****************************************************************************
 *
 *    Copyright (c) 2016-2026 by Sophgo Technologies Inc. All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Sophgo Technologies Inc. This is proprietary information owned by
 *    Sophgo Technologies Inc. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Sophgo Technologies Inc.
 *
 *****************************************************************************/
#ifndef CV_UTILS_H_
#define CV_UTILS_H_

#include "PillowResize.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Config {
  int SEQLEN;
  int MAX_PREFILL_LENGTH;
  int MAX_INPUT_LENGTH;
  int total_length;

  // vit config
  int max_pos;
  int MAX_PATCHES;
  int MAX_PIXELS;
  int MIN_PIXELS;
  std::vector<int> nchw;
  int media_offset;
  int media_size;
  int spatial_merge_size;
  int patch_size;
  int temporal_patch_size;
  float video_ratio;
  float video_fps;
};

//===------------------------------------------------------------===//
// Resize
//===------------------------------------------------------------===//
const int IMAGE_FACTOR = 28;
const int MAX_RATIO = 200;

int round_by_factor(int number, int factor) {
  return static_cast<int>(std::round(static_cast<double>(number) / factor)) *
         factor;
}

int ceil_by_factor(double number, int factor) {
  return static_cast<int>(std::ceil(number / factor)) * factor;
}

int floor_by_factor(double number, int factor) {
  return static_cast<int>(std::floor(number / factor)) * factor;
}

void tile(const std::vector<float> &x, std::vector<float> &y, int n) {
  for (int i = 0; i < n; ++i) {
    std::copy(x.begin(), x.end(), y.begin() + i * x.size());
  }
}

void flatten(const std::vector<std::vector<float>> &x, std::vector<float> &y) {
  for (size_t i = 0; i < x.size(); ++i) {
    std::copy(x[i].begin(), x[i].end(), y.begin() + x[i].size());
  }
}

// refs:transformers/models/lfm2_vl/image_processing_lfm2_vl_fast.py
// Correspondans to Python: convert_image_to_patches with permute(0, 2, 4, 3, 5, 1)
// Input: planar CHW format (all R, all G, all B)
// Output: (batch, num_patches_height * num_patches_width, patch_size * patch_size * num_channels)
void rearrange_patches(const std::vector<float> &image, std::vector<float> &out,
                       const Config &config) {
  int num_channels = config.nchw[1];
  int image_height = config.nchw[2];
  int image_width = config.nchw[3];
  int num_patches_height = image_height / config.patch_size;
  int num_patches_width = image_width / config.patch_size;
  int image_size = image.size();
  out.assign(image_size, 0);
  
  // Iterate over output positions
  for (int out_idx = 0; out_idx < image_size; ++out_idx) {
    // Decode output index: (ph, pw, ps1, ps0, c)
    int idx = out_idx;
    
    int c = idx % num_channels;
    idx /= num_channels;
    
    int ps0 = idx % config.patch_size;
    idx /= config.patch_size;
    
    int ps1 = idx % config.patch_size;
    idx /= config.patch_size;
    
    int pw = idx % num_patches_width;
    idx /= num_patches_width;
    
    int ph = idx % num_patches_height;
    // idx /= num_patches_height; (batch_idx = 0 for single image)
    
    // Calculate input index from planar CHW format
    // Input: (c, h, w) where h = ph * patch_size + ps1, w = pw * patch_size + ps0
    int h = ph * config.patch_size + ps1;
    int w = pw * config.patch_size + ps0;
    
    int in_idx = c * image_height * image_width + h * image_width + w;
    
    out[out_idx] = image[in_idx];
  }
}

cv::Mat convert_to_rgb(const cv::Mat &input_image) {
  CV_Assert(input_image.depth() == CV_8U);

  cv::Mat output_image;

  switch (input_image.channels()) {
  case 4: {
    std::vector<cv::Mat> bgra_channels;
    cv::split(input_image, bgra_channels);

    cv::Mat alpha;
    bgra_channels[3].convertTo(alpha, CV_32FC1, 1.0 / 255.0);

    cv::Mat white_bg(input_image.size(), CV_32FC3,
                     cv::Scalar(1.0f, 1.0f, 1.0f));

    std::vector<cv::Mat> blended_channels;
    for (int i = 0; i < 3; ++i) {
      cv::Mat channel;
      bgra_channels[i].convertTo(channel, CV_32FC1, 1.0 / 255.0);
      cv::Mat blended = channel.mul(alpha) + white_bg.col(i).mul(1.0 - alpha);
      blended_channels.push_back(blended * 255.0);
    }

    cv::merge(blended_channels, output_image);
    output_image.convertTo(output_image, CV_8UC3);

    // BGR -> RGB
    cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
    break;
  }

  case 1: { // Gray
    cv::cvtColor(input_image, output_image, cv::COLOR_GRAY2RGB);
    break;
  }

  case 3: { // BGR
    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2RGB);
    break;
  }

  default:
    CV_Error(cv::Error::StsBadArg, "Unsupported channel number");
  }

  return output_image;
}

void bicubic_resize(const cv::Mat &image, std::vector<float> &image_new,
                    int resized_height, int resized_width,
                    const std::vector<float> &image_mean,
                    const std::vector<float> &image_std) {
  auto rgb_image = convert_to_rgb(image);
  auto resized_image =
      PillowResize::resize(rgb_image, cv::Size(resized_width, resized_height),
                           PillowResize::INTERPOLATION_BICUBIC);
  // rescale
  resized_image.convertTo(resized_image, CV_32FC3, 0.00392156862745098, 0);

  // split channel
  std::vector<cv::Mat> rgbChannels(3);
  cv::split(resized_image, rgbChannels);

  // normaliza
  for (int c = 0; c < 3; c++) {
    rgbChannels[c] = (rgbChannels[c] - image_mean[c]) / image_std[c];
  }

  // combine channel
  cv::Mat normalized_image;
  cv::merge(rgbChannels, normalized_image);

  // convert to 1D
  image_new.reserve(resized_height * resized_width * 3);
  std::vector<cv::Mat> chw(3);
  cv::split(normalized_image, chw);
  for (int c = 0; c < 3; c++) {
    image_new.insert(image_new.end(), (float *)chw[c].datastart,
                     (float *)chw[c].dataend);
  }
}

bool process_image(std::vector<float> &data, const std::string &media_path,
                   Config &config) {
  cv::Mat image = cv::imread(media_path);
  if (image.empty()) {
    std::cerr << "Error: Unable to open image file: " << media_path
              << std::endl;
    return false;
  }

  // int width = image.cols;
  // int height = image.rows;
  std::vector<float> image_mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> image_std = {0.5f, 0.5f, 0.5f};

  int resized_height = 512;
  int resized_width = 512;
  std::vector<float> image_new;
  bicubic_resize(image, image_new, resized_height, resized_width, image_mean,
                 image_std);
  config.nchw = {1, 3, resized_height, resized_width};
  rearrange_patches(image_new, data, config);
  return true;
}

#endif // CV_UTILS_H_
