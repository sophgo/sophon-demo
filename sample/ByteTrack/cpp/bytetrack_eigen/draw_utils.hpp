#include <iostream>
#include <vector>

#include "bm_wrapper.hpp"
#include "opencv2/opencv.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},    {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {170, 255, 0},  {85, 255, 0},    {0, 255, 0},     {0, 255, 85},
    {0, 255, 170},  {0, 255, 255},   {0, 170, 255},   {0, 85, 255},
    {0, 0, 255},    {85, 0, 255},    {170, 0, 255},   {255, 0, 255},
    {255, 0, 170},  {255, 0, 85},    {255, 0, 0},     {255, 0, 255},
    {255, 85, 255}, {255, 170, 255}, {255, 255, 255}, {170, 255, 255},
    {85, 255, 255}};

void draw_bmcv(bm_handle_t& handle, int trackId, int classId, float conf,
               int left, int top, int width, int height, bm_image& frame,
               bool put_text_flag)  // Draw the predicted bounding box
{
  int colors_num = colors.size();
  // Draw a rectangle displaying the bounding box
  bmcv_rect_t rect;
  rect.start_x = MIN(MAX(left, 0), frame.width);
  rect.start_y = MIN(MAX(top, 0), frame.height);
  rect.crop_w = MAX(MIN(width, frame.width - left), 0);
  rect.crop_h = MAX(MIN(height, frame.height - top), 0);
  // std::cout << rect.start_x << "," << rect.start_y << "," << rect.crop_w <<
  // "," << rect.crop_h << std::endl;
  bmcv_image_draw_rectangle(
      handle, frame, 1, &rect, 1.5, colors[classId % colors_num][0],
      colors[classId % colors_num][1], colors[classId % colors_num][2]);

  if (put_text_flag) {
    std::string label =
        std::to_string(classId) + ", track:" + std::to_string(trackId);
    bmcv_point_t org = {left, top - 5};
    bmcv_color_t color = {colors[classId % colors_num][0],
                          colors[classId % colors_num][1],
                          colors[classId % colors_num][2]};
    int thickness = 2;
    float fontScale = 1.5;
    if (BM_SUCCESS != bmcv_image_put_text(handle, frame, label.c_str(), org,
                                          color, fontScale, thickness)) {
      std::cout << "bmcv put text error !!!" << std::endl;
    }
  }
}

void draw_opencv(int trackId, int classId, float conf, int left, int top,
                 int right, int bottom,
                 cv::Mat& frame)  // Draw the predicted bounding box
{
  // Draw a rectangle displaying the bounding box
  int colors_num = colors.size();
  cv::Scalar color(colors[classId % colors_num][0],
                   colors[classId % colors_num][1],
                   colors[classId % colors_num][2]);
  cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), color,
                2.5);

  std::string label = cv::format("cls=%d,Id=%d", classId, trackId);

  // Display the label at the top of the bounding box
  int baseLine;
  cv::Size labelSize =
      getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = std::max(top, labelSize.height);
  // rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left
  // + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
  cv::putText(frame, label, cv::Point(left, top - 5), cv::FONT_HERSHEY_SIMPLEX,
              1.5, color, 2);
}