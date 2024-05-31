#include <iostream>
#include <vector>

#include "bm_wrapper.hpp"
#include "opencv2/opencv.hpp"
#include "p2pnet.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1

void draw_bmcv(bm_handle_t h, const std::vector<std::vector<PPoint>>& points,
               std::vector<bm_image>& batch_imgs,
               const std::vector<std::string>& batch_names,
               std::string save_folder) {
  for (size_t i = 0; i < batch_imgs.size(); i++) {
    bm_image img = batch_imgs[i];
    for (size_t j = 0; j < points[i].size(); j++) {
      bmcv_rect_t rect;
      rect.start_x = MIN(MAX(int(points[i][j].x), 0), img.width);
      rect.start_y = MIN(MAX(int(points[i][j].y), 0), img.height);
      rect.crop_w = MAX(MIN(2, img.width - int(points[i][j].x)), 0);
      rect.crop_h = MAX(MIN(2, img.height - int(points[i][j].y)), 0);
      bmcv_image_draw_rectangle(h, img, 1, &rect, 3, 255, 0, 0);
    }
    // save image
    std::string save_name = save_folder + "/" + batch_names[i];
    void* jpeg_data = NULL;
    size_t out_size = 0;
    int ret = bmcv_image_jpeg_enc(h, 1, (bm_image*)&batch_imgs[i], &jpeg_data,
                                  &out_size);
    if (ret == BM_SUCCESS) {
      FILE* fp = fopen(save_name.c_str(), "wb");
      fwrite(jpeg_data, out_size, 1, fp);
      fclose(fp);
    }
    free(jpeg_data);
    bm_image_destroy(batch_imgs[i]);

    // write txt
    std::string txt_name = batch_names[i];
    txt_name.replace(batch_names[i].find(".jpg"), 4, ".txt");
    std::string txt_file = save_folder + "/" + txt_name;
    std::ofstream out;
    out.open(txt_file, std::ios::out | std::ios::trunc);
    for (size_t j = 0; j < points[i].size(); j++) {
      out << points[i][j].x << " " << points[i][j].y << "\n";
    }
    out.close();
  }
}

void draw_opencv(const std::vector<std::vector<PPoint>>& points,
                 const std::vector<cv::Mat>& batch_imgs,
                 const std::vector<std::string>& batch_names,
                 std::string save_folder) {
  for (size_t i = 0; i < batch_imgs.size(); i++) {
    cv::Mat img = batch_imgs[i];
    for (size_t j = 0; j < points[i].size(); j++) {
      auto px = MIN(MAX(int(points[i][j].x), 0), img.rows);
      auto py = MIN(MAX(int(points[i][j].y), 0), img.cols);
      cv::Point p(px, py);
      cv::circle(img, p, 2, cv::Scalar(0, 0, 255), -1);
    }

    // save image
    std::string save_name = save_folder + "/" + batch_names[i];
    cv::imwrite(save_name, img);

    // write txt
    std::string txt_name = batch_names[i];
    txt_name.replace(batch_names[i].find(".jpg"), 4, ".txt");
    std::string txt_file = save_folder + "/" + txt_name;
    std::ofstream out;
    out.open(txt_file, std::ios::out | std::ios::trunc);
    for (size_t j = 0; j < points[i].size(); j++) {
      out << points[i][j].x << " " << points[i][j].y << "\n";
    }
    out.close();
  }
}