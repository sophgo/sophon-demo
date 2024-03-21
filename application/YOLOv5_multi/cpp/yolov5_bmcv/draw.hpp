//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-STREAM is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov5.hpp"


void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<std::string>& class_names)   // Draw the predicted bounding box
{
  //Draw a rectangle displaying the bounding box
  cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3);

  //Get the label for the class name and its confidence
  std::string label = cv::format("%.2f", conf);

  label = class_names[classId] + ":" + label;


  //Display the label at the top of the bounding box
  int baseLine;
  cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = std::max(top, labelSize.height);
  cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
}

void draw_json(std::shared_ptr<DataPost> box_data, std::shared_ptr<cv::Mat> image, std::string& image_name, std::vector<json>& results_json, std::vector<std::string>& class_names){
  // 示例用，只对channel0做操作
  int channel_id = box_data->channel_id;
  if (channel_id != 0){
    return;
  }

  int frame_id = box_data->frame_id;
  auto boxes = box_data->boxes;

  for (auto& bbox: boxes){
    drawPred(bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.width+bbox.x, bbox.height+bbox.y, *image, class_names);
  }

  // 图片保留json，测试精度
  std::vector<json> bboxes_json;
  for (auto& bbox: boxes){
    json bbox_json;
    bbox_json["category_id"] = bbox.class_id;
    bbox_json["score"] = bbox.score;
    bbox_json["bbox"] = {bbox.x, bbox.y, bbox.width, bbox.height};
    bboxes_json.push_back(bbox_json);
  }
  json res_json;
  res_json["image_name"] = image_name;
  res_json["bboxes"] = bboxes_json;
  results_json.push_back(res_json);

  
  std::string save_path = "results/images/" + image_name;
  cv::imwrite(save_path, *image);


}


// 使用yolov5后处理后的数据进行操作
void worker_draw(YOLOv5& yolov5, std::vector<json>& results_json, std::vector<std::string>& class_names){
  while (true){
    auto box_data = std::make_shared<DataPost>();
    auto origin_image = std::make_shared<cv::Mat>();
    std::string image_name;

    int ret = yolov5.get_post_data(box_data, origin_image, image_name);
    if (ret){
      return;
    }
    draw_json(box_data, origin_image, image_name, results_json, class_names);

  }
}

