//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <istream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "superpoint.hpp"
#include "superglue.hpp"
#include "json.hpp"
using json = nlohmann::json;

std::vector<float> tensor_to_vector(torch::Tensor& tensor) {
    std::vector<float> data(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return data;
}

cv::Mat tensor2mat(torch::Tensor tensor) {
  tensor = tensor.to(torch::kCPU).contiguous();
  cv::Mat mat(tensor.size(-2), tensor.size(-1), CV_32F);
  std::memcpy((void *)mat.data, tensor.data_ptr(), sizeof(float) * tensor.numel());
  return mat;
}

cv::Mat make_matching_plot_fast(const cv::Mat& imgmat0, const cv::Mat& imgmat1,
                                const torch::Tensor &kpts0, const torch::Tensor &kpts1,
                                const torch::Tensor &mkpts0, const torch::Tensor &mkpts1,
                                const torch::Tensor &confidence, bool show_keypoints = true,
                                int margin = 10) {

  if (show_keypoints) {
    const cv::Scalar white(255, 255, 255);
    const cv::Scalar black(0, 0, 0);
    for (int i = 0; i < kpts0.size(0); ++i) {
      auto kp = kpts0[i];
      cv::Point pt(std::lround(kp[0].item<float>()), std::lround(kp[1].item<float>()));
      cv::circle(imgmat0, pt, 2, black, -1, cv::LINE_AA);
      cv::circle(imgmat0, pt, 1, white, -1, cv::LINE_AA);
    }
    for (int i = 0; i < kpts1.size(0); ++i) {
      auto kp = kpts1[i];
      cv::Point pt(std::lround(kp[0].item<float>()), std::lround(kp[1].item<float>()));
      cv::circle(imgmat1, pt, 2, black, -1, cv::LINE_AA);
      cv::circle(imgmat1, pt, 1, white, -1, cv::LINE_AA);
    }
  }

  int H0 = imgmat0.rows, W0 = imgmat0.cols;
  int H1 = imgmat1.rows, W1 = imgmat1.cols;
  int H = std::max(H0, H1), W = W0 + W1 + margin;

  cv::Mat out = 255 * cv::Mat::ones(H, W, CV_8U);
  imgmat0.copyTo(out.rowRange(0, H0).colRange(0, W0));
  imgmat1.copyTo(out.rowRange(0, H1).colRange(W0 + margin, W));
  cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);

  // Apply colormap to confidences
  cv::Mat conf_mat = tensor2mat(confidence.unsqueeze(0));
  conf_mat.convertTo(conf_mat, CV_8U, 255.0f);
  cv::Mat colors;
  cv::applyColorMap(conf_mat, colors, cv::COLORMAP_JET);

  int n = std::min(mkpts0.size(0), mkpts1.size(0));
  for (int i = 0; i < n; ++i) {
    auto kp0 = mkpts0[i];
    auto kp1 = mkpts1[i];
    cv::Point pt0(std::lround(kp0[0].item<float>()), std::lround(kp0[1].item<float>()));
    cv::Point pt1(std::lround(kp1[0].item<float>()), std::lround(kp1[1].item<float>()));
    auto c = colors.at<cv::Vec3b>({i, 0});
    cv::line(out, pt0, {pt1.x + margin + W0, pt1.y}, c, 1, cv::LINE_AA);
    // display line end-points as circles
    cv::circle(out, pt0, 2, c, -1, cv::LINE_AA);
    cv::circle(out, {pt1.x + margin + W0, pt1.y}, 2, c, -1, cv::LINE_AA);
  }

  return out;
}

int main(int argc, char* argv[]) {
    std::cout.setf(std::ios::fixed);
    // get params
    const char* keys =
        "{bmodel_superpoint | ../../models/BM1688/superpoint_fp32_1b.bmodel | Path to bmodel.}"
        "{bmodel_superglue | ../../models/BM1688/superglue_fp32_1b_1024.bmodel | Path to bmodel.}"
        "{nms_radius | 4 | Nms radius.}"
        "{keypoint_thresh | 0.0002 | Keypoint threshold.}"
        "{max_keypoint_size | 1024 | Max keypoint size.}"
        "{matching_thresh | 0.002 | Matching threshold.}"
        "{dev_id | 0 | TPU device id.}"
        "{help | 0 | Print help information.}"
        "{input_pairs | ../../datasets/scannet_sample_pairs_with_gt.txt | Path to the list of image pairs.}"
        "{input_dir | ../../datasets/scannet_sample_images | Path to the directory that contains the images.}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    std::string bmodel_superpoint = parser.get<std::string>("bmodel_superpoint");
    std::string bmodel_superglue = parser.get<std::string>("bmodel_superglue");
    std::string input_pairs = parser.get<std::string>("input_pairs");
    std::string input_dir = parser.get<std::string>("input_dir");
    
    int dev_id = parser.get<int>("dev_id");
    int nms_radius = parser.get<int>("nms_radius");
    float keypoint_thresh = parser.get<float>("keypoint_thresh");
    int max_keypoint_size = parser.get<int>("max_keypoint_size");
    float matching_thresh = parser.get<float>("matching_thresh");
    // check params
    struct stat info;
    if (stat(bmodel_superpoint.c_str(), &info) != 0) {
        std::cout << "Cannot find valid model file." << std::endl;
        exit(1);
    }

    if ((stat(input_pairs.c_str(), &info) != 0) || (stat(input_dir.c_str(), &info) != 0)) {
        std::cout << "Cannot find both input_pairs or input_dir path." << std::endl;
        exit(1);
    }

    TimeStamp ts;
    SuperGlue superglue(bmodel_superglue, dev_id);
    superglue.ts = &ts;
    SuperPoint superpoint(bmodel_superpoint, dev_id);
    superpoint.ts = &ts;
    superpoint.set_nms_radius(nms_radius);
    superpoint.set_keypoint_threshold(keypoint_thresh);
    superpoint.set_max_keypoint_size(max_keypoint_size);
    // creat save path
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);

    json results_json;
    std::ifstream file(input_pairs);
    std::string line;
    if(file.is_open()){
        while(std::getline(file, line)){
            std::istringstream iss(line);
            std::string anchor_image, match_image;
            iss >> anchor_image >> match_image;
            anchor_image = input_dir + '/' + anchor_image;
            match_image = input_dir + '/' + match_image;
            std::cout<<"anchor: "<<anchor_image<<"; match:"<<match_image<<std::endl;

            // get anchor info
            size_t anchor_index = anchor_image.rfind("/");
            std::string anchor_img_name = anchor_image.substr(anchor_index + 1);
            anchor_index = anchor_img_name.rfind(".");
            anchor_img_name = anchor_img_name.substr(0, anchor_index);
            ts.save("decode time");
            cv::Mat anchor_mat = cv::imread(anchor_image, cv::IMREAD_GRAYSCALE, dev_id);
            ts.save("decode time");
            bm_image anchor_bmimg;
            cv::bmcv::toBMI(anchor_mat, &anchor_bmimg);
            torch::Tensor anchor_keypoints, anchor_scores, anchor_descriptors;
            ts.save("superpoint time");
            assert(0 == superpoint.detect(anchor_bmimg, anchor_keypoints, anchor_scores, anchor_descriptors));
            ts.save("superpoint time");
            float anchor_x_scale = anchor_mat.cols / (float)superpoint.get_network_input_w();
            float anchor_y_scale = anchor_mat.rows / (float)superpoint.get_network_input_h();

            // auto anchor_kpt_accessor = anchor_keypoints.accessor<float, 2>();
            // for(int i = 0; i < anchor_keypoints.size(0); i++){
            //     cv::circle(anchor_mat, cv::Point(int(anchor_kpt_accessor[i][0]*anchor_x_scale), int(anchor_kpt_accessor[i][1]*anchor_y_scale)), 1, cv::Scalar(0, (255), 0), 2);
            // }
            // cv::imwrite("anchor.jpg", anchor_mat);

            bm_image bmimg;
            ts.save("decode time");
            cv::Mat cvmat = cv::imread(match_image, cv::IMREAD_GRAYSCALE, dev_id);
            ts.save("decode time");
            cv::bmcv::toBMI(cvmat, &bmimg);
            size_t index = match_image.rfind("/");
            std::string img_name = match_image.substr(index + 1);

            ts.save("superpoint time");
            torch::Tensor keypoints, scores, descriptors;
            assert(0 == superpoint.detect(bmimg, keypoints, scores, descriptors));
            ts.save("superpoint time");
                
            float x_scale = cvmat.cols / (float)superpoint.get_network_input_w();
            float y_scale = cvmat.rows / (float)superpoint.get_network_input_h();

            // cv::imwrite("results/images/"+img_name, cvmat);
            torch::Tensor matchings0;
            torch::Tensor matchings0_score;
            ts.save("superglue time");
            superglue.detect(anchor_keypoints, anchor_scores, anchor_descriptors, keypoints, scores, descriptors, matchings0, matchings0_score);
            matchings0 = matchings0.index({torch::indexing::Slice(0, anchor_keypoints.size(0))});
            matchings0_score = matchings0_score.index({torch::indexing::Slice(0, anchor_keypoints.size(0))});
            ts.save("superglue time");

            ts.save("visualization time");
            if(cvmat.cols == anchor_mat.cols && cvmat.rows == anchor_mat.rows){
                torch::Tensor valid = at::nonzero((matchings0 > -1) & (matchings0 < keypoints.size(0)) & (matchings0_score > matching_thresh)).squeeze();
                torch::Tensor confidence = matchings0_score.index_select(0, valid);
                torch::Tensor mkpts0 = anchor_keypoints.index_select(0, valid);
                torch::Tensor mkpts1 = keypoints.index_select(0, matchings0.index_select(0, valid).to(torch::kInt32));
                cv::Mat anchor_mat_resized;
                cv::resize(anchor_mat, anchor_mat_resized, cv::Size(superpoint.get_network_input_w(), superpoint.get_network_input_h()));
                cv::Mat cvmat_resized;
                cv::resize(cvmat, cvmat_resized, cv::Size(superpoint.get_network_input_w(), superpoint.get_network_input_h()));
                        
                std::cout<<"Anchor keypoint size: "<<anchor_keypoints.size(0)<<"; image for match keypoint size: "<<keypoints.size(0)<<std::endl;
                if(valid.dim() == 0 || valid.numel() == 0){
                    std::cout<<"No matches, skip drawing."<<std::endl;
                    continue;
                }
                std::cout<<"Matches: "<<valid.numel()<<std::endl;
                cv::Mat combined_mat =
                    make_matching_plot_fast(anchor_mat_resized, cvmat_resized, anchor_keypoints, keypoints, mkpts0, mkpts1, confidence);
                cv::putText(combined_mat, 
                            cv::format("keypoints:%ldvs%ld", anchor_keypoints.size(0), keypoints.size(0)), 
                            cv::Point(10, 20), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 1);
                cv::putText(combined_mat, 
                            cv::format("matches:%ld", valid.size(0)), 
                            cv::Point(10, 40), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 1);
                cv::imwrite("results/images/"+anchor_img_name+"_"+img_name, combined_mat);
            }else{
                std::cerr << "Not support combining two mat with different shapes now." << std::endl;
            }
            ts.save("visualization time");

            //save result json
            json match_results;
            match_results["keypoints0"] = tensor_to_vector(anchor_keypoints);
            match_results["keypoints1"] = tensor_to_vector(keypoints);
            match_results["scores0"] = tensor_to_vector(anchor_scores);
            match_results["scores1"] = tensor_to_vector(scores);
            match_results["matchings0"] = tensor_to_vector(matchings0);
            match_results["matchings0_score"] = tensor_to_vector(matchings0_score);
            match_results["scales0"] = {anchor_x_scale, anchor_y_scale};
            match_results["scales1"] = {x_scale, y_scale};
            results_json[anchor_img_name+"_"+img_name] = match_results;
            bm_image_destroy(bmimg);
        }
    }
    std::string json_file = "results/result.json";
    std::ofstream(json_file) << std::setw(4) << results_json;

    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    ts.calbr_basetime(base_time);
    ts.build_timeline("superpoint test");
    ts.show_summary("superpoint test");
    ts.clear();

    return 0;
}
