//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "c3d.hpp"

using json = nlohmann::json;

void getAllFiles(std::string path, std::vector<std::string>& files) {
    DIR* dir;
    struct dirent* ptr;
    if ((dir = opendir(path.c_str())) == NULL) {
        perror("Open dri error...");
        exit(1);
    }
    while ((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
            continue;
        else if (ptr->d_type == 8)  // file
            files.push_back(path + "/" + ptr->d_name);
        else if (ptr->d_type == 10)  // link file
            continue;
        else if (ptr->d_type == 4) {
            // files.push_back(ptr->d_name);//dir
            getAllFiles(path + "/" + ptr->d_name, files);
        }
    }
    closedir(dir);
}

int main(int argc, char** argv) {
    /*
     * Custom configurations.
     */
    const char* keys =
        "{bmodel | ../../models/BM1684X/c3d_fp32_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}"
        "{input | ../../datasets/UCF_test_01 | UCF-101 style video directory}"
        "{classnames | ../../datasets/ucf_names.txt | UCF-101 class names}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    struct stat info;
    std::string input_url = parser.get<std::string>("input");
    if (stat(input_url.c_str(), &info) != 0) {
        std::cout << "Cannot find dataset path." << std::endl;
        exit(1);
    }
    if (!(info.st_mode & S_IFDIR)) {
        std::cout << "Invalid dataset path, need directory!" << std::endl;
        exit(1);
    }
    std::string bmodel_file = parser.get<std::string>("bmodel");
    if (stat(bmodel_file.c_str(), &info) != 0) {
        std::cout << "Cannot find valid model file." << std::endl;
        exit(1);
    }
    std::string ucf_names = parser.get<std::string>("classnames");
    if (stat(ucf_names.c_str(), &info) != 0) {
        std::cout << "Cannot find classnames file." << std::endl;
        exit(1);
    }
    int dev_id = parser.get<int>("dev_id");
    int step_len = 6;
    /*------------------------------------------------------
     * Inference flow.
     *------------------------------------------------------
     */

    // 1. Get device handle and load bmodel file.
    std::shared_ptr<BMNNHandle> handle = std::make_shared<BMNNHandle>(dev_id);
    std::shared_ptr<BMNNContext> bm_ctx = std::make_shared<BMNNContext>(handle, bmodel_file.c_str());
    std::cout << "Set device id: " << dev_id << " ." << std::endl;

    // 2. Initialize network.
    C3D c3d(bm_ctx, step_len, dev_id);
    c3d.Init();
    int batch_size = c3d.batch_size();

    // 3. Profile and save results
    TimeStamp c3d_ts;
    TimeStamp* ts = &c3d_ts;
    c3d.enableProfile(&c3d_ts);
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    json results_json;

    // 4. Data structures for inference.
    std::vector<std::string> batch_videos;
    // 5. Forward data to network, output detected object boxes.
    // get classes in dataset.
    std::vector<std::string> class_folders, class_names;
    std::ifstream ifs(ucf_names);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            class_names.push_back(line);
        }
    }
    std::vector<std::string> video_paths;
    getAllFiles(input_url, video_paths);
    std::sort(video_paths.begin(), video_paths.end());

    auto check_video_path = [](std::string video_path) {
        auto index = video_path.rfind('.');
        std::string postfix = video_path.substr(index + 1);
        std::vector<std::string> video_postfixes = {"mp4", "avi"};
        if (std::find(video_postfixes.begin(), video_postfixes.end(), postfix) != video_postfixes.end()) {
            std::cout << "Read video path: " << video_path << std::endl;
            return true;
        } else {
            std::cout << "skipping video path, please check your dataset!" << std::endl;
            return false;
        }
    };
    for (int j = 0; j < video_paths.size(); j++) {
        if (!check_video_path(video_paths[j]))
            continue;
        batch_videos.push_back(video_paths[j]);
        if (batch_videos.size() == batch_size) {
            std::vector<int> predict_ids;
            c3d.detect(batch_videos, predict_ids);
            for (int k = 0; k < predict_ids.size(); k++) {
                std::cout << "Predict: " << class_names[predict_ids[k]] << std::endl;
                size_t index = batch_videos[k].rfind("/");
                std::string video_name = batch_videos[k].substr(index + 1);
                results_json[video_name] = class_names[predict_ids[k]];
            }
            batch_videos.clear();
        }
    }
    if (batch_videos.size() > 0) {
        std::vector<int> predict_ids;
        c3d.detect(batch_videos, predict_ids);
        for (int k = 0; k < predict_ids.size(); k++) {
            std::cout << "Predict: " << class_names[predict_ids[k]] << std::endl;
            size_t index = batch_videos[k].rfind("/");
            std::string video_name = batch_videos[k].substr(index + 1);
            results_json[video_name] = class_names[predict_ids[k]];
        }
        batch_videos.clear();
    }
    size_t index = bmodel_file.rfind("/");
    std::string model_name = bmodel_file.substr(index + 1);
    std::string json_file = "results/" + model_name + "_opencv_cpp.json";
    std::cout << "================" << std::endl;
    std::cout << "result saved in " << json_file << std::endl;
    std::ofstream(json_file) << std::setw(4) << results_json;

    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    ts->calbr_basetime(base_time);
    ts->build_timeline("C3D detect");
    ts->show_summary("C3D detect");
    ts->clear();
    return 0;
}