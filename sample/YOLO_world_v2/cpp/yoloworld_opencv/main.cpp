//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yoloworld.hpp"

#include <sys/stat.h>
#include <dirent.h>
#include <sstream>
#include <algorithm>

// Default COCO 80 class names (fallback when --class_names_file not given)
static const std::vector<std::string> DEFAULT_COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

static bool has_suffix(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool is_image_file(const std::string& name) {
    return has_suffix(name, ".jpg") || has_suffix(name, ".jpeg") ||
           has_suffix(name, ".png")  || has_suffix(name, ".bmp")  ||
           has_suffix(name, ".webp");
}

int main(int argc, char** argv) {
    const char* keys =
        "{bmodel          | ../../models/BM1684X/yoloworld_v2_fp16_1b.bmodel | bmodel file path }"
        "{txt_feats       | ../../models/BM1684X/coco80_txt_feats.npy        | precomputed txt_feats .npy [1,N,512] }"
        "{class_names_file| ../../datasets/coco.names                         | class names file }"
        "{input           | <none>                                            | input: image dir or video file }"
        "{output          | <none>                                            | output video/image path (optional) }"
        "{conf_thresh     | 0.25                                              | confidence threshold }"
        "{nms_thresh      | 0.7                                               | nms threshold }"
        "{dev_id          | 0                                                 | TPU device id }"
        "{help h          |                                                   | print help information }";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string bmodel_file       = parser.get<std::string>("bmodel");
    std::string txt_feats_file    = parser.get<std::string>("txt_feats");
    std::string class_names_file  = parser.get<std::string>("class_names_file");
    std::string input_file        = parser.get<std::string>("input");
    std::string output_file       = parser.get<std::string>("output");
    float conf_thresh             = parser.get<float>("conf_thresh");
    float nms_thresh              = parser.get<float>("nms_thresh");
    int dev_id                    = parser.get<int>("dev_id");

    if (input_file == "<none>") {
        std::cerr << "Error: --input is required." << std::endl;
        parser.printMessage();
        return 1;
    }

    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        std::cerr << "Cannot find bmodel file: " << bmodel_file << std::endl;
        return 1;
    }
    if (stat(txt_feats_file.c_str(), &info) != 0) {
        std::cerr << "Cannot find txt_feats file: " << txt_feats_file << std::endl;
        return 1;
    }
    if (stat(input_file.c_str(), &info) != 0) {
        std::cerr << "Cannot find input: " << input_file << std::endl;
        return 1;
    }

    // Load device handle and bmodel
    std::shared_ptr<BMNNHandle> handle = std::make_shared<BMNNHandle>(dev_id);
    std::shared_ptr<BMNNContext> bm_ctx =
        std::make_shared<BMNNContext>(handle, bmodel_file.c_str());
    std::cout << "Set device id: " << dev_id << std::endl;

    YoloWorld detector(bm_ctx, dev_id);
    detector.Init(conf_thresh, nms_thresh, class_names_file);

    // Load precomputed text embeddings (one-time, reused for all frames)
    if (detector.loadTxtFeats(txt_feats_file) != 0) {
        std::cerr << "Failed to load txt_feats" << std::endl;
        return 1;
    }

    TimeStamp detector_ts;
    detector.enableProfile(&detector_ts);

    bool is_dir = (stat(input_file.c_str(), &info) == 0 && S_ISDIR(info.st_mode));

    if (is_dir) {
        // ---- Image directory mode ----
        std::vector<std::string> img_files;
        DIR* dir = opendir(input_file.c_str());
        if (!dir) {
            std::cerr << "Cannot open directory: " << input_file << std::endl;
            return 1;
        }
        struct dirent* ent;
        while ((ent = readdir(dir)) != nullptr) {
            std::string name = ent->d_name;
            if (is_image_file(name))
                img_files.push_back(input_file + "/" + name);
        }
        closedir(dir);
        std::sort(img_files.begin(), img_files.end());
        std::cout << "Found " << img_files.size() << " images" << std::endl;

        std::string out_dir;
        if (!output_file.empty() && output_file != "<none>") {
            out_dir = output_file;
            // Create output directory if not exists
            std::string mkdir_cmd = "mkdir -p " + out_dir;
            system(mkdir_cmd.c_str());
        }

        int cn = 0;
        for (const auto& img_path : img_files) {
            cn++;
            cv::Mat frame = cv::imread(img_path);
            if (frame.empty()) {
                std::cerr << "Failed to read: " << img_path << std::endl;
                continue;
            }

            std::vector<YoloWorldBoxVec> results;
            detector.Detect({frame}, results);

            std::cout << "[" << cn << "/" << img_files.size() << "] "
                      << img_path << "  det nums: " << results[0].size()
                      << std::endl;

            if (!out_dir.empty()) {
                for (const auto& box : results[0])
                    detector.drawPred(frame, box);
                std::string base = img_path.substr(img_path.find_last_of('/') + 1);
                cv::imwrite(out_dir + "/" + base, frame);
            }
        }
    } else {
        // ---- Video file mode ----
        std::cout << "Loading video: " << input_file << std::endl;
        cv::VideoCapture cap(input_file);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video: " << input_file << std::endl;
            return 1;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30;
        int frame_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "Video: " << frame_w << "x" << frame_h << " @ " << fps
                  << " fps" << std::endl;

        cv::VideoWriter writer;
        if (!output_file.empty() && output_file != "<none>") {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer.open(output_file, fourcc, fps,
                        cv::Size(frame_w, frame_h));
            if (!writer.isOpened()) {
                std::cerr << "Warning: cannot open output video: "
                          << output_file << std::endl;
            }
        }

        int cn = 0;
        int total_dets = 0;
        while (true) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) break;
            cn++;

            std::vector<YoloWorldBoxVec> results;
            detector.Detect({frame}, results);
            total_dets += (int)results[0].size();

            if (cn % 50 == 0 || cn == 1) {
                std::cout << "  frame " << cn << ": det nums: "
                          << results[0].size() << std::endl;
            }

            if (writer.isOpened()) {
                for (const auto& box : results[0])
                    detector.drawPred(frame, box);
                writer.write(frame);
            }
        }
        cap.release();
        if (writer.isOpened()) {
            writer.release();
            std::cout << "Output saved to " << output_file << std::endl;
        }

        std::cout << "Processed " << cn << " frames, total detections: "
                  << total_dets << std::endl;
    }

    // Performance summary
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    detector_ts.calbr_basetime(base_time);
    detector_ts.build_timeline("YoloWorld");
    detector_ts.show_summary("YoloWorld");
    detector_ts.clear();

    return 0;
}
