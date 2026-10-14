//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "fear_tracker.hpp"

#include <sys/stat.h>
#include <sstream>

using json = nlohmann::json;

static void draw_bbox(cv::Mat& image, const Bbox& bbox, int width = 3) {
    cv::rectangle(image,
                  cv::Point((int)bbox.x, (int)bbox.y),
                  cv::Point((int)(bbox.x + bbox.w), (int)(bbox.y + bbox.h)),
                  cv::Scalar(0, 255, 0), width);
}

int main(int argc, char** argv) {
    const char* keys =
        "{bmodel        | ../../models/BM1684X/feartracker_fp16_1b.bmodel | bmodel file path }"
        "{input         | <none>                                         | input video file path }"
        "{initial_bbox  | <none>                                         | initial bbox: x,y,w,h (e.g. 163,53,45,174) }"
        "{output        | <none>                                         | output video file path (optional) }"
        "{dev_id        | 0                                              | TPU device id }"
        "{help h        |                                                | print help information }";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string bmodel_file = parser.get<std::string>("bmodel");
    std::string input_file  = parser.get<std::string>("input");
    std::string bbox_str    = parser.get<std::string>("initial_bbox");
    std::string output_file = parser.get<std::string>("output");
    int dev_id              = parser.get<int>("dev_id");

    if (input_file.empty() || bbox_str.empty()) {
        std::cerr << "Error: --input and --initial_bbox are required." << std::endl;
        parser.printMessage();
        return 1;
    }

    // Parse initial bbox
    Bbox init_bbox;
    {
        std::vector<float> vals;
        std::stringstream ss(bbox_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            vals.push_back(std::stof(tok));
        }
        if (vals.size() != 4) {
            std::cerr << "Error: --initial_bbox must be x,y,w,h (4 comma-separated ints)"
                      << std::endl;
            return 1;
        }
        init_bbox = {vals[0], vals[1], vals[2], vals[3]};
    }

    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        std::cerr << "Cannot find bmodel file: " << bmodel_file << std::endl;
        return 1;
    }
    if (stat(input_file.c_str(), &info) != 0) {
        std::cerr << "Cannot find input video: " << input_file << std::endl;
        return 1;
    }

    // Load device handle and bmodel
    std::shared_ptr<BMNNHandle> handle = std::make_shared<BMNNHandle>(dev_id);
    std::shared_ptr<BMNNContext> bm_ctx =
        std::make_shared<BMNNContext>(handle, bmodel_file.c_str());
    std::cout << "Set device id: " << dev_id << std::endl;

    FearTracker tracker(bm_ctx, dev_id);
    tracker.Init();

    TimeStamp tracker_ts;
    tracker.enableProfile(&tracker_ts);

    // Open video
    std::cout << "Loading video: " << input_file << std::endl;
    cv::VideoCapture cap(input_file);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << input_file << std::endl;
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30;

    // Read all frames, convert BGR→RGB
    std::vector<cv::Mat> frames;
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        frames.push_back(rgb);
    }
    cap.release();
    int total_frames = (int)frames.size();
    std::cout << "Video: " << total_frames << " frames, " << fps << " fps" << std::endl;

    // Initialize tracker on first frame
    std::cout << "Initial bbox: [" << init_bbox.x << ", " << init_bbox.y << ", "
              << init_bbox.w << ", " << init_bbox.h << "]" << std::endl;
    tracker.initialize(frames[0], init_bbox);

    // Track through remaining frames
    std::vector<Bbox> tracked_bboxes;
    tracked_bboxes.push_back(init_bbox);

    for (int i = 1; i < total_frames; i++) {
        Bbox pred = tracker.update(frames[i]);
        tracked_bboxes.push_back(pred);
        if (i % 50 == 0 || i == total_frames - 1) {
            std::cout << "  frame " << i << "/" << total_frames
                      << ": bbox=[" << (int)pred.x << ", " << (int)pred.y
                      << ", " << (int)pred.w << ", " << (int)pred.h << "]"
                      << std::endl;
        }
    }

    std::cout << "Tracked " << total_frames << " frames, "
              << tracked_bboxes.size() << " bboxes" << std::endl;

    // Performance summary
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    tracker_ts.calbr_basetime(base_time);
    tracker_ts.build_timeline("FearTracker");
    tracker_ts.show_summary("FearTracker");
    tracker_ts.clear();

    // Optional output video
    if (!output_file.empty()) {
        int h = frames[0].rows;
        int w = frames[0].cols;
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::VideoWriter writer(output_file, fourcc, fps, cv::Size(w, h));
        for (int i = 0; i < total_frames; i++) {
            cv::Mat vis;
            cv::cvtColor(frames[i], vis, cv::COLOR_RGB2BGR);
            draw_bbox(vis, tracked_bboxes[i]);
            writer.write(vis);
        }
        writer.release();
        std::cout << "Output saved to " << output_file << std::endl;
    }

    return 0;
}
