//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "sam2.hpp"

#include <sys/stat.h>
#include <sstream>
#include <algorithm>
#include <vector>

// Parse a "x,y" or "x1,y1,x2,y2" string into integers.
static bool parsePointStr(const std::string& s, std::vector<int>& vals) {
    vals.clear();
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            vals.push_back(std::stoi(item));
        } catch (...) {
            return false;
        }
    }
    return vals.size() == 2 || vals.size() == 4;
}

int main(int argc, char** argv) {
    const char* keys =
        "{encoder_bmodel | ../../models/BM1684X/image_encoder/sam2_encoder_f16_1b.bmodel | encoder bmodel file path }"
        "{decoder_bmodel | ../../models/BM1684X/image_decoder/sam2_decoder_f16_1b.bmodel | decoder bmodel file path }"
        "{input_image    | ../../datasets/images/truck.jpg                          | input image path }"
        "{input_point    | 500,375                                                  | point x,y or box x1,y1,x2,y2 }"
        "{output         | <none>                                                   | output image path (optional) }"
        "{dev_id         | 0                                                        | TPU device id }"
        "{help h         |                                                          | print help information }";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string encoderBmodel = parser.get<std::string>("encoder_bmodel");
    std::string decoderBmodel = parser.get<std::string>("decoder_bmodel");
    std::string inputFile     = parser.get<std::string>("input_image");
    std::string inputPoint    = parser.get<std::string>("input_point");
    std::string outputFile     = parser.get<std::string>("output");
    int dev_id                 = parser.get<int>("dev_id");

    struct stat info;
    if (stat(encoderBmodel.c_str(), &info) != 0) {
        std::cerr << "Cannot find encoder bmodel: " << encoderBmodel << std::endl;
        return 1;
    }
    if (stat(decoderBmodel.c_str(), &info) != 0) {
        std::cerr << "Cannot find decoder bmodel: " << decoderBmodel << std::endl;
        return 1;
    }
    if (stat(inputFile.c_str(), &info) != 0) {
        std::cerr << "Cannot find input image: " << inputFile << std::endl;
        return 1;
    }

    std::vector<int> pointVals;
    if (!parsePointStr(inputPoint, pointVals)) {
        std::cerr << "Invalid --input_point: " << inputPoint
                  << " (expected x,y or x1,y1,x2,y2)" << std::endl;
        return 1;
    }
    bool isBox = (pointVals.size() == 4);

    // Load device handle and both bmodels
    std::shared_ptr<BMNNHandle> handle = std::make_shared<BMNNHandle>(dev_id);
    std::cout << "Set device id: " << dev_id << std::endl;

    Sam2 sam2(handle, encoderBmodel, decoderBmodel, dev_id);
    sam2.Init();

    TimeStamp sam2_ts;
    sam2.enableProfile(&sam2_ts);

    // Read image
    cv::Mat image = cv::imread(inputFile);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << inputFile << std::endl;
        return 1;
    }
    std::cout << "Input image: " << image.cols << "x" << image.rows << std::endl;

    // Run SAM2
    std::vector<Sam2Mask> masks;
    int ret;
    if (isBox) {
        ret = sam2.processBox(image, pointVals[0], pointVals[1],
                              pointVals[2], pointVals[3], masks);
    } else {
        ret = sam2.processPoint(image, pointVals[0], pointVals[1], masks);
    }
    if (ret != 0) {
        std::cerr << "SAM2 process failed" << std::endl;
        return 1;
    }

    // Print results
    std::cout << "Got " << masks.size() << " masks" << std::endl;
    for (size_t i = 0; i < masks.size(); i++) {
        int nPixels = cv::countNonZero(masks[i].mask);
        std::cout << "  Mask " << i << ": score=" << masks[i].score
                  << " pixels=" << nPixels << std::endl;
    }

    // Draw and save
    if (!outputFile.empty() && outputFile != "<none>") {
        // Draw all masks on the image with distinct colors
        const std::vector<cv::Scalar> colors = {
            cv::Scalar(30, 144, 255),   // dodger blue
            cv::Scalar(255, 0, 0),       // blue
            cv::Scalar(0, 255, 0),       // green
        };
        for (size_t i = 0; i < masks.size() && i < 3; i++) {
            // Only draw the first 3 masks
            cv::Mat overlay = image.clone();
            overlay.setTo(colors[i % colors.size()], masks[i].mask);
            cv::addWeighted(overlay, 0.5, image, 0.5, 0, image);

            if (!isBox) {
                sam2.drawPoint(image, pointVals[0], pointVals[1]);
            } else {
                sam2.drawBox(image, pointVals[0], pointVals[1],
                            pointVals[2], pointVals[3]);
            }

            // Write per-mask output
            std::string base = outputFile;
            size_t dot = base.find_last_of('.');
            std::string ext, stem;
            if (dot != std::string::npos) {
                stem = base.substr(0, dot);
                ext = base.substr(dot);
            } else {
                stem = base;
            }
            std::string outFile = stem + "_mask" + std::to_string(i) + ext;
            cv::imwrite(outFile, image);
            std::cout << "Saved: " << outFile << std::endl;

            // Reset image for next mask (reload original)
            image = cv::imread(inputFile);
        }
    }

    // Performance summary
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    sam2_ts.calbr_basetime(base_time);
    sam2_ts.build_timeline("SAM2");
    sam2_ts.show_summary("SAM2");
    sam2_ts.clear();

    return 0;
}
