//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "groundingdino.hpp"


int main(int argc, char* argv[]) {
	cout.setf(ios::fixed);
    // get params
    const char* keys =
        "{bmodel | ../../models/BM1684X/groundingdino_bm1684x_fp16.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{box_threshold | 0.3 | confidence threshold for filter boxes}"
        "{vocab_path | ../../models/bert-base-uncased/vocab.txt | vocab path}"
        "{help | 0 | print help information.}"
        "{text_threshold | 0.25 | confidence threshold for filter texts}"
        "{image_path | ../../datasets/test/zidane.jpg | test image path}"
        "{text_prompt | person | text prompt}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    std::string bmodel_file = parser.get<std::string>("bmodel");
    std::string vocab_path = parser.get<std::string>("vocab_path");
    std::string image_path = parser.get<std::string>("image_path");
    std::string text_prompt = parser.get<std::string>("text_prompt");
    int dev_id = parser.get<int>("dev_id");
    float box_threshold = parser.get<float>("box_threshold");
    float text_threshold = parser.get<float>("text_threshold");

    // check params
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        cout << "Cannot find valid model file." << endl;
        exit(1);
    }
    if (stat(image_path.c_str(), &info) != 0) {
        cout << "Cannot find input path." << endl;
        exit(1);
    }
    if (stat(vocab_path.c_str(), &info) != 0) {
        cout << "Cannot find vocab path." << endl;
        exit(1);
    }

    // create result dir
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    // profiling
    TimeStamp gd_ts;
    TimeStamp* ts = &gd_ts;

	GroundingDINO gd(bmodel_file, dev_id, box_threshold, vocab_path, text_threshold);
    gd.enableProfile(&gd_ts);
    // creat handle
    auto handle = sail::Handle(dev_id);
    sail::Bmcv bmcv(handle); // for imwrite

    ts->save("decode time");
    cv::Mat cvmat = cv::imread(image_path, cv::IMREAD_COLOR, dev_id);
    sail::BMImage bmimg;
    bmcv.mat_to_bm_image(cvmat, bmimg);
    ts->save("decode time");
	std::vector<Object> objects = gd.detect(bmimg, text_prompt);
    // std::vector<Object> objects = gd.detect(cvmat, text_prompt);

	for (size_t i = 0; i < objects.size(); i++)
	{
		cv::rectangle(cvmat, objects[i].box, cv::Scalar(0, 0, 255), 2);
		std::string label = cv::format("%.2f", objects[i].prob);
		label = objects[i].text + ":" + label;
		cv::putText(cvmat, label, cv::Point(objects[i].box.x, objects[i].box.y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0));
	}

	imwrite("results/result.jpg", cvmat);

    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    gd_ts.calbr_basetime(base_time);
    gd_ts.build_timeline("groundingdino test");
    gd_ts.show_summary("groundingdino test");
    gd_ts.clear();

	return 0;
}