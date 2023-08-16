//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#define USE_FFMPEG 1
#define USE_OPENCV 1
#define USE_BMCV 1
#include "bert_sail.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <time.h>
#include <utils.hpp>
#include <memory>
#include "opencv2/opencv.hpp"
int main(int argc, char *argv[]) {
    const char *keys =
        "{bmodel | ../../models/BM1684/bert4torch_output_fp32_1b.bmodel | "
        "bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{dict_path | ../../models/pre_train/chinese-bert-wwm/vocab.txt | pre "
        "train vab path}"
        "{input | ../../datasets/china-people-daily-ner-corpus/example.test | "
        "input}"
        "{help | 0 | Print help information.}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    std::string bmodel_file = parser.get<std::string>("bmodel");
    int dev_id = parser.get<int>("dev_id");
    std::string dict_path = parser.get<std::string>("dict_path");
    std::string input = parser.get<std::string>("input");
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        cout << "Cannot find valid model file." << endl;
        exit(1);
    }
    if (stat(dict_path.c_str(), &info) != 0) {
        cout << "Cannot find dict_path file." << endl;
        exit(1);
    }
    if (input != "dev" && stat(input.c_str(), &info) != 0) {
        cout << "Cannot find input path." << endl;
        exit(1);
    }

    BERT bert(bmodel_file, dict_path, dev_id);
    TimeStamp Bert_ts;
    TimeStamp *bert_ts = &Bert_ts;
    // bert.enableProfile(&bert_ts);
    int b = bert.get_batch_size();
    if (access("results", 0) != F_OK) mkdir("results", S_IRWXU);
    std::string output_path = "results/" +
                              bmodel_file.substr(bmodel_file.rfind("/") + 1) +
                              "_test_sail_cpp_result.txt";

    if (input == "dev") { // dev test
        cout << "请输入中文文本，以空格/回车分隔：" << endl;
        while (1) {
            string input;
            cin >> input;
            bert.pre_process(input);
            bert.Detect();
            std::vector<
                std::pair<std::vector<std::string>, std::vector<std::string>>>
                ans = bert.post_process();
            cout << "实体：";
            if (ans[0].first.empty()) cout << "无" << endl;
            for (auto i : ans[0].first) {
                std::cout << i << ',';
            }
        }
    } else { // text test
        fstream file;

        file.open(input.c_str(), ios::in);
        if (!file.is_open()) {
            std::cout << "NO test" << std::endl;
            exit(1);
        }
        if (input.substr(input.size() - 4, 4) != "test") { // one test
            string text;
            file >> text;
            bert.pre_process(text);
            bert.Detect();
            std::vector<
                std::pair<std::vector<std::string>, std::vector<std::string>>>
                ans = bert.post_process();

            for (auto i : ans[0].first) {
                std::cout << i << endl;
            }
            return 0;
        }
        ofstream out(output_path.c_str()); // dataset test

        if (out) {
            cout << "out create" << std::endl;
        }
        string input, label, text, line;
        text = "";
        int tot = 0;
        vector<string> texts;
        while (getline(file, line)) {
            stringstream ss;
            ss << line;
            ss >> input;
            if (!line.size()) {
                texts.push_back(text);
                text = "";
            } else {
                text += input;
                ss >> label;
            }
        }
        if (b == 8) { // 8 batch

            for (int i = 0; i < texts.size(); i += b) {
                tot++;
                if (tot % 10 == 0) cout << tot * 8 << endl;
                vector<string> ts;
                for (int j = 0; j < min(b, (int)texts.size() - i); j++) {
                    ts.push_back(texts[i + j]);
                }
                for (int j = (int)texts.size() - i; j < b; j++) {
                    ts.push_back("");
                }
                LOG_TS(bert_ts, "bert tots");
                LOG_TS(bert_ts, "bert preprocess");
                bert.pre_process(ts);
                LOG_TS(bert_ts, "bert preprocess");

                LOG_TS(bert_ts, "bert inference");
                bert.Detect();
                LOG_TS(bert_ts, "bert inference");
                LOG_TS(bert_ts, "bert postprocess");
                std::vector<std::pair<std::vector<std::string>,
                                      std::vector<std::string>>>
                    ans = bert.post_process();
                LOG_TS(bert_ts, "bert postprocess");
                LOG_TS(bert_ts, "bert tots");
                for (int j = 0; j < min(b, (int)texts.size() - i); j++) {
                    string t = "[";
                    for (auto k : ans[j].second) {
                        t += '\'';
                        t += k;
                        t += '\'';
                        t += ',';
                    }
                    t[t.length() - 1] = ']';
                    out << t << endl;
                }
            }

        } else { // 1 batch

            for (auto text : texts) {
                tot++;
                if (tot % 100 == 0) cout << tot << endl;

                LOG_TS(bert_ts, "bert tots");
                LOG_TS(bert_ts, "bert preprocess");
                bert.pre_process(text);
                LOG_TS(bert_ts, "bert preprocess");

                LOG_TS(bert_ts, "bert inference");
                bert.Detect();
                LOG_TS(bert_ts, "bert inference");
                LOG_TS(bert_ts, "bert postprocess");
                std::vector<std::pair<std::vector<std::string>,
                                      std::vector<std::string>>>
                    ans = bert.post_process();
                LOG_TS(bert_ts, "bert postprocess");
                LOG_TS(bert_ts, "bert tots");
                string t = "[";

                for (auto i : ans[0].second) {
                    t += '\'';
                    t += i;
                    t += '\'';
                    t += ',';
                }
                t[t.length() - 1] = ']';
                out << t << endl;
            }
        }

        time_stamp_t base_time =
            time_point_cast<microseconds>(steady_clock::now());
        Bert_ts.calbr_basetime(base_time);
        Bert_ts.build_timeline("bert test");
        Bert_ts.show_summary("bert test");
        Bert_ts.clear();
        out.close();
    }

    return 0;
}