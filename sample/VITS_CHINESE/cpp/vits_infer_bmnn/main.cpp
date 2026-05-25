//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "vits_engine.hpp"
#include "vits_preprocess.hpp"

using namespace std;

static vector<string> split_text_near_punctuation(const string& text, int max_len) {
    vector<string> result;
    string remaining = text;
    const string punct = "\xe3\x80\x82\xef\xbc\x81\xef\xbc\x9f\xef\xbc\x8c\xe3\x80\x81\xef\xbc\x9b\xef\xbc\x9a,.";

    while ((int)remaining.size() > max_len) {
        int split_pos = max_len;
        while (split_pos > 0) {
            bool found = false;
            for (size_t i = 0; i < punct.size();) {
                int clen = 1;
                if ((unsigned char)punct[i] >= 0xE0 && (unsigned char)punct[i] <= 0xEF) clen = 3;
                if (split_pos + clen <= (int)remaining.size() &&
                    remaining.compare(split_pos, clen, punct, i, clen) == 0) {
                    found = true;
                    break;
                }
                i += clen;
            }
            if (found) break;
            split_pos--;
        }
        if (split_pos == 0) split_pos = max_len;
        int punct_end = split_pos;
        if (split_pos < (int)remaining.size()) {
            int clen = 1;
            if ((unsigned char)remaining[split_pos] >= 0xE0) clen = 3;
            punct_end = split_pos + clen;
        }
        result.push_back(remaining.substr(0, punct_end));
        remaining = remaining.substr(punct_end);
        remaining.erase(remaining.begin(), find_if(remaining.begin(), remaining.end(), [](unsigned char ch) {
            return !isspace(ch);
        }));
    }
    if (!remaining.empty()) {
        result.push_back(remaining);
    }
    return result;
}

int main(int argc, char* argv[]) {
    cout.setf(ios::fixed);

    const char* keys =
        "{bmodel | ../../models/BM1684X/vits_chinese_f16.bmodel | VITS bmodel file path}"
        "{bert_model | ../../models/BM1684X/bert_f16_1core.bmodel | BERT bmodel file path}"
        "{text_file | ../../datasets/vits_infer_item.txt | input text file}"
        "{pinyin_map | ./data/pinyin_map.txt | pinyin map file}"
        "{vocab_file | ../../python/bert/vocab.txt | BERT vocab file}"
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    string bmodel_file = parser.get<string>("bmodel");
    string bert_model = parser.get<string>("bert_model");
    string text_file = parser.get<string>("text_file");
    string pinyin_map = parser.get<string>("pinyin_map");
    string vocab_file = parser.get<string>("vocab_file");
    int dev_id = parser.get<int>("dev_id");

    // check params
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        cout << "Cannot find valid VITS model file: " << bmodel_file << endl;
        exit(1);
    }
    if (stat(bert_model.c_str(), &info) != 0) {
        cout << "Cannot find valid BERT model file: " << bert_model << endl;
        exit(1);
    }
    if (stat(text_file.c_str(), &info) != 0) {
        cout << "Cannot find text file: " << text_file << endl;
        exit(1);
    }
    if (stat(pinyin_map.c_str(), &info) != 0) {
        cout << "Cannot find pinyin map file: " << pinyin_map << endl;
        exit(1);
    }
    if (stat(vocab_file.c_str(), &info) != 0) {
        cout << "Cannot find vocab file: " << vocab_file << endl;
        exit(1);
    }

    // initialize preprocessor (BERT engine + text processing)
    VitsPreprocessor preprocessor(bert_model, pinyin_map, vocab_file, dev_id);

    // initialize VITS engine
    VitsEngine engine(bmodel_file, dev_id);

    // profiling
    TimeStamp ts;
    ts.save("vits overall");

    // read text file
    ifstream text_stream(text_file);
    if (!text_stream.is_open()) {
        cerr << "Cannot open " << text_file << endl;
        return -1;
    }

    // create results dir
    if (access("results", 0) != F_OK) {
        mkdir("results", S_IRWXU);
    }

    string line;
    int n = 0;
    while (getline(text_stream, line)) {
        // strip
        line.erase(line.begin(), find_if(line.begin(), line.end(), [](unsigned char ch) {
            return !isspace(ch);
        }));
        line.erase(find_if(line.rbegin(), line.rend(), [](unsigned char ch) {
            return !isspace(ch);
        }).base(), line.end());
        if (line.empty()) continue;

        cout << "Processing: " << line << endl;

        // split long text near punctuation
        int max_chars = engine.max_length() / 2 - 5;
        auto segments = split_text_near_punctuation(line, max_chars);

        int seg_idx = 0;
        for (const auto& seg : segments) {
            // run preprocessing (text -> phoneme IDs + char_embeds)
            vector<int32_t> input_ids;
            vector<float> char_embeds;
            ts.save("preprocess", 1);
            preprocessor.process(seg, input_ids, char_embeds);
            ts.save("preprocess", 1);

            // pad input_ids to VITS max_length
            int max_len = engine.max_length();
            int x_len = (int)input_ids.size();
            vector<int32_t> x_padded(max_len, 0);
            for (int i = 0; i < min(x_len, max_len); i++) {
                x_padded[i] = input_ids[i];
            }
            x_len = max_len;

            // char_embeds should already be [expand_len * embed_dim] = [128 * 256]
            int embeds_len = (int)char_embeds.size();

            // run VITS inference (raw)
            float* y_segment = nullptr;
            float* y_max = nullptr;
            int y_segment_len = 0;
            ts.save("vits inference", 1);
            if (0 != engine.RunInferenceRaw(x_padded.data(), x_len,
                                            char_embeds.data(), embeds_len,
                                            y_segment, y_segment_len, y_max)) {
                cerr << "Inference failed for segment " << seg_idx << endl;
                continue;
            }
            ts.save("vits inference", 1);

            // postprocess (truncation + silence removal)
            vector<float> output_audio;
            ts.save("postprocess", 1);
            engine.Postprocess(y_segment, y_segment_len, y_max, output_audio);
            ts.save("postprocess", 1);

            delete[] y_segment;
            delete[] y_max;

            // save wav
            string wav_file = "results/ts_" + to_string(n) + "_" + to_string(seg_idx) + ".wav";
            VitsEngine::WriteWav(wav_file, output_audio, 16000);
            seg_idx++;
        }
        n++;
    }

    text_stream.close();
    ts.save("vits overall");

    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    ts.calbr_basetime(base_time);
    ts.build_timeline("vits infer");
    ts.show_summary("vits infer");
    ts.clear();

    cout << endl << "Processed " << n << " text lines." << endl;
    return 0;
}