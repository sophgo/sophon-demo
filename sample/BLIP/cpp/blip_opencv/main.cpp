//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "blip.hpp"
#include "tokenizer/bert_tokenizer.hpp"

#include <sys/stat.h>
#include <sstream>
#include <algorithm>
#include <vector>
#include <iostream>
#include <string>
#include <iomanip>

static bool fileExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0;
}

// Split a comma-separated string into trimmed tokens: "a,b,c" → ["a","b","c"]
static std::vector<std::string> splitComma(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // trim whitespace
        size_t a = item.find_first_not_of(" \t");
        size_t b = item.find_last_not_of(" \t");
        if (a == std::string::npos) continue;
        out.push_back(item.substr(a, b - a + 1));
    }
    return out;
}

static void usage() {
    std::cout
        << "Usage:\n"
        << "  blip_opencv --task itm  --input <image> --text \"text1,text2\" "
           "[--bmodel ...] [--vocab ...] [--dev_id 0]\n"
        << "  blip_opencv --task cap  --input <image> "
           "[--bmodel ...] [--vocab ...] [--dev_id 0]\n"
        << "  blip_opencv --task vqa  --input <image> "
           "[--venc_bmodel ...] [--tenc_bmodel ...] [--tdec_bmodel ...] "
           "[--vocab ...] [--dev_id 0]\n";
}

// Build an attention mask from input_ids: 1 where id != 0 (PAD), else 0.
// BLIP bmodels use INT32 for attention_mask.
static void buildAttentionMask(const std::vector<int>& ids,
                               std::vector<int>& mask) {
    mask.assign(ids.size(), 0);
    for (size_t i = 0; i < ids.size(); i++)
        mask[i] = (ids[i] != 0) ? 1 : 0;
}

// Remove the "a picture of " prompt prefix (13 chars), matching Python's
// caption[13:]. Falls back to a blind substr(13) if the exact prefix is not
// present (F16 platforms may tokenize the prefix slightly differently).
static std::string stripCaptionPrefix(const std::string& caption) {
    static const std::string prefix = "a picture of ";
    if (caption.size() >= prefix.size() &&
        caption.compare(0, prefix.size(), prefix) == 0) {
        return caption.substr(prefix.size());
    }
    if (caption.size() > 13) return caption.substr(13);
    return caption;
}

int main(int argc, char** argv) {
    const char* keys =
        "{task          | itm         | sub-task: itm | cap | vqa }"
        "{input         | <none>      | input image path }"
        "{text          | a woman sitting on the beach with a dog,a woman sitting on the beach with a cat | comma-separated texts (itm) }"
        "{bmodel        | ../../models/BM1684X/blip_itm_bm1684x_f32_1b.bmodel | bmodel path (itm/cap) }"
        "{venc_bmodel   | ../../models/BM1684X/blip_vqa_venc_bm1684x_f32_1b.bmodel | vqa image-encoder bmodel }"
        "{tenc_bmodel   | ../../models/BM1684X/blip_vqa_tenc_bm1684x_f32_1b.bmodel | vqa text-encoder bmodel }"
        "{tdec_bmodel   | ../../models/BM1684X/blip_vqa_tdec_bm1684x_f32_1b.bmodel | vqa text-decoder bmodel }"
        "{vocab         | ../../models/bert-base-uncased/vocab.txt | bert-base-uncased vocab.txt }"
        "{dev_id        | 0           | TPU device id }"
        "{help h        |             | print help information }";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        usage();
        return 0;
    }

    std::string task     = parser.get<std::string>("task");
    std::string inputImg = parser.get<std::string>("input");
    std::string textArg  = parser.get<std::string>("text");
    std::string bmodel   = parser.get<std::string>("bmodel");
    std::string vencBm   = parser.get<std::string>("venc_bmodel");
    std::string tencBm   = parser.get<std::string>("tenc_bmodel");
    std::string tdecBm   = parser.get<std::string>("tdec_bmodel");
    std::string vocab    = parser.get<std::string>("vocab");
    int dev_id           = parser.get<int>("dev_id");

    if (inputImg.empty() || inputImg == "<none>") {
        std::cerr << "Missing --input image path" << std::endl;
        usage();
        return 1;
    }
    if (!fileExists(inputImg)) {
        std::cerr << "Cannot find input image: " << inputImg << std::endl;
        return 1;
    }
    if (!fileExists(vocab)) {
        std::cerr << "Cannot find vocab.txt: " << vocab << std::endl;
        return 1;
    }

    // --- Load tokenizer ---
    BertTokenizer tokenizer(vocab);

    // --- Load image ---
    cv::Mat image = cv::imread(inputImg);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << inputImg << std::endl;
        return 1;
    }
    std::cout << "Input image: " << image.cols << "x" << image.rows << std::endl;

    // --- Device handle ---
    BMNNHandlePtr handle = std::make_shared<BMNNHandle>(dev_id);
    std::cout << "Set device id: " << dev_id << std::endl;

    TimeStamp ts;

    if (task == "itm") {
        if (!fileExists(bmodel)) {
            std::cerr << "Cannot find bmodel: " << bmodel << std::endl;
            return 1;
        }
        BlipItm model(handle, bmodel, dev_id);
        model.Init();
        model.enableProfile(&ts);

        std::vector<std::string> texts = splitComma(textArg);
        int maxLen = 35;
        // Tokenize all texts once
        std::vector<std::vector<int>> textIds(texts.size());
        std::vector<std::vector<int>> textMasks(texts.size());
        for (size_t i = 0; i < texts.size(); i++) {
            textIds[i] = tokenizer.encode(
                texts[i], maxLen, /*add_special_token=*/true, /*padding=*/true);
            buildAttentionMask(textIds[i], textMasks[i]);
        }

        std::vector<float> scores;
        int ret = model.predict(image, textIds, textMasks, scores);
        if (ret != 0) {
            std::cerr << "BLIP itm predict failed" << std::endl;
            return 1;
        }
        std::cout << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < texts.size(); i++) {
            std::cout << "Text: " << texts[i]
                      << ", Similarity: "
                      << (i < scores.size() ? scores[i] : 0.0f) << std::endl;
        }
        std::cout << std::defaultfloat;
    } else if (task == "cap") {
        if (!fileExists(bmodel)) {
            std::cerr << "Cannot find bmodel: " << bmodel << std::endl;
            return 1;
        }
        BlipCap model(handle, bmodel, dev_id);
        model.Init();
        model.enableProfile(&ts);

        std::vector<int> outIds;
        int ret = model.predict(image, outIds);
        if (ret != 0) {
            std::cerr << "BLIP cap predict failed" << std::endl;
            return 1;
        }
        std::string caption = tokenizer.decode(outIds);
        caption = stripCaptionPrefix(caption);
        std::cout << "Caption: " << caption << std::endl;

    } else if (task == "vqa") {
        if (!fileExists(vencBm) || !fileExists(tencBm) || !fileExists(tdecBm)) {
            std::cerr << "Cannot find vqa bmodels (need venc/tenc/tdec)"
                      << std::endl;
            return 1;
        }
        // [ENC] = vocab_size + 1 (init_tokenizer adds [DEC] at vocab_size, then
        // [ENC] as the next additional_special_token).
        int encTokenId = (int)tokenizer.vocabSize() + 1;
        BlipVqa model(handle, vencBm, tencBm, tdecBm, dev_id);
        model.Init(encTokenId);
        model.enableProfile(&ts);

        // Encode image once
        int ret = model.imageEncode(image);
        if (ret != 0) {
            std::cerr << "BLIP vqa imageEncode failed" << std::endl;
            return 1;
        }

        // Interactive Q&A loop (type "exit" to quit)
        std::cout << "Image encoded. Enter your question (or 'exit' to quit):"
                  << std::endl;
        std::string question;
        int qnum = 0;
        int maxLen = 35;
        while (std::getline(std::cin, question)) {
            // trim
            size_t a = question.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) continue;
            if (question.substr(a) == "exit") break;

            std::vector<int> ids = tokenizer.encode(
                question, maxLen, /*add_special_token=*/true, /*padding=*/true);
            // VQA text encoder expects [ENC] at input_ids[0]
            ids[0] = encTokenId;
            std::vector<int> mask;
            buildAttentionMask(ids, mask);

            std::vector<int> outIds;
            ret = model.predict(ids, mask, outIds);
            if (ret != 0) {
                std::cerr << "BLIP vqa predict failed" << std::endl;
                continue;
            }
            std::string answer = tokenizer.decode(outIds);
            std::cout << "Answer: " << answer << std::endl;
            std::cout << "Enter your question (or 'exit' to quit):" << std::endl;
            qnum++;
        }
        if (qnum == 0) qnum = 1;
    } else {
        std::cerr << "Unknown task: " << task << " (use itm | cap | vqa)"
                  << std::endl;
        usage();
        return 1;
    }

    // --- Performance summary ---
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    ts.calbr_basetime(base_time);
    ts.build_timeline("BLIP");
    ts.show_summary("BLIP");
    ts.clear();

    std::cout << "all done." << std::endl;
    return 0;
}
