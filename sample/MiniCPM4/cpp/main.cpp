//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "minicpm4.hpp"


// split the string by delimiter
void split(const std::string &s, const std::string &delim,
           std::vector<std::string> &ret)
{
    size_t last = 0;
    size_t index = s.find_first_of(delim, last);
    while (index != std::string::npos)
    {
        ret.push_back(s.substr(last, index - last));
        last = index + 1;
        index = s.find_first_of(delim, last);
    }
    if (last < s.length())
    {
        ret.push_back(s.substr(last));
    }
}

// parse the devices string
static std::vector<int> parseCascadeDevices(const std::string &str)
{
    std::vector<int> devices;
    std::vector<std::string> sub_str;
    split(str, ",", sub_str);
    for (auto &s : sub_str)
    {
        devices.push_back(std::atoi(s.c_str()));
    }
    return devices;
}

// show the help info
void Usage()
{
    printf("Usage:\n"
           "  --help         : Show help info.\n"
           "  --model        : Set model path \n"
           "  --tokenizer    : Set tokenizer path \n"
           "  --devid        : Set devices to run for model, e.g. 1,2, if not provided, use 0\n"
           "\n");
}

// process the arguments
void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &tokenizer_path, std::vector<int> &devices)
{
    struct option longOptions[] = {
        {"model", required_argument, nullptr, 'm'},
        {"tokenizer", required_argument, nullptr, 't'},
        {"devid", required_argument, nullptr, 'd'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    int optionIndex = 0;
    int option;

    while ((option = getopt_long(argc, argv, "m:t:d:h:", longOptions,
                                 &optionIndex)) != -1)
    {
        switch (option)
        {
        case 'm':
            model_path = optarg;
            break;
        case 't':
            tokenizer_path = optarg;
            break;
        case 'd':
            devices = parseCascadeDevices(optarg);
            break;
        case 'h':
            Usage();
            exit(EXIT_FAILURE);
        case '?':
            Usage();
            exit(EXIT_FAILURE);
        default:
            exit(EXIT_FAILURE);
        }
    }
}

//------------------ main function ------------------
int main(int argc, char **argv)
{
    std::cout.setf(std::ios::fixed);
    printf("Demo for MiniCPM4 in BM1684X, BM1688 and cv186ah.\n");

    // get params
    std::string model_path = "../../models/BM1684X/minicpm4-8b_w4bf16_seq512_bm1684x_1dev_20250613_175044.bmodel";
    // std::string model_path = "../models/BM1684X/minicpm4-8b_w4bf16_seq8192_bm1684x_1dev_20250613_182940.bmodel";
    std::string tokenizer_path = "../../python/token_config/tokenizer.json";
    std::vector<int> devids = {0};
    processArguments(argc, argv, model_path, tokenizer_path, devids);

    // check params
    struct stat info;
    if (stat(model_path.c_str(), &info) != 0)
    {
        std::cout << "Cannot find valid model file: " << model_path << std::endl;
        exit(1);
    }
    if (stat(tokenizer_path.c_str(), &info) != 0)
    {
        std::cout << "Cannot find tokenizer file: " << tokenizer_path << std::endl;
        exit(1);
    }

    auto minicpm4 = MiniCPM4();
    minicpm4.init(model_path, devids, tokenizer_path);

    std::vector<std::pair<std::string, std::string>> history_vector;

    while (true)
    {
        std::cout << "\nQuestion: ";

        std::string input_str;
        std::getline(std::cin, input_str);
        if (input_str == "exit")
            break;

        std::cout << "\nAnswer: " << std::flush;
        minicpm4.answer(input_str, history_vector);
        std::cout << std::endl;
    }

    minicpm4.deinit();
    return 0;
}
