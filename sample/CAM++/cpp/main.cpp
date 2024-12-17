//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "campplus.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include "cxxopts.hpp"
#include "npy.hpp"

int main(int argc, char *argv[]) {
    cxxopts::Options options("campplus", "a demo of CAM++ on SOPHON devices");
    options.add_options()
    ("bmodel", "bmodel file path", cxxopts::value<std::string>()->default_value("../models/BM1684X/campplus_bm1684x_fp32_1b.bmodel"))
    ("dev_id", "TPU device id", cxxopts::value<int>()->default_value("0"))
    ("input", "input path, wav file dir path", cxxopts::value<std::string>()->default_value("../datasets/test"));
    auto parser = options.parse(argc, argv);

    std::string bmodel_file  = parser["bmodel"].as<std::string>();
    std::string input        = parser["input"].as<std::string>();
    int dev_id               = parser["dev_id"].as<int>();
    bool ret;

    // check params
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        std::cout << "Cannot find valid model file." << std::endl;
        exit(1);
    }
    if (stat(input.c_str(), &info) != 0){
        std::cout << "Cannot find input path." << std::endl;
        exit(1);
    }

    // BMRuntime
    bm_handle_t bm_handle;
    bm_status_t status = bm_dev_request(&bm_handle, dev_id);
    assert(BM_SUCCESS == status);

    // create bmruntime
    void *p_bmrt = bmrt_create(bm_handle);
    assert(NULL != p_bmrt);

    // load bmodel by file
    ret = bmrt_load_bmodel(p_bmrt, bmodel_file.c_str());
    assert(true == ret);

    // print model infomation
    const char **net_names = NULL;
    bmrt_get_network_names(p_bmrt, &net_names);

    const bm_net_info_t *net_info = bmrt_get_network_info(p_bmrt, net_names[0]);
    assert(NULL != net_info);

    bmrt_print_network_info(net_info);
    std::cout << std::endl;

    // input
    int cn = 0;
    std::vector<std::string> files_vector;
    if (stat(input.c_str(), &info) != 0) {
        std::cout << "[INFO]: Input directory or file does not exist: " << input << std::endl;
    } else if (info.st_mode & S_IFDIR) {
        DIR *pDir;
        struct dirent* ptr;
        pDir = opendir(input.c_str());
        while((ptr = readdir(pDir))!=0) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
                files_vector.push_back(input + "/" + ptr->d_name);
                ++cn;
            }
        }
        closedir(pDir);
    } else {
        files_vector.push_back(input);
        ++cn;
    }
    std::sort(files_vector.begin(), files_vector.end());

    // output directory
    std::string output_dir = "./results";
    if (stat(output_dir.c_str(), &info) != 0) {
        // directory does not exist, create
        if (mkdir(output_dir.c_str(), 0755) == 0) {
            std::cout << "[INFO]: Directory created: " << output_dir << std::endl;
        } else {
            std::cerr << "[Error]: Failed to create directory: " << output_dir << std::endl;
        }
    } else if (info.st_mode & S_IFDIR) {
        std::cout << "[INFO]: Results directory already exists: " << output_dir << std::endl;
    } else {
        std::cerr << "[Error]: Path exists but is not a directory: " << output_dir << std::endl;
    }

    TimeStamp campplus_ts;
    TimeStamp* ts = &campplus_ts;

    // initialize feature_extractor with n_mels = 80 and sampling rate=16000 Hz
    speakerlab::FbankOptions fbank_opts;
    fbank_opts.mel_opts.num_bins = 80;
    fbank_opts.frame_opts.sample_freq = 16000.0;
    fbank_opts.frame_opts.dither = 0.0f;

    auto fbank_computer = std::make_shared<speakerlab::FbankComputer>(fbank_opts);

    // calculate embedding for each file and stored data in emb variable and npy file
    std::cout << "[INFO]: Extracting embeddings..." << std::endl;
    unsigned long output_shape1 = static_cast<unsigned long>(net_info->stages[0].output_shapes->dims[0]);
    unsigned long output_shape2 = static_cast<unsigned long>(net_info->stages[0].output_shapes->dims[1]);
    std::vector<std::vector<float>> emb(cn, std::vector<float>(output_shape1 * output_shape2));
    for (int i=0; i < cn; ++i) {
        // calculating embedding for wav_file
        compute_embedding(files_vector[i], fbank_computer, p_bmrt, emb[i], ts);

        // save embedding to npy
        npy::npy_data_ptr<float> d;
        d.data_ptr = emb[i].data();
        d.shape = {output_shape1, output_shape2};
        d.fortran_order = false;
        size_t slash_position = files_vector[i].find_last_of('/');
        size_t dot_position = files_vector[i].find_last_of('.');
        std::string npypath = output_dir + "/" + files_vector[i].substr(slash_position+1, dot_position) + ".npy";
        npy::write_npy(npypath, d);
        std::cout << "[INFO]: The extracted embedding from " << files_vector[i] << " is saved to " << npypath << std::endl;
    }

    if (cn>1)
        std::cout << "[INFO]: Computing the similarity score..." << std::endl;

    for (int i=0;i < cn;++i)
        for (int j=i+1;j < cn; ++j)
            std::cout << "[INFO]: The similarity score between "
                << files_vector[i] << " and " << files_vector[j] << " is "
                << cosine_similarity(emb[i], emb[j]) << std::endl;

    bmrt_destroy(p_bmrt);
    bm_dev_free(bm_handle);

    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    ts->calbr_basetime(base_time);
    ts->build_timeline("Campplus detect");
    ts->show_summary("Campplus detect");
    ts->clear();

    return 0;
}
