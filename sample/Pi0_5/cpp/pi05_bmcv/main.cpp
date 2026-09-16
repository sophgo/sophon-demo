//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <sys/stat.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "npy_io.h"
#include "pi05.h"

namespace {

// Prints the command line reference.
//
// in:  prog  argv[0], used as the program name in the usage text
void PrintUsage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " --input DIR [options]\n"
        << "\n"
        << "  --bmodel_dir DIR   directory holding the six bmodels (default ../models/BM1684X)\n"
        << "  --input DIR        observation directory, must contain agentview.npy and\n"
        << "                     wrist.npy: uint8, shape [224,224,3], RGB\n"
        << "  --cam3 FILE        optional third camera view (official slot right_wrist_0_rgb),\n"
        << "                     same .npy layout. The shipped bmodels have only two image\n"
        << "                     slots, so passing this is an error until siglip/dkv are\n"
        << "                     rebuilt for three views -- see the camera-slot comment in pi05.h\n"
        << "  --data_dir DIR     dataset root holding prefix_assets/ and action_unnorm.npz\n"
        << "                     (default: parent directory of --input)\n"
        << "  --task_id N        task index 0-9, selects the prefix asset set (default 0)\n"
        << "  --num_steps N      denoise steps (default " << Pi05::kNumSteps << ")\n"
        << "  --loops N          repeat the inference N times for timing (default 1)\n"
        << "  --seed N           noise seed; a fixed value makes runs reproducible (default 0)\n"
        << "  --noise FILE       read the initial noise from a float32 [10,32] .npy instead of\n"
        << "                     generating it from --seed. Feed the same file to the reference\n"
        << "                     implementation to compare action chunks exactly\n"
        << "  --dev_id N         TPU device id (default 0)\n"
        << "  --output FILE      output action path (default results/action.npy)\n"
        << "  -h, --help         show this message\n";
}

// Returns the directory part of a path, or "." when there is none.
//
// in:  path  file or directory path
// out: parent directory
std::string DirName(const std::string& path) {
    const size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

// Creates every directory in a path, like "mkdir -p".
//
// in:  path  directory path to create
// out: true when the directory exists afterwards
bool MakeDirs(const std::string& path) {
    std::string acc;
    size_t start = 0;
    if (!path.empty() && path[0] == '/') {
        acc = "/";
        start = 1;
    }
    while (start <= path.size()) {
        const size_t slash = path.find('/', start);
        const std::string part = path.substr(start, slash == std::string::npos
                                                       ? std::string::npos
                                                       : slash - start);
        if (!part.empty()) {
            acc += part;
            if (mkdir(acc.c_str(), 0755) != 0 && errno != EEXIST) return false;
            acc += "/";
        }
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string bmodel_dir = "../models/BM1684X";
    std::string input_dir;
    std::string data_dir;
    std::string noise_path;
    std::string cam3_path;
    std::string output = "results/action.npy";
    int task_id = 0, num_steps = Pi05::kNumSteps, loops = 1, dev_id = 0;
    uint64_t seed = 0;

    for (int i = 1; i < argc; i++) {
        // Returns the value that follows the current option, or exits when it is missing.
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << std::endl;
                exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--bmodel_dir")) {
            bmodel_dir = next("--bmodel_dir");
        } else if (!strcmp(argv[i], "--input")) {
            input_dir = next("--input");
        } else if (!strcmp(argv[i], "--data_dir")) {
            data_dir = next("--data_dir");
        } else if (!strcmp(argv[i], "--task_id")) {
            task_id = atoi(next("--task_id"));
        } else if (!strcmp(argv[i], "--num_steps")) {
            num_steps = atoi(next("--num_steps"));
        } else if (!strcmp(argv[i], "--loops")) {
            loops = atoi(next("--loops"));
        } else if (!strcmp(argv[i], "--seed")) {
            seed = strtoull(next("--seed"), nullptr, 10);
        } else if (!strcmp(argv[i], "--noise")) {
            noise_path = next("--noise");
        } else if (!strcmp(argv[i], "--cam3")) {
            cam3_path = next("--cam3");
        } else if (!strcmp(argv[i], "--dev_id")) {
            dev_id = atoi(next("--dev_id"));
        } else if (!strcmp(argv[i], "--output")) {
            output = next("--output");
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            PrintUsage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::cerr << "unknown option: " << argv[i] << std::endl;
            PrintUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (input_dir.empty()) {
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (data_dir.empty()) data_dir = DirName(input_dir);

    // Observations are plain .npy arrays: uint8, shape [224, 224, 3], RGB, row-major.
    // Keeping them in the sample's own array format means the example needs no image
    // decoding library at all, so it builds with nothing but libsophon and a C++ compiler.
    std::vector<uint8_t> agentview, wrist, right_wrist;
    std::vector<size_t> shape;
    const std::vector<size_t> want = {224, 224, 3};
    for (const auto& view : {std::make_pair("agentview", &agentview),
                             std::make_pair("wrist", &wrist)}) {
        const std::string path = input_dir + "/" + view.first + ".npy";
        if (!pi05::npy_load_u8(path, view.second, &shape)) {
            std::cerr << "failed to read " << path << " (expected uint8 .npy)" << std::endl;
            return EXIT_FAILURE;
        }
        if (shape != want) {
            std::cerr << "  " << path << " has shape [";
            for (size_t i = 0; i < shape.size(); i++) std::cerr << (i ? "," : "") << shape[i];
            std::cerr << "], expected [224,224,3]" << std::endl;
            return EXIT_FAILURE;
        }
    }
    if (!cam3_path.empty()) {
        if (!pi05::npy_load_u8(cam3_path, &right_wrist, &shape) || shape != want) {
            std::cerr << "failed to read " << cam3_path
                      << " (expected uint8 .npy of shape [224,224,3])" << std::endl;
            return EXIT_FAILURE;
        }
    }

    Pi05 model(dev_id, bmodel_dir);
    if (!model.ok()) {
        std::cerr << "init failed: " << model.last_error() << std::endl;
        return EXIT_FAILURE;
    }
    if (!model.load_task(task_id, data_dir)) {
        std::cerr << "load task " << task_id << " failed: " << model.last_error() << std::endl;
        return EXIT_FAILURE;
    }

    // The initial noise is an input like any other: pinning it lets the same draw be fed to
    // the reference implementation, so a comparison measures the port and not the sampler.
    std::vector<float> noise;
    std::vector<size_t> noise_shape;
    if (!noise_path.empty()) {
        if (!pi05::npy_load_f32(noise_path, &noise, &noise_shape)) {
            std::cerr << "failed to read " << noise_path << " (expected float32 .npy)" << std::endl;
            return EXIT_FAILURE;
        }
        if (noise.size() != static_cast<size_t>(Pi05::kActionHorizon) * Pi05::kActionDim) {
            std::cerr << "  " << noise_path << " has " << noise.size() << " elements, expected "
                      << Pi05::kActionHorizon * Pi05::kActionDim << std::endl;
            return EXIT_FAILURE;
        }
    } else {
        noise = Pi05::make_noise(seed);
    }

    std::vector<float> action;
    Pi05::Timing timing;
    for (int i = 0; i < loops; i++) {
        if (!model.infer(agentview, wrist, right_wrist, noise, num_steps, action, &timing)) {
            std::cerr << "infer failed: " << model.last_error() << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << std::fixed << std::setprecision(1)
              << "preprocess " << timing.preprocess_ms << " ms\n"
              << "siglip     " << timing.siglip_ms << " ms\n"
              << "dkv        " << timing.dkv_ms << " ms\n"
              << "ddn x" << num_steps << "    " << timing.ddn_ms << " ms\n"
              << "postproc   " << timing.postprocess_ms << " ms\n"
              << "total      " << timing.total_ms << " ms  (loops=" << loops
              << ", dev=" << dev_id << ")" << std::endl;

    std::cout << std::setprecision(4) << "action[0][:7] =";
    for (int d = 0; d < 7; d++) std::cout << " " << action[d];
    std::cout << std::endl;

    if (!MakeDirs(DirName(output))) {
        std::cerr << "failed to create output directory for " << output << std::endl;
        return EXIT_FAILURE;
    }
    if (!pi05::write_npy_f32(output, action,
                             {static_cast<size_t>(Pi05::kActionHorizon), 7})) {
        std::cerr << "failed to write " << output << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "wrote " << output << std::endl;
    return EXIT_SUCCESS;
}
