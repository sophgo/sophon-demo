//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <getopt.h>
#include "silero_vad.hpp"

using namespace std;

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --bmodel     <path>  bmodel file path (default: ../../models/BM1684X/silero_vad_bm1684x_f16.bmodel)\n");
    printf("  --input      <path>  input WAV file path (required, 16kHz mono)\n");
    printf("  --dev_id     <int>   TPU device id (default: 0)\n");
    printf("  --threshold  <float> speech probability threshold, 0.0~1.0 (default: 0.5)\n");
    printf("  --save_segments       save detected speech segments as separate WAV files\n");
    printf("  --help                print this help\n");
}

static const struct option long_options[] = {
    {"bmodel",        required_argument, 0, 0},
    {"input",         required_argument, 0, 1},
    {"dev_id",        required_argument, 0, 2},
    {"threshold",     required_argument, 0, 3},
    {"save_segments", no_argument,       0, 4},
    {"help",          no_argument,       0, 5},
    {0, 0, 0, 0}
};

int main(int argc, char* argv[]) {
    cout.setf(ios::fixed);

    string bmodel_file = "../../models/BM1684X/silero_vad_bm1684x_f16.bmodel";
    string input_file;
    int dev_id = 0;
    float threshold = 0.5f;
    bool save_segments = false;

    int opt;
    int opt_idx = 0;
    while ((opt = getopt_long(argc, argv, "", long_options, &opt_idx)) != -1) {
        switch (opt) {
            case 0: bmodel_file = optarg; break;
            case 1: input_file = optarg; break;
            case 2: dev_id = atoi(optarg); break;
            case 3: threshold = atof(optarg); break;
            case 4: save_segments = true; break;
            case 5: print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }

    if (input_file.empty()) {
        fprintf(stderr, "Error: --input is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Check files exist
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        fprintf(stderr, "Cannot find bmodel: %s\n", bmodel_file.c_str());
        return 1;
    }
    if (stat(input_file.c_str(), &info) != 0) {
        fprintf(stderr, "Cannot find input: %s\n", input_file.c_str());
        return 1;
    }

    // Create results dirs
    if (access("results", F_OK) != 0) {
        mkdir("results", S_IRWXU);
    }
    if (save_segments) {
        if (access("results/segments", F_OK) != 0) {
            mkdir("results/segments", S_IRWXU);
        }
    }

    // Load model
    SileroVAD vad(bmodel_file, dev_id);
    TimeStamp ts;
    vad.m_ts = &ts;

    // Read audio
    ts.save("read audio", 1);
    int num_samples = 0, sample_rate = 0;
    float* audio = read_wav(input_file, num_samples, sample_rate);
    if (!audio) {
        fprintf(stderr, "Failed to read audio file\n");
        return 1;
    }
    ts.save("read audio", 1);

    float duration = static_cast<float>(num_samples) / sample_rate;
    printf("Loaded %s :: %.2fs (%d samples @ %dHz)\n",
           input_file.c_str(), duration, num_samples, sample_rate);

    // Resample if not 16kHz
    if (sample_rate != SileroVAD::SAMPLE_RATE) {
        fprintf(stderr, "Warning: sample rate is %dHz, but model expects %dHz\n",
                sample_rate, SileroVAD::SAMPLE_RATE);
        fprintf(stderr, "Simple linear resampling is not implemented. Please resample with ffmpeg/sox.\n");
        fprintf(stderr, "  e.g.: ffmpeg -i %s -ar 16000 test_16k.wav\n", input_file.c_str());
        delete[] audio;
        return 1;
    }

    // Run VAD
    vector<SpeechSegment> speeches;
    vector<float> probs;
    vad.process_audio(audio, num_samples, speeches, probs, threshold);

    // Print results
    int num_frames = probs.size();
    printf("Frames: %d, speech segments: %zu\n", num_frames, speeches.size());
    for (size_t i = 0; i < speeches.size(); ++i) {
        float s = speeches[i].start / 16000.0f;
        float e = speeches[i].end / 16000.0f;
        printf("  seg %zu: %7.2fs -> %7.2fs (%.2fs)\n", i, s, e, e - s);
    }

    // Save segments
    if (save_segments) {
        string basename = input_file;
        size_t slash = input_file.rfind('/');
        if (slash != string::npos) basename = input_file.substr(slash + 1);
        size_t dot = basename.rfind('.');
        if (dot != string::npos) basename = basename.substr(0, dot);

        for (size_t i = 0; i < speeches.size(); ++i) {
            int seg_start = speeches[i].start;
            int seg_len = speeches[i].end - speeches[i].start;
            char outpath[512];
            snprintf(outpath, sizeof(outpath),
                     "results/segments/%s_seg%02zu_%.2fs_%.2fs.wav",
                     basename.c_str(), i,
                     seg_start / 16000.0f,
                     speeches[i].end / 16000.0f);
            save_wav(outpath, audio + seg_start, seg_len, 16000);
            printf("  saved: %s\n", outpath);
        }
    }

    // Save JSON result
    {
        string basename = input_file;
        size_t slash = input_file.rfind('/');
        if (slash != string::npos) basename = input_file.substr(slash + 1);
        size_t dot = basename.rfind('.');
        if (dot != string::npos) basename = basename.substr(0, dot);

        char json_path[512];
        snprintf(json_path, sizeof(json_path), "results/%s_vad_cpp.json", basename.c_str());
        FILE* fp = fopen(json_path, "w");
        if (fp) {
            fprintf(fp, "{\n");
            fprintf(fp, "  \"audio_file\": \"%s\",\n", input_file.c_str());
            fprintf(fp, "  \"duration_s\": %.2f,\n", duration);
            fprintf(fp, "  \"threshold\": %.2f,\n", threshold);
            fprintf(fp, "  \"num_frames\": %d,\n", num_frames);
            fprintf(fp, "  \"segments\": [\n");
            for (size_t i = 0; i < speeches.size(); ++i) {
                fprintf(fp, "    {\n");
                fprintf(fp, "      \"start_s\": %.4f,\n", speeches[i].start / 16000.0f);
                fprintf(fp, "      \"end_s\": %.4f,\n", speeches[i].end / 16000.0f);
                fprintf(fp, "      \"duration_s\": %.4f\n", (speeches[i].end - speeches[i].start) / 16000.0f);
                fprintf(fp, "    }%s\n", i < speeches.size() - 1 ? "," : "");
            }
            fprintf(fp, "  ]\n");
            fprintf(fp, "}\n");
            fclose(fp);
            printf("Result saved to %s\n", json_path);
        }
    }

    // Performance summary
    printf("\n------------------ Inference Time Info ----------------------\n");
    printf("frames: %d\n", num_frames);
    ts.show_summary("Silero VAD C++ test");

    delete[] audio;
    return 0;
}