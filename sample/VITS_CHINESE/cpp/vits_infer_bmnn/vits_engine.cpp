//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "vits_engine.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>

using namespace std;

VitsEngine::VitsEngine(string bmodel_file, int dev_id) {
    // get handle
    auto ret = bm_dev_request(&handle, dev_id);
    assert(BM_SUCCESS == ret);

    // judge pcie or soc
    ret = bm_get_misc_info(handle, &misc_info);
    assert(BM_SUCCESS == ret);

    // create bmrt
    bmrt = bmrt_create(handle);
    if (!bmrt_load_bmodel(bmrt, bmodel_file.c_str())) {
        cout << "load bmodel(" << bmodel_file << ") failed" << endl;
        exit(1);
    }

    // get network names
    const char **names;
    int num = bmrt_get_network_number(bmrt);
    if (num > 1) {
        cout << "This bmodel have " << num << " networks, and this program will only take network 0." << endl;
    }
    bmrt_get_network_names(bmrt, &names);
    for (int i = 0; i < num; ++i) {
        network_names.push_back(names[i]);
    }
    free(names);

    // get netinfo
    netinfo = bmrt_get_network_info(bmrt, network_names[0].c_str());
    if (netinfo->stage_num > 1) {
        cout << "This bmodel have " << netinfo->stage_num << " stages, and this program will only take stage 0." << endl;
    }

    // get input shapes
    m_max_length = netinfo->stages[0].input_shapes[0].dims[1];
    m_char_embed_seq_len = netinfo->stages[0].input_shapes[1].dims[1];
    m_char_embed_dim = netinfo->stages[0].input_shapes[1].dims[2];

    // get output shape (output[0] = y_Squeeze = audio [230400])
    m_audio_len = netinfo->stages[0].output_shapes[0].dims[0];

    cout << "Input 0: [1, " << m_max_length << "] dtype=" << netinfo->input_dtypes[0] << endl;
    cout << "Input 1: [1, " << m_char_embed_seq_len << ", " << m_char_embed_dim << "] dtype=" << netinfo->input_dtypes[1] << endl;
    cout << "Output 1 audio_len: " << m_audio_len << endl;

    m_ts = &tmp_ts;
}

VitsEngine::~VitsEngine() {
    if (bmrt != NULL) {
        bmrt_destroy(bmrt);
        bmrt = NULL;
    }
    bm_dev_free(handle);
}

float VitsEngine::estimate_silence_threshold(const float* audio, int len) {
    int num_samples = static_cast<int>(16000 * 0.1);
    if (num_samples > len) num_samples = len;
    int start = len - num_samples;
    float sum = 0.0f;
    for (int i = start; i < len; i++) {
        sum += fabsf(audio[i]);
    }
    return sum / num_samples;
}

int VitsEngine::remove_silence_from_end(float* audio, int len) {
    float threshold = estimate_silence_threshold(audio, len);
    int num_frames = len / 512;
    int last_non_silent_frame = -1;

    for (int f = num_frames - 1; f >= 0; f--) {
        int start = f * 512;
        int end = min(start + 512, len);
        float sum = 0.0f;
        for (int i = start; i < end; i++) {
            sum += fabsf(audio[i]);
        }
        float energy = sum / (end - start);
        if (energy > threshold) {
            last_non_silent_frame = f;
            break;
        }
    }

    if (last_non_silent_frame < 0) return 0;

    int end_index = (last_non_silent_frame + 1) * 512;
    return min(end_index, len);
}

float* VitsEngine::get_cpu_data(bm_tensor_t* tensor) {
    int count = bmrt_shape_count(&tensor->shape);
    float* pFP32 = new float[count];
    assert(pFP32 != nullptr);

    if (misc_info.pcie_soc_mode == 1) { // soc
        if (tensor->dtype == BM_FLOAT32) {
            unsigned long long addr;
            bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            memcpy(pFP32, (float*)addr, count * sizeof(float));
            bm_mem_unmap_device_mem(handle, (float*)addr, bm_mem_get_device_size(tensor->device_mem));
        } else {
            cerr << "unsupport dtype: " << tensor->dtype << endl;
        }
    } else { // pcie
        bm_memcpy_d2s_partial(handle, pFP32, tensor->device_mem, count * sizeof(float));
    }
    return pFP32;
}

int VitsEngine::RunInferenceRaw(const int32_t* x, int x_len,
                                  const float* char_embeds, int char_embeds_len,
                                  float*& y_segment, int& y_segment_len, float*& y_max) {
    int ret = 0;
    m_ts->save("vits inference");

    // create input tensors
    bm_tensor_t input_tensors[2];
    bm_tensor_t output_tensors[2];

    bool ok = bmrt_tensor(&input_tensors[0], bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]);
    assert(ok == true);
    ok = bmrt_tensor(&input_tensors[1], bmrt, netinfo->input_dtypes[1], netinfo->stages[0].input_shapes[1]);
    assert(ok == true);

    // copy input data to device
    ret = bm_memcpy_s2d(handle, input_tensors[0].device_mem, (void*)x);
    assert(BM_SUCCESS == ret);
    ret = bm_memcpy_s2d(handle, input_tensors[1].device_mem, (void*)char_embeds);
    assert(BM_SUCCESS == ret);

    // launch inference
    ok = bmrt_launch_tensor(bmrt, netinfo->name, input_tensors, netinfo->input_num,
                            output_tensors, netinfo->output_num);
    assert(ok == true);
    bm_thread_sync(handle);

    // free input device mem
    bm_free_device(handle, input_tensors[0].device_mem);
    bm_free_device(handle, input_tensors[1].device_mem);

    // get output data
    // output[0] = y_Squeeze = audio [230400]
    // output[1] = /Clip_output_0_Clip_f32 = y_max [1]
    y_segment = get_cpu_data(&output_tensors[0]);
    y_max = get_cpu_data(&output_tensors[1]);
    y_segment_len = output_tensors[0].shape.dims[0];

    // free output device mem
    bm_free_device(handle, output_tensors[0].device_mem);
    bm_free_device(handle, output_tensors[1].device_mem);

    m_ts->save("vits inference");

    return 0;
}

int VitsEngine::Postprocess(float* y_segment, int y_segment_len,
                              const float* y_max, std::vector<float>& output_audio) {
    // truncate audio based on y_max
    int truncated_len = static_cast<int>(ceil(y_max[0] / 900.0f * y_segment_len + 1));
    if (truncated_len > y_segment_len) truncated_len = y_segment_len;
    if (truncated_len < 1) truncated_len = y_segment_len;

    // remove silence from end
    int valid_len = remove_silence_from_end(y_segment, truncated_len);
    if (valid_len <= 0) valid_len = truncated_len;

    output_audio.assign(y_segment, y_segment + valid_len);

    return 0;
}

int VitsEngine::RunInference(const int32_t* x, int x_len,
                              const float* char_embeds, int char_embeds_len,
                              vector<float>& output_audio) {
    float* y_segment = nullptr;
    float* y_max = nullptr;
    int y_segment_len = 0;

    int ret = RunInferenceRaw(x, x_len, char_embeds, char_embeds_len,
                               y_segment, y_segment_len, y_max);
    if (ret != 0) return ret;

    ret = Postprocess(y_segment, y_segment_len, y_max, output_audio);

    delete[] y_max;
    delete[] y_segment;
    return ret;
}

void VitsEngine::WriteWav(const string& filename,
                           const vector<float>& audio,
                           int sample_rate) {
    vector<short> pcm(audio.size());
    for (size_t i = 0; i < audio.size(); i++) {
        float sample = audio[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        pcm[i] = static_cast<short>(sample * 32767.0f);
    }

    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Failed to open WAV file: " << filename << endl;
        return;
    }

    int data_size = pcm.size() * sizeof(short);
    int file_size = 36 + data_size;

    auto write32 = [&](int val) {
        file.put(val & 0xFF);
        file.put((val >> 8) & 0xFF);
        file.put((val >> 16) & 0xFF);
        file.put((val >> 24) & 0xFF);
    };
    auto write16 = [&](short val) {
        file.put(val & 0xFF);
        file.put((val >> 8) & 0xFF);
    };

    // RIFF header
    file.write("RIFF", 4);
    write32(file_size);
    file.write("WAVE", 4);

    // fmt chunk
    file.write("fmt ", 4);
    write32(16);           // chunk size
    write16(1);            // PCM format
    write16(1);            // mono
    write32(sample_rate);
    write32(sample_rate * 2); // byte rate
    write16(2);            // block align
    write16(16);           // bits per sample

    // data chunk
    file.write("data", 4);
    write32(data_size);
    file.write(reinterpret_cast<const char*>(pcm.data()), data_size);
    file.close();

    cout << "Saved WAV: " << filename << " ("
         << audio.size() << " samples, "
         << (float)audio.size() / sample_rate << "s)" << endl;
}