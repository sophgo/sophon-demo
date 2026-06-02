//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "seaco_paraformer.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

// ===========================================================================
// CIF (Continuous Integrate-and-Fire) -- CPU
// ===========================================================================

void cif_cpu(const float* hidden, const float* alphas,
             int B, int T, int D, float threshold,
             std::vector<float>& acoustic_embeds, int& out_n) {
    // hidden: (B, T, D), alphas: (B, T)
    // Returns acoustic_embeds shaped (B, max_n, D) with max_n = out_n
    //
    // Matches FunASR CIF: save frame copy at EVERY timestep (before fire reset),
    // then collect frames at fire positions. This is critical — the old code
    // incorrectly reused the final frame for all fire positions.
    std::vector<float> integrate(B, 0.0f);
    std::vector<float> frame(B * D, 0.0f);

    // Per-batch: saved frame copies at each timestep
    std::vector<std::vector<std::vector<float>>> saved_frames(B);
    // Fire detection values (accumulated integrate at each step)
    std::vector<std::vector<float>> fires(B);

    for (int t = 0; t < T; t++) {
        for (int b = 0; b < B; b++) {
            float alpha = alphas[b * T + t];
            float dist_completion = 1.0f - integrate[b];
            integrate[b] += alpha;

            fires[b].push_back(integrate[b]);

            bool fire = (integrate[b] >= threshold);
            if (fire) integrate[b] -= threshold;

            float cur = fire ? dist_completion : alpha;
            float remains = alpha - cur;

            // frame += cur * hidden[t]
            for (int d = 0; d < D; d++)
                frame[b * D + d] += cur * hidden[b * T * D + t * D + d];

            // Save frame COPY at this timestep (before fire reset)
            saved_frames[b].push_back(
                std::vector<float>(frame.data() + b * D, frame.data() + (b + 1) * D));

            if (fire) {
                // frame = remains * hidden[t]  -- start new accumulation
                for (int d = 0; d < D; d++)
                    frame[b * D + d] = remains * hidden[b * T * D + t * D + d];
            }
        }
    }

    // Collect acoustic embeds: frames[fire_positions]
    out_n = 0;
    for (int b = 0; b < B; b++) {
        int count = 0;
        for (int t = 0; t < T; t++)
            if (fires[b][t] >= threshold) count++;
        if (count > out_n) out_n = count;
    }

    if (out_n == 0) {
        acoustic_embeds.clear();
        return;
    }

    acoustic_embeds.assign(B * out_n * D, 0.0f);
    for (int b = 0; b < B; b++) {
        int idx = 0;
        for (int t = 0; t < T; t++) {
            if (fires[b][t] >= threshold && idx < out_n) {
                auto& sf = saved_frames[b][t];
                for (int d = 0; d < D; d++)
                    acoustic_embeds[b * out_n * D + idx * D + d] = sf[d];
                idx++;
            }
        }
    }
}

void cif_wo_hidden_cpu(const float* alphas, int T, float threshold,
                       std::vector<float>& fires) {
    fires.resize(T, 0.0f);
    float integrate = 0.0f;
    for (int t = 0; t < T; t++) {
        integrate += alphas[t];
        fires[t] = integrate;
        if (integrate >= threshold) integrate -= threshold;
    }
}

// ===========================================================================
// Timestamp prediction
// ===========================================================================

void ts_prediction_lfr6(const float* us_alphas, const float* us_peaks,
                        int num_frames,
                        const std::vector<std::string>& char_list,
                        float force_time_shift, int upsample_rate,
                        std::vector<TimestampInfo>& timestamps,
                        std::vector<std::string>& new_char_list) {
    const int START_END_THRESHOLD = 5;
    const int MAX_TOKEN_DURATION = 12;
    const float TIME_RATE = 10.0f * 6 / upsample_rate; // ms per frame

    timestamps.clear();
    new_char_list.clear();

    // Find fire places (peaks >= 1.0 - 1e-4)
    std::vector<int> fire_place;
    for (int i = 0; i < num_frames; i++) {
        if (us_peaks[i] >= 1.0f - 1e-4f)
            fire_place.push_back((int)(i + force_time_shift));
    }

    if (fire_place.empty()) return;

    // Leading silence
    if (fire_place[0] > START_END_THRESHOLD) {
        timestamps.push_back({0, (int)std::round(fire_place[0] * TIME_RATE), "<sil>"});
    }

    // Token timestamps
    int num_tokens = (int)char_list.size();
    int num_fires = (int)fire_place.size();
    for (int i = 0; i < num_fires - 1 && i < num_tokens; i++) {
        int start_ms = (int)std::round(fire_place[i] * TIME_RATE);
        if (MAX_TOKEN_DURATION < 0 || fire_place[i + 1] - fire_place[i] <= MAX_TOKEN_DURATION) {
            int end_ms = (int)std::round(fire_place[i + 1] * TIME_RATE);
            timestamps.push_back({start_ms, end_ms, char_list[i]});
        } else {
            int split = fire_place[i] + MAX_TOKEN_DURATION;
            timestamps.push_back({start_ms, (int)std::round(split * TIME_RATE), char_list[i]});
            timestamps.push_back({(int)std::round(split * TIME_RATE),
                                  (int)std::round(fire_place[i + 1] * TIME_RATE), "<sil>"});
        }
    }

    // Trailing silence
    if (num_frames - fire_place.back() > START_END_THRESHOLD) {
        float end_frame = (num_frames + fire_place.back()) * 0.5f;
        if (!timestamps.empty())
            timestamps.back().end_ms = (int)(end_frame * TIME_RATE);
        timestamps.push_back({(int)(end_frame * TIME_RATE),
                              (int)(num_frames * TIME_RATE), "<sil>"});
    } else {
        if (!timestamps.empty())
            timestamps.back().end_ms = (int)(num_frames * TIME_RATE);
    }

    // Build new_char_list from timestamps
    for (auto& ts : timestamps)
        new_char_list.push_back(ts.text);
}

// ===========================================================================
// SeacoParaformer implementation
// ===========================================================================

SeacoParaformer::SeacoParaformer(const std::string& model_dir, int dev_id)
    : handle_(nullptr), bmrt_(nullptr), t_pre_(0), t_enc_(0), t_cif_(0),
      t_dec_(0), t_pred_(0), t_tok_(0) {
    // Request device
    auto ret = bm_dev_request(&handle_, dev_id);
    assert(BM_SUCCESS == ret);

    ret = bm_get_misc_info(handle_, &misc_info_);
    assert(BM_SUCCESS == ret);

    bmrt_ = bmrt_create(handle_);
    assert(bmrt_ != nullptr);

    // Load three bmodels
    load_network(model_dir + "/encoder_fp32_10b.bmodel", enc_net_);
    load_network(model_dir + "/decoder_fp32_10b.bmodel", dec_net_);
    load_network(model_dir + "/predictor_fp32_10b.bmodel", pred_net_);

    // Load tokens
    std::ifstream tok_file(model_dir + "/tokens.json");
    if (tok_file.is_open()) {
        json j;
        tok_file >> j;
        auto& tok_list = j.is_array() ? j : j["tokens"];
        for (auto& t : tok_list)
            tokens_.push_back(t.get<std::string>());
        std::cout << "Vocabulary size: " << tokens_.size() << std::endl;
    } else {
        std::cerr << "Warning: tokens.json not found" << std::endl;
    }

    // Load CMVN
    cmvn_ = load_cmvn(model_dir + "/am.mvn");
    if (cmvn_.means.n_elem == 0)
        std::cerr << "Warning: CMVN not found" << std::endl;
}

SeacoParaformer::~SeacoParaformer() {
    if (bmrt_) {
        bmrt_destroy(bmrt_);
        bmrt_ = nullptr;
    }
    if (handle_) {
        bm_dev_free(handle_);
        handle_ = nullptr;
    }
}

void SeacoParaformer::reset_timing() {
    t_pre_ = t_enc_ = t_cif_ = t_dec_ = t_pred_ = t_tok_ = 0.0;
}

// ---------------------------------------------------------------------------
// Load a single bmodel network
// ---------------------------------------------------------------------------

void SeacoParaformer::load_network(const std::string& bmodel_path, NetInfo& net) {
    bool ok = bmrt_load_bmodel(bmrt_, bmodel_path.c_str());
    if (!ok) {
        std::cerr << "Failed to load bmodel: " << bmodel_path << std::endl;
        throw std::runtime_error("bmrt_load_bmodel failed");
    }

    const char** names;
    int num_nets = bmrt_get_network_number(bmrt_);
    bmrt_get_network_names(bmrt_, &names);
    std::string net_name = names[num_nets - 1];
    free(names);

    net.info = bmrt_get_network_info(bmrt_, net_name.c_str());
    if (!net.info) {
        throw std::runtime_error("bmrt_get_network_info failed for " + net_name);
    }

    int stage = net.info->stage_num > 1 ? net.info->stage_num - 1 : 0;
    net.batch_size = net.info->stages[stage].input_shapes[0].dims[0];
    net.is_dynamic = net.info->is_dynamic;

    for (int i = 0; i < net.info->input_num; i++) {
        net.input_shapes.push_back(net.info->stages[stage].input_shapes[i]);
        net.input_dtypes.push_back(net.info->input_dtypes[i]);
        net.input_scales.push_back(net.info->input_scales[i]);
        net.input_names.push_back(net.info->input_names[i]);
    }
    for (int i = 0; i < net.info->output_num; i++) {
        net.output_shapes.push_back(net.info->stages[stage].output_shapes[i]);
        net.output_dtypes.push_back(net.info->output_dtypes[i]);
        net.output_scales.push_back(net.info->output_scales[i]);
        net.output_names.push_back(net.info->output_names[i]);
    }

    std::cout << "Loaded " << net_name << " from " << bmodel_path
              << " (dynamic=" << net.is_dynamic << ")" << std::endl;
    std::cout << "  inputs: " << net.info->input_num << "  outputs: " << net.info->output_num << std::endl;
    for (int i = 0; i < net.info->input_num; i++) {
        auto& s = net.info->stages[stage].input_shapes[i];
        std::cout << "    in[" << i << "] " << net.info->input_names[i]
                  << " dims=[";
        for (int d = 0; d < s.num_dims; d++) {
            if (d > 0) std::cout << ",";
            std::cout << s.dims[d];
        }
        std::cout << "] dtype=" << net.info->input_dtypes[i] << std::endl;
    }
    for (int i = 0; i < net.info->output_num; i++) {
        auto& s = net.info->stages[stage].output_shapes[i];
        std::cout << "    out[" << i << "] " << net.info->output_names[i]
                  << " dims=[";
        for (int d = 0; d < s.num_dims; d++) {
            if (d > 0) std::cout << ",";
            std::cout << s.dims[d];
        }
        std::cout << "]" << std::endl;
    }
}

// ---------------------------------------------------------------------------
// get_cpu_data / free_cpu_data (pattern from YOLOv8_cls)
// ---------------------------------------------------------------------------

float* SeacoParaformer::get_cpu_data(bm_tensor_t* tensor, float scale) {
    int count = bmrt_shape_count(&tensor->shape);
    float* pFP32 = nullptr;

    if (misc_info_.pcie_soc_mode == 1) { // SoC
        if (tensor->dtype == BM_FLOAT32) {
            unsigned long long addr;
            bm_mem_mmap_device_mem(handle_, &tensor->device_mem, &addr);
            bm_mem_invalidate_device_mem(handle_, &tensor->device_mem);
            pFP32 = (float*)addr;
        } else if (tensor->dtype == BM_INT32) {
            int32_t* pI32 = nullptr;
            unsigned long long addr;
            bm_mem_mmap_device_mem(handle_, &tensor->device_mem, &addr);
            bm_mem_invalidate_device_mem(handle_, &tensor->device_mem);
            pI32 = (int32_t*)addr;
            pFP32 = new float[count];
            for (int i = 0; i < count; i++) pFP32[i] = (float)pI32[i] * scale;
            bm_mem_unmap_device_mem(handle_, pI32, bm_mem_get_device_size(tensor->device_mem));
        } else {
            std::cerr << "unsupported dtype: " << tensor->dtype << std::endl;
        }
    } else { // PCIe
        if (tensor->dtype == BM_FLOAT32) {
            pFP32 = new float[count];
            bm_memcpy_d2s_partial(handle_, pFP32, tensor->device_mem, count * sizeof(float));
        } else if (tensor->dtype == BM_INT32) {
            int tensor_size = bmrt_tensor_bytesize(tensor);
            int32_t* pI32 = new int32_t[tensor_size / sizeof(int32_t)];
            bm_memcpy_d2s_partial(handle_, pI32, tensor->device_mem, tensor_size);
            pFP32 = new float[count];
            for (int i = 0; i < count; i++) pFP32[i] = (float)pI32[i] * scale;
            delete[] pI32;
        } else {
            std::cerr << "unsupported dtype: " << tensor->dtype << std::endl;
        }
    }
    return pFP32;
}

void SeacoParaformer::free_cpu_data(bm_tensor_t* tensor, float* data) {
    if (misc_info_.pcie_soc_mode == 1) { // SoC
        if (tensor->dtype == BM_FLOAT32) {
            bm_mem_unmap_device_mem(handle_, data, bm_mem_get_device_size(tensor->device_mem));
        } else {
            delete[] data;
        }
    } else {
        delete[] data;
    }
}

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

int SeacoParaformer::shape_count(const bm_shape_t& shape) {
    return (int)bmrt_shape_count(&shape);
}

int SeacoParaformer::shape_dim_at(const bm_shape_t& shape, int dim, int default_val) {
    if (dim < shape.num_dims) return shape.dims[dim];
    return default_val;
}

// ===========================================================================
// Dynamic network launch helper
// ===========================================================================
// Follows Qwen/cpp/qwen_bmlib net_launch_dyn pattern:
//   1. Allocate device memory for MAX compiled shape
//   2. Copy actual data
//   3. bm_set_device_mem to shrink to actual size
//   4. Set tensor.shape.dims to actual dimensions
//   5. Launch with bmrt_launch_tensor_ex(user_mem=true)
//
// The valid_tensors list specifies which input tensor indices need dynamic
// shape adjustment, along with the dim index and actual value to set.

struct DynamicDim {
    int tensor_idx;    // which input tensor
    int dim_idx;       // which dim (0=batch, 1=T, 2=D, etc.)
    int actual_val;    // actual value for this dim
};

static void launch_dynamic(SeacoParaformer* self, bm_handle_t handle, void* bmrt,
                           const bm_net_info_t* net_info, int stage,
                           std::vector<bm_tensor_t>& inputs,
                           std::vector<bm_tensor_t>& outputs,
                           const std::vector<DynamicDim>& dyn_dims) {
    // Get pre-allocated device memory from compiled stage
    auto& stg = net_info->stages[stage];

    // === Inputs: use pre-allocated device memory ===
    for (int i = 0; i < net_info->input_num; i++) {
        // Start from compiled max shape
        bm_shape_t max_shape = stg.input_shapes[i];

        if (net_info->is_dynamic) {
            // Compute actual shape by applying all DynamicDim overrides for this tensor
            bm_shape_t actual_shape = max_shape;
            for (auto& dd : dyn_dims) {
                if (dd.tensor_idx == i) {
                    actual_shape.dims[dd.dim_idx] = dd.actual_val;
                }
            }

            // Get pre-allocated device memory for max shape
            auto& dev_mem = stg.input_mems[i];
            // Compute actual bytes needed
            int actual_count = 1;
            for (int d = 0; d < actual_shape.num_dims; d++)
                actual_count *= actual_shape.dims[d];
            int dtype_size = (net_info->input_dtypes[i] == BM_INT32) ?
                             sizeof(int32_t) : sizeof(float);

            // Shrink device_mem to actual size
            bm_set_device_mem((bm_device_mem_t*)&dev_mem,
                              actual_count * dtype_size,
                              bm_mem_get_device_addr(dev_mem));

            // Create tensor with ACTUAL shape
            bmrt_tensor_with_device(&inputs[i], dev_mem,
                                    net_info->input_dtypes[i], actual_shape);
        } else {
            bmrt_tensor_with_device(&inputs[i], stg.input_mems[i],
                                    net_info->input_dtypes[i], max_shape);
        }
    }

    // === Outputs: use pre-allocated device memory (max shape) ===
    for (int i = 0; i < net_info->output_num; i++) {
        bmrt_tensor_with_device(&outputs[i], stg.output_mems[i],
                                net_info->output_dtypes[i],
                                stg.output_shapes[i]);
    }

    // === Launch ===
    bool ret = bmrt_launch_tensor_ex(bmrt, net_info->name,
                                     inputs.data(), net_info->input_num,
                                     outputs.data(), net_info->output_num,
                                     true, false);
    assert(ret);
    bm_thread_sync(handle);

    // Note: After launch, output tensor shapes SHOULD reflect actual dims.
    // If dynamic dim T was changed, outputs with T dim will have actual T.
    // We verify this by checking output shape after launch.
}

// ===========================================================================
// Encoder forward (TPU)
// ===========================================================================

void SeacoParaformer::encoder_forward(const float* speech, int speech_len,
                                       std::vector<float>& enc_out, int& enc_out_T, int& enc_out_D,
                                       std::vector<float>& hidden, int& hidden_T,
                                       std::vector<float>& alphas, int& alphas_T,
                                       float& token_num) {
    auto t0 = std::chrono::steady_clock::now();

    int batch = enc_net_.batch_size;           // compiled batch (10)
    int T_max = enc_net_.info->stages[0].input_shapes[0].dims[1];  // 1000
    int n_inputs = enc_net_.info->input_num;   // 2
    int n_outputs = enc_net_.info->output_num; // 4

    if (speech_len > T_max) {
        std::cerr << "Encoder: speech_len " << speech_len << " > T_max " << T_max << std::endl;
        speech_len = T_max;
    }

    std::vector<bm_tensor_t> input_tensors(n_inputs);
    std::vector<bm_tensor_t> output_tensors(n_outputs);

    // Copy data into pre-allocated device memory
    auto& stg = enc_net_.info->stages[0];

    // Input 0: speech (batch=10, T_max, FEAT_DIM) -> we write actual data into batch 0
    {
        int max_count = batch * T_max * FEAT_DIM;
        std::vector<float> padded(max_count, 0.0f);
        // Only write batch 0 with actual speech_len frames
        for (int t = 0; t < speech_len; t++)
            for (int d = 0; d < FEAT_DIM; d++)
                padded[t * FEAT_DIM + d] = speech[t * FEAT_DIM + d];
        bm_memcpy_s2d(handle_, stg.input_mems[0], padded.data());
    }

    // Input 1: speech_lengths (batch,) int32
    {
        std::vector<int32_t> slens(batch, 1);  // min length 1
        slens[0] = speech_len;
        bm_memcpy_s2d(handle_, stg.input_mems[1], slens.data());
    }

    // Dynamic dims: input[0].dim[0] -> 1 (batch), input[0].dim[1] -> speech_len (T)
    // input[1].dim[0] -> 1 (batch)
    std::vector<DynamicDim> dyn_dims;
    if (enc_net_.is_dynamic) {
        dyn_dims.push_back({0, 0, 1});           // actual batch = 1
        dyn_dims.push_back({0, 1, speech_len});  // actual T = speech_len
        dyn_dims.push_back({1, 0, 1});           // actual batch = 1 for speech_lengths
    }

    launch_dynamic(this, handle_, bmrt_, enc_net_.info, 0,
                   input_tensors, output_tensors, dyn_dims);

    auto t1 = std::chrono::steady_clock::now();
    t_enc_ += std::chrono::duration<double>(t1 - t0).count();

    // Read outputs — after dynamic launch, output shapes should reflect actual dims
    // Output order (from SAIL): enc_out(1,T,512), hidden(1,T+1,512), alphas(1,T+1), token_num(1)
    for (int i = 0; i < n_outputs; i++) {
        float scale = (i < (int)enc_net_.output_scales.size()) ? enc_net_.output_scales[i] : 1.0f;
        float* data = get_cpu_data(&output_tensors[i], scale);
        int ndim = output_tensors[i].shape.num_dims;
        int dim1 = (ndim > 1) ? output_tensors[i].shape.dims[1] : 0;
        int dim2 = (ndim > 2) ? output_tensors[i].shape.dims[2] : 0;

        // Match by output index (confirmed order from SAIL/Python):
        // out[0] = enc_LayerNormalization: (1, T, 512)
        // out[1] = hidden_Concat:           (1, T+1, 512)
        // out[2] = alphas_Add:              (1, T+1)
        // out[3] = token_num_Floor:         (1,)
        if (i == 0) {
            // enc_out (1, speech_len, 512)
            enc_out_T = dim1;
            enc_out_D = dim2;
            int n = enc_out_T * enc_out_D;
            enc_out.assign(data, data + n);
        } else if (i == 1) {
            // hidden (1, speech_len+1, 512)
            hidden_T = dim1;
            int n = hidden_T * dim2;
            hidden.assign(data, data + n);
        } else if (i == 2) {
            // alphas (1, speech_len+1)
            alphas_T = dim1;
            alphas.assign(data, data + dim1);
        } else if (i == 3) {
            // token_num (1,)
            token_num = data[0];
        }

        free_cpu_data(&output_tensors[i], data);
    }
}

// ===========================================================================
// Decoder forward (TPU)
// ===========================================================================

void SeacoParaformer::decoder_forward(const float* enc_out, int enc_T, int enc_D,
                                       const float* pre_embeds, int pre_N, int pre_D,
                                       int pre_token_len,
                                       std::vector<float>& logits, int& logits_N, int& vocab_size,
                                       std::vector<float>& dec_hidden) {
    auto t0 = std::chrono::steady_clock::now();

    int batch = dec_net_.batch_size;
    int dec_T_max = dec_net_.info->stages[0].input_shapes[0].dims[1];  // 1000
    int dec_N_max = dec_net_.info->stages[0].input_shapes[2].dims[1];  // 600
    int n_inputs = dec_net_.info->input_num;   // 4
    int n_outputs = dec_net_.info->output_num; // 2

    if (enc_T > dec_T_max) enc_T = dec_T_max;
    if (pre_N > dec_N_max) pre_N = dec_N_max;

    std::vector<bm_tensor_t> input_tensors(n_inputs);
    std::vector<bm_tensor_t> output_tensors(n_outputs);

    auto& stg = dec_net_.info->stages[0];

    // Input 0: enc (batch, T_max, enc_D) float32
    {
        int max_count = batch * dec_T_max * enc_D;
        std::vector<float> padded(max_count, 0.0f);
        for (int t = 0; t < enc_T; t++)
            for (int d = 0; d < enc_D; d++)
                padded[t * enc_D + d] = enc_out[t * enc_D + d];
        bm_memcpy_s2d(handle_, stg.input_mems[0], padded.data());
    }

    // Input 1: enc_len (batch,) int32
    {
        std::vector<int32_t> lens(batch, 1);
        lens[0] = enc_T;
        bm_memcpy_s2d(handle_, stg.input_mems[1], lens.data());
    }

    // Input 2: pre_acoustic_embeds (batch, N_max, pre_D) float32
    {
        int max_count = batch * dec_N_max * pre_D;
        std::vector<float> padded(max_count, 0.0f);
        for (int n = 0; n < pre_N; n++)
            for (int d = 0; d < pre_D; d++)
                padded[n * pre_D + d] = pre_embeds[n * pre_D + d];
        bm_memcpy_s2d(handle_, stg.input_mems[2], padded.data());
    }

    // Input 3: pre_token_length (batch,) int32
    {
        std::vector<int32_t> ptls(batch, 1);
        ptls[0] = pre_token_len;
        bm_memcpy_s2d(handle_, stg.input_mems[3], ptls.data());
    }

    // Dynamic dims
    std::vector<DynamicDim> dyn_dims;
    if (dec_net_.is_dynamic) {
        dyn_dims.push_back({0, 0, 1});         // enc batch = 1
        dyn_dims.push_back({0, 1, enc_T});     // enc T = actual
        dyn_dims.push_back({1, 0, 1});         // enc_len batch = 1
        dyn_dims.push_back({2, 0, 1});         // pre_embeds batch = 1
        dyn_dims.push_back({2, 1, pre_N});     // pre_embeds N = actual
        dyn_dims.push_back({3, 0, 1});         // pre_token_len batch = 1
    }

    launch_dynamic(this, handle_, bmrt_, dec_net_.info, 0,
                   input_tensors, output_tensors, dyn_dims);

    auto t1 = std::chrono::steady_clock::now();
    t_dec_ += std::chrono::duration<double>(t1 - t0).count();

    // Read outputs by index:
    // out[0] = decoder_out_LogSoftmax:          (1, N, vocab_size)
    // out[1] = decoder_hidden_LayerNormalization: (1, N, 512)
    for (int i = 0; i < n_outputs; i++) {
        float scale = (i < (int)dec_net_.output_scales.size()) ? dec_net_.output_scales[i] : 1.0f;
        float* data = get_cpu_data(&output_tensors[i], scale);
        int ndim = output_tensors[i].shape.num_dims;
        int dim2 = (ndim > 2) ? output_tensors[i].shape.dims[2] : 0;

        if (i == 0) {
            // logits (1, N, vocab_size)
            logits_N = output_tensors[i].shape.dims[1];
            vocab_size = dim2;
            int n = logits_N * vocab_size;
            logits.assign(data, data + n);
        } else if (i == 1) {
            // decoder hidden (1, N, 512)
            int N = output_tensors[i].shape.dims[1];
            int n = N * dim2;
            dec_hidden.assign(data, data + n);
        }

        free_cpu_data(&output_tensors[i], data);
    }
}

// ===========================================================================
// Predictor V3 forward (TPU)
// ===========================================================================

void SeacoParaformer::predictor_forward(const float* enc_out, int enc_T, int enc_D,
                                         std::vector<float>& us_alphas, int& us_T,
                                         float& pred_token_num) {
    auto t0 = std::chrono::steady_clock::now();

    int batch = pred_net_.batch_size;
    int pred_T_max = pred_net_.info->stages[0].input_shapes[0].dims[1];  // 1000
    int n_inputs = pred_net_.info->input_num;   // 2
    int n_outputs = pred_net_.info->output_num; // 2

    if (enc_T > pred_T_max) enc_T = pred_T_max;

    std::vector<bm_tensor_t> input_tensors(n_inputs);
    std::vector<bm_tensor_t> output_tensors(n_outputs);

    auto& stg = pred_net_.info->stages[0];

    // Input 0: enc (batch, T_max, enc_D) float32
    {
        int max_count = batch * pred_T_max * enc_D;
        std::vector<float> padded(max_count, 0.0f);
        for (int t = 0; t < enc_T; t++)
            for (int d = 0; d < enc_D; d++)
                padded[t * enc_D + d] = enc_out[t * enc_D + d];
        bm_memcpy_s2d(handle_, stg.input_mems[0], padded.data());
    }

    // Input 1: enc_len (batch,) int32
    {
        std::vector<int32_t> lens(batch, 1);
        lens[0] = enc_T;
        bm_memcpy_s2d(handle_, stg.input_mems[1], lens.data());
    }

    // Dynamic dims
    std::vector<DynamicDim> dyn_dims;
    if (pred_net_.is_dynamic) {
        dyn_dims.push_back({0, 0, 1});         // batch = 1
        dyn_dims.push_back({0, 1, enc_T});     // T = actual
        dyn_dims.push_back({1, 0, 1});         // batch = 1
    }

    launch_dynamic(this, handle_, bmrt_, pred_net_.info, 0,
                   input_tensors, output_tensors, dyn_dims);

    auto t1 = std::chrono::steady_clock::now();
    t_pred_ += std::chrono::duration<double>(t1 - t0).count();

    // Read outputs by index:
    // out[0] = alphas2_Squeeze:     (1, T_up) where T_up = enc_T * 3
    // out[1] = _token_num_ReduceSum: (1,)
    for (int i = 0; i < n_outputs; i++) {
        float scale = (i < (int)pred_net_.output_scales.size()) ? pred_net_.output_scales[i] : 1.0f;
        float* data = get_cpu_data(&output_tensors[i], scale);
        int ndim = output_tensors[i].shape.num_dims;
        int dim1 = (ndim > 1) ? output_tensors[i].shape.dims[1] : 0;

        if (i == 0) {
            // us_alphas (1, T_up)
            us_T = dim1;
            us_alphas.assign(data, data + dim1);
        } else if (i == 1) {
            // pred_token_num (1,)
            pred_token_num = data[0];
        }

        free_cpu_data(&output_tensors[i], data);
    }
}

// ---------------------------------------------------------------------------
// Full inference
// ---------------------------------------------------------------------------

SeacoParaformer::InferResult SeacoParaformer::infer(const std::vector<float>& audio,
                                                      int audio_sample_rate) {
    InferResult result;
    reset_timing();

    int num_samples = (int)audio.size();
    std::cout << "Audio: " << num_samples / (float)SAMPLE_RATE << " s  ("
              << num_samples << " samples)" << std::endl;

    // 1. Preprocess: FBANK + LFR + CMVN
    auto t0 = std::chrono::steady_clock::now();
    int frame_length_samples = FRAME_LENGTH_MS * SAMPLE_RATE / 1000;
    int frame_shift_samples = FRAME_SHIFT_MS * SAMPLE_RATE / 1000;

    // Scale float32 audio to int16 range (same as Python: audio * (1 << 15))
    // torchaudio Kaldi fbank expects integer-valued PCM samples
    arma::fmat waveform(1, num_samples);
    for (int i = 0; i < num_samples; i++) waveform(0, i) = audio[i] * (1 << 15);

    arma::fmat fbank_feats = fbank(waveform, N_MELS, frame_length_samples,
                                    frame_shift_samples, SAMPLE_RATE,
                                    0.0, 0.0, true, true, false);
    arma::fmat lfr_feats = apply_lfr(fbank_feats, LFR_M, LFR_N);
    apply_cmvn(lfr_feats, cmvn_);

    int speech_len = (int)lfr_feats.n_rows;
    std::vector<float> speech(1 * speech_len * FEAT_DIM, 0.0f);
    for (int i = 0; i < speech_len; i++)
        for (int d = 0; d < FEAT_DIM; d++)
            speech[i * FEAT_DIM + d] = lfr_feats(i, d);

    auto t1 = std::chrono::steady_clock::now();
    t_pre_ = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Features: " << speech_len << " frames" << std::endl;

    // 2. Encoder (TPU)
    std::vector<float> enc_out, hidden, alphas;
    int enc_out_T, enc_out_D, hidden_T, alphas_T;
    float token_num;
    encoder_forward(speech.data(), speech_len,
                    enc_out, enc_out_T, enc_out_D,
                    hidden, hidden_T, alphas, alphas_T, token_num);
    std::cout << "Encoder: enc_out=(" << enc_out_T << "," << enc_out_D
              << ") hidden=(" << hidden_T << ") alphas=(" << alphas_T
              << ") token_num=" << token_num << std::endl;

    // 3. CIF (CPU)
    t0 = std::chrono::steady_clock::now();
    std::vector<float> pre_embeds;
    int cif_n;
    int cif_T = std::min(hidden_T, speech_len + 1);
    cif_cpu(hidden.data(), alphas.data(), 1, cif_T, 512, 1.0f, pre_embeds, cif_n);
    int token_len_int = (int)std::ceil(token_num);
    if (cif_n > token_len_int) cif_n = token_len_int;
    int pre_token_len = (int)std::round(token_num);

    t1 = std::chrono::steady_clock::now();
    t_cif_ = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "CIF: embeds=(" << cif_n << ",512) token_len=" << pre_token_len << std::endl;

    if (cif_n == 0) {
        return result;
    }

    // 4. Decoder (TPU)
    std::vector<float> logits, dec_hidden;
    int logits_N, vocab_size;
    decoder_forward(enc_out.data(), enc_out_T, enc_out_D,
                    pre_embeds.data(), cif_n, 512, pre_token_len,
                    logits, logits_N, vocab_size, dec_hidden);
    std::cout << "Decoder: logits=(" << logits_N << "," << vocab_size << ")" << std::endl;

    // 5. Predictor V3 (TPU)
    std::vector<float> us_alphas;
    int us_T;
    float pred_token_num;
    predictor_forward(enc_out.data(), enc_out_T, enc_out_D,
                      us_alphas, us_T, pred_token_num);
    std::cout << "Predictor: us_alphas=" << us_T << " pred_token_num=" << pred_token_num << std::endl;

    // 6. Greedy decode (CPU)
    t0 = std::chrono::steady_clock::now();
    int N = pre_token_len;
    for (int i = 0; i < N && i < logits_N; i++) {
        int max_idx = 0;
        float max_val = logits[i * vocab_size];
        for (int v = 1; v < vocab_size; v++) {
            float val = logits[i * vocab_size + v];
            if (val > max_val) { max_val = val; max_idx = v; }
        }
        if (max_idx != SOS_ID && max_idx != EOS_ID && max_idx != BLANK_ID) {
            if (max_idx < (int)tokens_.size())
                result.tokens.push_back(tokens_[max_idx]);
            else
                result.tokens.push_back("<unk>");
            result.token_ids.push_back(max_idx);
        }
    }
    for (auto& tok : result.tokens)
        result.text += tok;
    // Clean up text
    size_t pos;
    while ((pos = result.text.find("@@")) != std::string::npos)
        result.text.erase(pos, 2);
    pos = 0;
    while ((pos = result.text.find(" ", pos)) != std::string::npos)
        result.text.erase(pos, 1);

    t1 = std::chrono::steady_clock::now();
    t_tok_ = std::chrono::duration<double>(t1 - t0).count();

    // 7. Timestamps
    if (!us_alphas.empty() && pred_token_num > 0) {
        // Normalize us_alphas sum to match pre_token_length
        float ratio = (float)pre_token_len / pred_token_num;
        std::vector<float> norm_alphas(us_T);
        for (int i = 0; i < us_T; i++) norm_alphas[i] = us_alphas[i] * ratio;

        std::vector<float> us_peaks;
        cif_wo_hidden_cpu(norm_alphas.data(), us_T, 1.0f - 1e-4f, us_peaks);

        int ts_frames = std::min(us_T, enc_out_T * 3);
        std::vector<std::string> new_char_list;
        ts_prediction_lfr6(norm_alphas.data(), us_peaks.data(), ts_frames,
                           result.tokens, -1.5f, 3,
                           result.sentence_info, new_char_list);
    }

    return result;
}
