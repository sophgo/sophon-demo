//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "pi05.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

#include "npy_io.h"

using pi05::npy_load_f32;
using pi05::npz_load_f32;

namespace {

constexpr int kImgSize = 224;
constexpr int kImgPixels = kImgSize * kImgSize;
constexpr int kMaxIO = 64;

// Converts an interleaved RGB byte image to normalised CHW float.
//
// in:  rgb  kImgPixels * 3 bytes, interleaved RGB
// out: out  3 * kImgPixels floats, CHW, mapped to [-1, 1] as rgb/127.5 - 1
void RgbToChw(const uint8_t* rgb, float* out) {
    for (int i = 0; i < kImgPixels; i++) {
        out[i] = static_cast<float>(rgb[i * 3 + 0]) * (1.0f / 127.5f) - 1.0f;
        out[kImgPixels + i] = static_cast<float>(rgb[i * 3 + 1]) * (1.0f / 127.5f) - 1.0f;
        out[2 * kImgPixels + i] = static_cast<float>(rgb[i * 3 + 2]) * (1.0f / 127.5f) - 1.0f;
    }
}

}  // namespace

Pi05::Pi05(int dev_id, const std::string& model_dir) : model_dir_(model_dir), dev_id_(dev_id) {
    // All three heaps are required: the visual encoder uses the VPU decode path, the backbone
    // and the action expert use the NPU.
    setenv("BMRUNTIME_NEURON_HEAP_MASK", "7", 1);

    if (bm_dev_request(&handle_, dev_id_)) {
        fail(Status::kDeviceOpenFail, "bm_dev_request failed");
        return;
    }
    bmrt_ = bmrt_create(handle_);
    if (!bmrt_) {
        fail(Status::kRuntimeCreateFail, "bmrt_create failed");
        return;
    }

    const char* kModelFiles[6] = {
        "/pi05_siglip_w8bf16_2b.bmodel",  "/pi05_dkv0_9_w8bf16_1b.bmodel",
        "/pi05_dkv9_18_w8bf16_1b.bmodel", "/pi05_ddn0_6_bf16_1b.bmodel",
        "/pi05_ddn6_12_bf16_1b.bmodel",   "/pi05_ddn12_18_bf16_1b.bmodel",
    };
    Net* nets[6] = {&siglip_, &dkv_[0], &dkv_[1], &ddn_[0], &ddn_[1], &ddn_[2]};
    for (int i = 0; i < 6; i++) {
        if (!load_net(model_dir_ + kModelFiles[i], nets[i])) return;
    }

    prefix_embs_.resize(static_cast<size_t>(kPrefixLen) * kHidden);
    p_amask_.resize(static_cast<size_t>(kPrefixLen) * kPrefixLen);
    p_cos_.resize(static_cast<size_t>(kPrefixLen) * kHeadDim);
    p_sin_.resize(static_cast<size_t>(kPrefixLen) * kHeadDim);
    f4d_.resize(static_cast<size_t>(kActionHorizon) * (kPrefixLen + kActionHorizon));
    s_cos_.resize(static_cast<size_t>(kActionHorizon) * kHeadDim);
    s_sin_.resize(static_cast<size_t>(kActionHorizon) * kHeadDim);
    x_t_.resize(static_cast<size_t>(kActionHorizon) * kActionDim);
    v_t_.resize(static_cast<size_t>(kActionHorizon) * kActionDim);
    memset(kv_dev_, 0, sizeof(kv_dev_));

    status_ = Status::kOk;
}

Pi05::~Pi05() {
    // kv_dev_ only aliases the dkv segments' output buffers, and the loop below releases
    // those; freeing them here as well would hand the same device addresses back twice.
    Net* nets[6] = {&siglip_, &dkv_[0], &dkv_[1], &ddn_[0], &ddn_[1], &ddn_[2]};
    for (Net* net : nets) {
        if (!net->in_t && !net->out_t) continue;
        for (int i = 0; i < net->nin; i++) bm_free_device_mem(handle_, net->in_t[i].device_mem.u.device.device_addr);
        for (int i = 0; i < net->nout; i++) bm_free_device_mem(handle_, net->out_t[i].device_mem.u.device.device_addr);
        free(net->in_t);
        free(net->out_t);
        free(net->name);
    }
    if (bmrt_) bmrt_destroy(bmrt_);
    if (handle_) bm_dev_free(handle_);
}

void Pi05::fail(Status status, const std::string& msg) {
    if (status_ != Status::kOk) return;   // keep the first failure, it carries the root cause
    status_ = status;
    err_ = msg;
}

std::vector<float> Pi05::make_noise(uint64_t seed) {
    const size_t n = static_cast<size_t>(kActionHorizon) * kActionDim;
    std::vector<float> out(n);
    uint64_t state = seed + 0x9E3779B97F4A7C15ull;

    // SplitMix64, mapped into the open interval (0,1).
    auto next_unit = [&state]() {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z = z ^ (z >> 31);
        return (static_cast<double>(z >> 11) + 0.5) / 9007199254740992.0;
    };

    // Box-Muller, two draws at a time. The official sampler draws N(0,1), not a uniform
    // range: substituting a uniform one changes the action chunk it converges to.
    for (size_t i = 0; i < n; i += 2) {
        const double radius = std::sqrt(-2.0 * std::log(next_unit()));
        const double angle = 2.0 * 3.14159265358979323846 * next_unit();
        out[i] = static_cast<float>(radius * std::cos(angle));
        if (i + 1 < n) out[i + 1] = static_cast<float>(radius * std::sin(angle));
    }
    return out;
}

bool Pi05::load_net(const std::string& path, Net* net) {
    int n_before = bmrt_get_network_number(bmrt_);
    if (!bmrt_load_bmodel(bmrt_, path.c_str())) {
        fail(Status::kModelLoadFail, "load bmodel failed: " + path);
        return false;
    }
    int n_after = bmrt_get_network_number(bmrt_);
    const char** names = nullptr;
    bmrt_get_network_names(bmrt_, &names);
    if (!names || n_after <= n_before) {
        fail(Status::kModelLoadFail, "no network in bmodel: " + path);
        return false;
    }
    const char* net_name = names[n_before];
    net->info = bmrt_get_network_info(bmrt_, net_name);
    if (!net->info) {
        fail(Status::kModelLoadFail, "bmrt_get_network_info failed: " + path);
        return false;
    }
    net->stage = &net->info->stages[0];
    net->name = strdup(net_name);
    net->nin = net->info->input_num;
    net->nout = net->info->output_num;
    std::cout << "[pi05] net " << net_name << " in=" << net->nin << " out=" << net->nout
              << std::endl;
    return true;
}

bool Pi05::net_bufs_init(Net* net) {
    if (net->bufs_ready) return true;
    net->in_t = static_cast<bm_tensor_t*>(calloc(net->nin, sizeof(bm_tensor_t)));
    net->out_t = static_cast<bm_tensor_t*>(calloc(net->nout, sizeof(bm_tensor_t)));
    if (!net->in_t || !net->out_t) {
        fail(Status::kDeviceMemFail, "tensor buffer alloc failed");
        return false;
    }
    // Caching these per network matters: allocating them on every call makes a long-running
    // process grow without bound (hundreds of alloc/free pairs per inference) until it OOMs.
    for (int i = 0; i < net->nin; i++) {
        size_t want = net->info->max_input_bytes[i];
        unsigned long long pa = 0;
        if (bm_malloc_device_mem(handle_, &pa, 0, want)) {
            fail(Status::kDeviceMemFail, "device input mem alloc failed");
            return false;
        }
        net->in_t[i].dtype = net->info->input_dtypes[i];
        net->in_t[i].shape = net->stage->input_shapes[i];
        net->in_t[i].st_mode = BM_STORE_1N;
        net->in_t[i].device_mem = bm_mem_from_device(pa, want);
    }
    for (int i = 0; i < net->nout; i++) {
        size_t want = net->info->max_output_bytes[i];
        unsigned long long pa = 0;
        if (bm_malloc_device_mem(handle_, &pa, 0, want)) {
            fail(Status::kDeviceMemFail, "device output mem alloc failed");
            return false;
        }
        net->out_t[i].dtype = net->info->output_dtypes[i];
        net->out_t[i].shape = net->stage->output_shapes[i];
        net->out_t[i].st_mode = BM_STORE_1N;
        net->out_t[i].device_mem = bm_mem_from_device(pa, want);
    }
    net->bufs_ready = true;
    return true;
}

bool Pi05::net_forward(Net* net, void** in_bufs, const bm_device_mem_t* in_dev,
                       void** out_bufs, size_t* out_sizes, int keep_out,
                       bm_device_mem_t* out_dev) {
    if (net->nin > kMaxIO || net->nout > kMaxIO) {
        fail(Status::kInvalidArgument, "network has too many inputs or outputs");
        return false;
    }
    if (!net_bufs_init(net)) return false;

    bm_tensor_t tin[kMaxIO], tout[kMaxIO];
    for (int i = 0; i < net->nin; i++) {
        // Copy into a local tensor: writing the device alias back into net->in_t would
        // permanently pin that input to device memory for every later call.
        tin[i] = net->in_t[i];
        if (in_dev && in_dev[i].size > 0) {
            tin[i].device_mem = in_dev[i];
        } else if (bm_memcpy_s2d(handle_, net->in_t[i].device_mem, in_bufs[i])) {
            fail(Status::kForwardFail, "host to device copy failed");
            return false;
        }
    }
    for (int i = 0; i < net->nout; i++) tout[i] = net->out_t[i];

    bool ok = bmrt_launch_tensor_ex(bmrt_, net->name, tin, net->nin, tout, net->nout, true, false);
    bm_thread_sync(handle_);
    // The runtime does not rewrite user tensors for static graphs; this write-back is only a
    // safety net for dynamic graphs that relocate output memory.
    for (int i = 0; i < net->nout; i++) net->out_t[i].device_mem = tout[i].device_mem;
    if (!ok) {
        fail(Status::kForwardFail, std::string("launch failed: ") + net->name);
        return false;
    }
    for (int i = 0; i < net->nout; i++) {
        out_sizes[i] = net->info->max_output_bytes[i];
        if (keep_out) {
            out_dev[i] = tout[i].device_mem;
            out_bufs[i] = nullptr;
        } else {
            out_bufs[i] = malloc(out_sizes[i]);
            if (!out_bufs[i] || bm_memcpy_d2s(handle_, out_bufs[i], tout[i].device_mem)) {
                // Release what this call already handed out, so a failed copy does not leak.
                for (int j = 0; j <= i; j++) {
                    free(out_bufs[j]);
                    out_bufs[j] = nullptr;
                    out_sizes[j] = 0;
                }
                fail(Status::kForwardFail, "device to host copy failed");
                return false;
            }
        }
    }
    return true;
}

bool Pi05::run_siglip(const float* obs, std::vector<float>* feats) {
    void* in_bufs[1] = {const_cast<float*>(obs)};
    void* out_bufs[1] = {nullptr};
    size_t out_sizes[1] = {0};
    if (!net_forward(&siglip_, in_bufs, nullptr, out_bufs, out_sizes, 0, nullptr)) return false;
    // batch=2: both views come out of a single forward pass, so the 571 MB of weights is
    // fetched from DDR once instead of twice.
    const size_t count = static_cast<size_t>(kNumViews) * kTokensPerView * kHidden;
    feats->assign(static_cast<float*>(out_bufs[0]), static_cast<float*>(out_bufs[0]) + count);
    free(out_bufs[0]);
    return true;
}

bool Pi05::load_task(int task_id, const std::string& data_dir) {
    if (status_ != Status::kOk) return false;
    if (task_id < 0 || task_id > 9) {
        fail(Status::kInvalidArgument, "task_id must be in 0..9");
        return false;
    }
    char tag[8];
    snprintf(tag, sizeof(tag), "t%02d", task_id);
    const std::string base = data_dir + "/prefix_assets/" + tag + "_";

    std::vector<size_t> shape;
    // Loads one .npy asset and checks its element count.
    // Only the count matters, not the numpy shape: the assets are stored with whatever
    // leading batch dimensions the producer used (e.g. p_amask is [1,1,536,536] on the
    // device but [536,536] in a hand-built set), and the bmodel takes a flat buffer either
    // way. Counting avoids rejecting an asset that is in fact correct.
    auto load = [&](const char* what, std::vector<float>* dst, size_t want) -> bool {
        const std::string path = base + what + ".npy";
        if (!npy_load_f32(path, dst, &shape)) {
            fail(Status::kAssetLoadFail, "asset load failed: " + path);
            return false;
        }
        if (dst->size() != want) {
            std::ostringstream oss;
            oss << path << " has " << dst->size() << " elements, expected " << want;
            fail(Status::kAssetShapeMismatch, oss.str());
            return false;
        }
        return true;
    };

    const size_t pl = static_cast<size_t>(kPrefixLen);
    const size_t hd = static_cast<size_t>(kHeadDim);
    const size_t ah = static_cast<size_t>(kActionHorizon);
    if (!load("p_amask", &p_amask_, pl * pl)) return false;
    if (!load("p_cos", &p_cos_, pl * hd)) return false;
    if (!load("p_sin", &p_sin_, pl * hd)) return false;
    if (!load("f4d", &f4d_, ah * (pl + ah))) return false;
    if (!load("s_cos", &s_cos_, ah * hd)) return false;
    if (!load("s_sin", &s_sin_, ah * hd)) return false;

    // The language prefix holds only the real tokens (L = 16..21 depending on the task). The
    // remaining slots stay zero: p_amask masks them out entirely, so their value cannot affect
    // the output.
    const std::string prompt_path = data_dir + "/prefix_assets/promptL_" + tag + ".npy";
    if (!npy_load_f32(prompt_path, &prompt_, &shape)) {
        fail(Status::kAssetLoadFail, "asset load failed: " + prompt_path);
        return false;
    }
    if (prompt_.empty() || prompt_.size() % kHidden != 0 ||
        prompt_.size() / kHidden > static_cast<size_t>(kPrefixLen) - kVisTokens) {
        std::ostringstream oss;
        oss << prompt_path << " holds " << prompt_.size()
            << " elements; expected a multiple of " << kHidden << " and at most "
            << (static_cast<size_t>(kPrefixLen) - kVisTokens) * kHidden;
        fail(Status::kAssetShapeMismatch, oss.str());
        return false;
    }
    // Unnormalization coefficients live with the dataset, next to prefix_assets/.
    // The keys are named mean/std in the upstream asset, but the values are the quantile affine
    // coefficients: action = x * scale + bias. Using them as x * std + mean makes the first six
    // dimensions off by a factor of 2 to 3.
    const std::string unnorm = data_dir + "/action_unnorm.npz";
    if (!npz_load_f32(unnorm, "mean", &unnorm_bias_) ||
        !npz_load_f32(unnorm, "std", &unnorm_scale_) ||
        unnorm_bias_.size() < 7 || unnorm_scale_.size() < 7) {
        fail(Status::kAssetLoadFail,
             "failed to read " + unnorm +
             " (needs STORED zip entries and quantile affine coefficients)");
        return false;
    }

    task_id_ = task_id;
    task_loaded_ = true;
    return true;
}

bool Pi05::denoise_step(const float* suffix_in, float time, float* suffix_out) {
    float* carry = const_cast<float*>(suffix_in);
    float* v = nullptr;
    size_t v_bytes = 0;
    for (int s = 0; s < 3; s++) {
        Net* seg = &ddn_[s];
        void* in_bufs[kMaxIO];
        bm_device_mem_t in_dev[kMaxIO];
        memset(in_dev, 0, sizeof(in_dev));
        int use_dev = 0;
        // Inputs are matched by name rather than by position: the three segments declare
        // different layer ranges, so p_k{L}/p_v{L} resolve to different KV slots per segment.
        for (int i = 0; i < seg->nin; i++) {
            const char* nm = seg->info->input_names[i];
            if (strstr(nm, "suffix_in")) {
                in_bufs[i] = carry;
            } else if (strstr(nm, "time")) {
                in_bufs[i] = &time;
            } else if (strstr(nm, "f4d")) {
                in_bufs[i] = f4d_.data();
            } else if (strcmp(nm, "s_cos") == 0) {
                in_bufs[i] = s_cos_.data();
            } else if (strcmp(nm, "s_sin") == 0) {
                in_bufs[i] = s_sin_.data();
            } else if (strncmp(nm, "p_k", 3) == 0 || strncmp(nm, "p_v", 3) == 0) {
                int layer = atoi(nm + 3);
                if (layer < 0 || layer >= kNumLayers) {
                    fail(Status::kInvalidArgument, std::string("bad kv input name: ") + nm);
                    return false;
                }
                int idx = (nm[2] == 'k') ? layer * 2 : layer * 2 + 1;
                if (kv_dev_[idx].size == 0) {
                    fail(Status::kForwardFail, "KV not resident: dkv forward must run first");
                    return false;
                }
                in_dev[i] = kv_dev_[idx];
                use_dev = 1;
            } else {
                fail(Status::kInvalidArgument, std::string("unmatched input name: ") + nm);
                return false;
            }
        }
        void* out_bufs[4] = {nullptr, nullptr, nullptr, nullptr};
        size_t out_sizes[4] = {0, 0, 0, 0};
        if (!net_forward(seg, in_bufs, use_dev ? in_dev : nullptr, out_bufs, out_sizes, 0,
                         nullptr)) {
            return false;
        }
        if (s < 2) {
            if (carry != suffix_in) free(carry);   // release the previous segment output
            carry = static_cast<float*>(out_bufs[0]);
            for (int i = 1; i < seg->nout; i++) free(out_bufs[i]);
        } else {
            v = static_cast<float*>(out_bufs[0]);
            v_bytes = out_sizes[0];
            for (int i = 1; i < seg->nout; i++) free(out_bufs[i]);
        }
    }
    const size_t want = static_cast<size_t>(kActionHorizon) * kActionDim * 4;
    if (!v || v_bytes < want) {
        fail(Status::kForwardFail, "v_t output smaller than expected");
        if (v) free(v);
        if (carry != suffix_in) free(carry);
        return false;
    }
    memcpy(suffix_out, v, want);
    free(v);
    if (carry != suffix_in) free(carry);
    return true;
}

bool Pi05::infer(const std::vector<uint8_t>& agentview, const std::vector<uint8_t>& wrist,
                 const std::vector<uint8_t>& right_wrist,
                 const std::vector<float>& noise, int num_steps,
                 std::vector<float>& action, Timing* timing) {
    if (status_ != Status::kOk) return false;
    if (!task_loaded_) {
        fail(Status::kInferWithoutTask, "no task loaded; call load_task() first");
        return false;
    }
    if (num_steps <= 0 || num_steps > 50) {
        fail(Status::kInvalidArgument, "num_steps must be in 1..50");
        return false;
    }
    if (noise.size() != static_cast<size_t>(kActionHorizon) * kActionDim) {
        fail(Status::kInvalidArgument, "noise must hold kActionHorizon * kActionDim floats");
        return false;
    }
    const size_t img_bytes = static_cast<size_t>(3) * kImgPixels;
    if (agentview.size() != img_bytes || wrist.size() != img_bytes) {
        std::ostringstream oss;
        oss << "each view must hold 3*224*224 RGB bytes, got " << agentview.size() << " and "
            << wrist.size();
        fail(Status::kInvalidArgument, oss.str());
        return false;
    }
    if (kNumViews < 3 && !right_wrist.empty()) {
        // Dropping it quietly would leave the output looking fine while half the input went
        // nowhere, so refuse instead. See the camera-slot comment in pi05.h for how to build
        // the three-slot graph.
        std::ostringstream oss;
        oss << "a third camera view was given (" << right_wrist.size() << " bytes), but the "
            << "loaded bmodels have " << kNumViews << " image slots";
        fail(Status::kInvalidArgument, oss.str());
        return false;
    }
    if (kNumViews >= 3 && !right_wrist.empty() && right_wrist.size() != img_bytes) {
        fail(Status::kInvalidArgument, "the third view must hold 3*224*224 RGB bytes");
        return false;
    }

    auto now = []() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
    };
    Timing t;
    const double begin = now();

    // Stage 1: every view into one CHW float batch, in slot order. The dataset stores RGB
    // directly, so no colour conversion is needed here. A view that was not supplied stays
    // zero, matching how the official policy pads a camera that does not exist.
    // Slot order matches the official policy: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb.
    const std::vector<uint8_t>* views[3] = {&agentview, &wrist, &right_wrist};
    std::vector<float> obs(static_cast<size_t>(kNumViews) * 3 * kImgPixels, 0.0f);
    for (int v = 0; v < kNumViews; v++) {
        const std::vector<uint8_t>* img = views[v];
        if (img->empty()) img = nullptr;   // view not supplied: leave the slot zeroed
        if (img) RgbToChw(img->data(), obs.data() + static_cast<size_t>(v) * 3 * kImgPixels);
    }
    const double after_pre = now();
    t.preprocess_ms = after_pre - begin;

    // Stage 2: visual encoder.
    std::vector<float> feats;
    if (!run_siglip(obs.data(), &feats)) return false;
    const double after_sig = now();
    t.siglip_ms = after_sig - after_pre;

    // Stage 3: assemble the prefix and run the two backbone segments, leaving the 36 KV tensors
    // resident on the device.
    {
        memcpy(prefix_embs_.data(), feats.data(), kVisTokens * kHidden * 4);
        size_t lang_tokens = prompt_.size() / kHidden;
        const size_t room = static_cast<size_t>(kPrefixLen) - kVisTokens;
        if (lang_tokens > room) lang_tokens = room;
        if (lang_tokens) {
            memcpy(prefix_embs_.data() + static_cast<size_t>(kVisTokens) * kHidden, prompt_.data(),
                   lang_tokens * kHidden * 4);
        }
        const size_t filled = kVisTokens + lang_tokens;
        if (filled < static_cast<size_t>(kPrefixLen)) {
            memset(prefix_embs_.data() + filled * kHidden, 0,
                   (static_cast<size_t>(kPrefixLen) - filled) * kHidden * 4);
        }
    }
    {
        // Nothing to release here: the dkv segments write the KV into their own output
        // buffers, allocated once by net_bufs_init and reused on every call. Freeing them
        // would return that device memory while net->out_t still points into it.
        bm_device_mem_t kv0[19], kv1[18];
        memset(kv0, 0, sizeof(kv0));
        memset(kv1, 0, sizeof(kv1));
        void* in0[4] = {prefix_embs_.data(), p_amask_.data(), p_cos_.data(), p_sin_.data()};
        void* out0[19];
        size_t os0[19];
        if (!net_forward(&dkv_[0], in0, nullptr, out0, os0, 1, kv0)) return false;

        // The hidden state produced by dkv0 is fed to dkv1 straight from device memory, which
        // avoids a 7.9 MB round trip through the host.
        bm_device_mem_t in1_dev[4];
        memset(in1_dev, 0, sizeof(in1_dev));
        in1_dev[0] = kv0[18];
        void* in1[4] = {nullptr, p_amask_.data(), p_cos_.data(), p_sin_.data()};
        void* out1[18];
        size_t os1[18];
        if (!net_forward(&dkv_[1], in1, in1_dev, out1, os1, 1, kv1)) return false;

        for (int i = 0; i < 18; i++) kv_dev_[i] = kv0[i];        // layers 0-8
        for (int i = 0; i < 18; i++) kv_dev_[18 + i] = kv1[i];   // layers 9-17
    }
    const double after_dkv = now();
    t.dkv_ms = after_dkv - after_sig;

    // Stage 4: denoise loop. Each step runs the three action-expert segments and the host
    // performs one Euler step of the flow ODE.
    x_t_ = noise;
    const float dt = -1.0f / static_cast<float>(num_steps);
    for (int st = 0; st < num_steps; st++) {
        const float time = 1.0f + st * dt;
        if (!denoise_step(x_t_.data(), time, v_t_.data())) return false;
        for (size_t i = 0; i < x_t_.size(); i++) x_t_[i] += dt * v_t_[i];
    }
    const double after_ddn = now();
    t.ddn_ms = after_ddn - after_dkv;

    // Stage 5: quantile affine unnormalization, first 7 dimensions.
    action.assign(static_cast<size_t>(kActionHorizon) * 7, 0.0f);
    for (int j = 0; j < kActionHorizon; j++) {
        for (int d = 0; d < 7; d++) {
            action[j * 7 + d] = x_t_[j * kActionDim + d] * unnorm_scale_[d] + unnorm_bias_[d];
        }
    }
    t.postprocess_ms = now() - after_ddn;
    t.total_ms = now() - begin;
    if (timing) *timing = t;
    return true;
}