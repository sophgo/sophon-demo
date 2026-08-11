//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tapnext.h"
#include <cassert>
#include <cstring>
#include <stdexcept>

#ifndef FFALIGN
#define FFALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#endif

using std::cout;
using std::endl;
using std::string;
using std::vector;

TAPNext::TAPNext(const string& init_bmodel, const string& step_bmodel, int dev_id)
    : dev_id_(dev_id), init_bmodel_(init_bmodel), step_bmodel_(step_bmodel) {
    // get handle
    auto ret = bm_dev_request(&handle, dev_id);
    assert(BM_SUCCESS == ret);

    // judge now is pcie or soc
    ret = bm_get_misc_info(handle, &misc_info);
    assert(BM_SUCCESS == ret);

    in_cache_.assign(TAPNEXT_NUM_CACHES, -1);
    out_cache_.assign(TAPNEXT_NUM_CACHES, -1);

    // set temp timestamp
    m_ts = &tmp_ts;
}

TAPNext::~TAPNext() {
    free_net();
    if (imgs_created_) {
        bm_image_destroy(resized_img_);
        bm_image_destroy(converto_img_);
    }
    if (qp_dev_mem_.size != 0) bm_free_device(handle, qp_dev_mem_);
    if (step_dev_mem_.size != 0) bm_free_device(handle, step_dev_mem_);
    release_held_caches();
    bm_dev_free(handle);
}

// ---------------------------------------------------------------------------
//  Network loading + name resolution
// ---------------------------------------------------------------------------
int TAPNext::load_net(const string& bmodel, bool is_init) {
    // create bmrt
    bmrt = bmrt_create(handle);
    if (!bmrt_load_bmodel(bmrt, bmodel.c_str())) {
        cout << "load bmodel(" << bmodel << ") failed" << endl;
        return -1;
    }

    // get network names from bmodel, keep only network 0
    const char** names;
    int num = bmrt_get_network_number(bmrt);
    if (num > 1) {
        cout << "This bmodel has " << num << " networks, only network 0 is used." << endl;
    }
    bmrt_get_network_names(bmrt, &names);
    net_name = names[0];
    free(names);

    // get netinfo by netname
    netinfo = bmrt_get_network_info(bmrt, net_name.c_str());
    if (netinfo->stage_num > 1) {
        cout << "This bmodel has " << netinfo->stage_num << " stages, only stage 0 is used." << endl;
    }

    // resolve input/output indices
    in_frame_ = find_input("frame");
    in_qp_ = find_input("query_points");
    in_step_ = is_init ? -1 : find_input("step");
    out_tracks_ = find_output("tracks");
    out_vis_ = find_output("vis_logits");
    for (int i = 0; i < TAPNEXT_NUM_BLOCKS; ++i) {
        out_cache_[i] = find_output("new_rg_lru_" + std::to_string(i));
        out_cache_[TAPNEXT_NUM_BLOCKS + i] = find_output("new_conv1d_" + std::to_string(i));
        if (!is_init) {
            in_cache_[i] = find_input("rg_lru_" + std::to_string(i));
            in_cache_[TAPNEXT_NUM_BLOCKS + i] = find_input("conv1d_" + std::to_string(i));
        }
    }
    if (in_frame_ < 0 || in_qp_ < 0 || out_tracks_ < 0 || out_vis_ < 0 ||
        (!is_init && in_step_ < 0)) {
        return -1;
    }

    // create input tensor shells; device_mem is attached per frame
    input_tensors.resize(netinfo->input_num);
    for (int i = 0; i < netinfo->input_num; ++i) {
        input_tensors[i].dtype = netinfo->input_dtypes[i];
        input_tensors[i].shape = netinfo->stages[0].input_shapes[i];
        input_tensors[i].st_mode = BM_STORE_1N;
        input_tensors[i].device_mem = bm_mem_null();
    }
    output_tensors.resize(netinfo->output_num);
    return 0;
}

void TAPNext::free_net() {
    if (bmrt != NULL) {
        bmrt_destroy(bmrt);
        bmrt = NULL;
        netinfo = NULL;
    }
    input_tensors.clear();
    output_tensors.clear();
}

int TAPNext::find_input(const string& name) {
    for (int i = 0; i < netinfo->input_num; ++i)
        if (name == netinfo->input_names[i]) return i;
    cout << "ERROR: input '" << name << "' not found" << endl;
    return -1;
}

int TAPNext::find_output(const string& prefix) {
    // TPU-MLIR renames outputs to <onnx_name>_<op>_f32 — match by prefix.
    for (int i = 0; i < netinfo->output_num; ++i) {
        string oname(netinfo->output_names[i]);
        if (oname.rfind(prefix, 0) == 0) return i;
    }
    cout << "ERROR: output prefix '" << prefix << "' not found" << endl;
    return -1;
}

// ---------------------------------------------------------------------------
//  Preprocessing:  BGR/YUV bm_image -> [1,3,256,256] float32 [-1,1] on device
// ---------------------------------------------------------------------------
bm_device_mem_t TAPNext::pre_process(const bm_image& bgr) {
    // Create preprocessing images once (batch=1, reused across frames).
    if (!imgs_created_) {
        int net_h = TAPNEXT_MODEL_SIZE, net_w = TAPNEXT_MODEL_SIZE;
        int aligned_w = FFALIGN(net_w, 64);
        int strides[3] = {aligned_w, aligned_w, aligned_w};
        auto ret = bm_image_create(handle, net_h, net_w, FORMAT_RGB_PLANAR,
                                   DATA_TYPE_EXT_1N_BYTE, &resized_img_, strides);
        assert(ret == BM_SUCCESS);

        // The FP16 bmodel was compiled without --mean/--scale, so the frame
        // input is raw normalized float32.
        ret = bm_image_create(handle, net_h, net_w, FORMAT_RGB_PLANAR,
                              DATA_TYPE_EXT_FLOAT32, &converto_img_, NULL);
        assert(ret == BM_SUCCESS);

        // alpha/beta for x / 127.5 - 1.0  ->  [-1, 1]
        float alpha = 1.0f / 127.5f;
        float beta = -1.0f;
        memset(&converto_attr_, 0, sizeof(converto_attr_));
        converto_attr_.alpha_0 = alpha;
        converto_attr_.alpha_1 = alpha;
        converto_attr_.alpha_2 = alpha;
        converto_attr_.beta_0 = beta;
        converto_attr_.beta_1 = beta;
        converto_attr_.beta_2 = beta;
        imgs_created_ = true;
    }

    // 1. resize + color-convert (any input format -> RGB_PLANAR 256x256)
    auto ret = bmcv_image_vpp_convert(handle, 1, bgr, &resized_img_);
    assert(ret == BM_SUCCESS);

    // 2. normalize: uint8 -> float32,  x * alpha + beta  ->  [-1, 1]
    ret = bmcv_image_convert_to(handle, 1, converto_attr_, &resized_img_, &converto_img_);
    assert(ret == BM_SUCCESS);

    // 3. get device mem handle (the image owns the memory)
    bm_device_mem_t dev_mem;
    bm_image_get_contiguous_device_mem(1, &converto_img_, &dev_mem);
    return dev_mem;
}

// ---------------------------------------------------------------------------
//  Inference
// ---------------------------------------------------------------------------
void TAPNext::set_input_host(int input_idx, const float* host_data, size_t count,
                             bm_device_mem_t& dev_mem) {
    size_t bytes = count * sizeof(float);
    if (dev_mem.size == 0) {
        auto ret = bm_malloc_device_byte(handle, &dev_mem, bytes);
        assert(ret == BM_SUCCESS);
    }
    assert(dev_mem.size >= bytes);
    bm_memcpy_s2d_partial(handle, dev_mem, (void*)host_data, bytes);
    input_tensors[input_idx].device_mem = dev_mem;
}

int TAPNext::forward() {
    bool ok = bmrt_launch_tensor(bmrt, net_name.c_str(), input_tensors.data(),
                                 netinfo->input_num, output_tensors.data(),
                                 netinfo->output_num);
    assert(ok == true);
    auto ret = bm_thread_sync(handle);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    return 0;
}

float* TAPNext::get_cpu_data(bm_tensor_t* tensor, float scale) {
    int ret = 0;
    float* pFP32 = NULL;
    int count = bmrt_shape_count(&tensor->shape);
    if (misc_info.pcie_soc_mode == 1) {  // soc
        if (tensor->dtype == BM_FLOAT32) {
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            if (ret != BM_SUCCESS) throw std::runtime_error("BMRuntime 操作失败");
            ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            if (ret != BM_SUCCESS) throw std::runtime_error("BMRuntime 操作失败");
            pFP32 = (float*)addr;
        } else if (BM_INT8 == tensor->dtype) {
            int8_t* pI8 = nullptr;
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            if (ret != BM_SUCCESS) throw std::runtime_error("BMRuntime 操作失败");
            ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            if (ret != BM_SUCCESS) throw std::runtime_error("BMRuntime 操作失败");
            pI8 = (int8_t*)addr;
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            for (int i = 0; i < count; ++i) pFP32[i] = pI8[i] * scale;
            ret = bm_mem_unmap_device_mem(handle, pI8, bm_mem_get_device_size(tensor->device_mem));
            if (ret != BM_SUCCESS) throw std::runtime_error("BMRuntime 操作失败");
        } else {
            std::cerr << "unsupport dtype: " << tensor->dtype << std::endl;
        }
    } else {  // pcie
        if (tensor->dtype == BM_FLOAT32) {
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(handle, pFP32, tensor->device_mem, count * sizeof(float));
            assert(BM_SUCCESS == ret);
        } else if (BM_INT8 == tensor->dtype) {
            int tensor_size = bmrt_tensor_bytesize(tensor);
            int8_t* pI8 = new int8_t[tensor_size];
            assert(pI8 != nullptr);
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(handle, pI8, tensor->device_mem, tensor_size);
            assert(BM_SUCCESS == ret);
            for (int i = 0; i < count; ++i) pFP32[i] = pI8[i] * scale;
            delete[] pI8;
        } else {
            std::cerr << "unsupport dtype: " << tensor->dtype << std::endl;
        }
    }
    return pFP32;
}

void TAPNext::release_cpu_data(bm_tensor_t* tensor, float* p) {
    if (p == NULL) return;
    if (misc_info.pcie_soc_mode == 1 && tensor->dtype == BM_FLOAT32) {  // soc mmap
        bm_mem_unmap_device_mem(handle, p, bm_mem_get_device_size(tensor->device_mem));
    } else {
        delete[] p;
    }
}

void TAPNext::read_tracks_vis(int num_queries, float* tracks, float* vis) {
    // tracks: [1, 1, Q, 2] -> Q*2 floats
    float* tracks_data = get_cpu_data(&output_tensors[out_tracks_],
                                      netinfo->output_scales[out_tracks_]);
    memcpy(tracks, tracks_data, num_queries * 2 * sizeof(float));
    release_cpu_data(&output_tensors[out_tracks_], tracks_data);

    // vis: [1, 1, Q, 1] -> Q floats
    float* vis_data = get_cpu_data(&output_tensors[out_vis_],
                                   netinfo->output_scales[out_vis_]);
    memcpy(vis, vis_data, num_queries * sizeof(float));
    release_cpu_data(&output_tensors[out_vis_], vis_data);

    // free + null so the next launch allocates fresh output mems
    for (int idx : {out_tracks_, out_vis_}) {
        if (output_tensors[idx].device_mem.size != 0) {
            bm_free_device(handle, output_tensors[idx].device_mem);
            output_tensors[idx].device_mem = bm_mem_null();
        }
    }
}

void TAPNext::hold_cache_outputs() {
    if ((int)held_cache_mems_.size() != TAPNEXT_NUM_CACHES)
        held_cache_mems_.assign(TAPNEXT_NUM_CACHES, bm_mem_null());
    for (int i = 0; i < TAPNEXT_NUM_CACHES; ++i) {
        bm_device_mem_t& m = output_tensors[out_cache_[i]].device_mem;
        assert(m.size != 0);                 // launch must have allocated it
        assert(held_cache_mems_[i].size == 0);  // previous set already released
        held_cache_mems_[i] = m;
        // Null the tensor field: the next launch must allocate FRESH output
        // buffers.  (These mems become the next step's inputs — letting the
        // runtime write outputs into them in place would alias input/output.)
        m = bm_mem_null();
    }
}

void TAPNext::release_held_caches() {
    for (auto& m : held_cache_mems_) {
        if (m.size != 0) {
            bm_free_device(handle, m);
            m = bm_mem_null();
        }
    }
}

// ---------------------------------------------------------------------------
//  Full rollout
// ---------------------------------------------------------------------------
int TAPNext::track(const vector<bm_image>& frames, const float* query_points,
                   int num_queries, vector<float>& tracks_out, vector<float>& vis_out) {
    int n = (int)frames.size();
    assert(n >= 1);
    tracks_out.resize((size_t)n * num_queries * 2);
    vis_out.resize((size_t)n * num_queries);

    // =================== frame 0: init graph ===================
    cout << "[TAPNext] loading init bmodel: " << init_bmodel_ << endl;
    if (load_net(init_bmodel_, true) != 0) return -1;

    m_ts->save("preprocess time");
    bm_device_mem_t frame_mem = pre_process(frames[0]);
    m_ts->save("preprocess time");

    cout << "[TAPNext] running init graph on frame 0" << endl;
    m_ts->save("init inference time");
    input_tensors[in_frame_].device_mem = frame_mem;
    set_input_host(in_qp_, query_points, (size_t)num_queries * 3, qp_dev_mem_);
    forward();
    m_ts->save("init inference time");

    m_ts->save("postprocess time");
    read_tracks_vis(num_queries, tracks_out.data(), vis_out.data());
    // Keep the 24 cache output mems alive: they are caller-owned (allocated
    // by bmrt_launch_tensor against `handle`), so they survive free_net()
    // and become the step graph's first cache inputs — no host round trip.
    hold_cache_outputs();
    m_ts->save("postprocess time");

    // free init graph (SE9: can't hold both bmodels simultaneously)
    free_net();

    // =================== frames 1..N: step graph ===================
    cout << "[TAPNext] loading step bmodel: " << step_bmodel_ << endl;
    if (load_net(step_bmodel_, false) != 0) return -1;

    for (int k = 1; k < n; ++k) {
        m_ts->save("preprocess time");
        frame_mem = pre_process(frames[k]);
        m_ts->save("preprocess time");

        m_ts->save("step inference time");
        input_tensors[in_frame_].device_mem = frame_mem;
        float step_val = (float)k;
        set_input_host(in_step_, &step_val, 1, step_dev_mem_);
        set_input_host(in_qp_, query_points, (size_t)num_queries * 3, qp_dev_mem_);
        // 24 caches: zero-copy feedback — previous outputs become inputs
        for (int i = 0; i < TAPNEXT_NUM_CACHES; ++i)
            input_tensors[in_cache_[i]].device_mem = held_cache_mems_[i];
        forward();  // allocates fresh output mems (all output fields are null)
        m_ts->save("step inference time");

        m_ts->save("postprocess time");
        release_held_caches();  // previous cache inputs consumed by the TPU
        float* trk = tracks_out.data() + (size_t)k * num_queries * 2;
        float* vis = vis_out.data() + (size_t)k * num_queries;
        read_tracks_vis(num_queries, trk, vis);
        hold_cache_outputs();
        m_ts->save("postprocess time");

        if (k % 10 == 0 || k == n - 1)
            cout << "[TAPNext] step " << k << "/" << (n - 1) << endl;
    }

    release_held_caches();
    // free step graph
    free_net();
    return 0;
}
