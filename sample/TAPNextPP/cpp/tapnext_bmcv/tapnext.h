//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef TAPNEXT_H
#define TAPNEXT_H

#include <iostream>
#include <vector>
#include <string>
#include "utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bm_wrapper.hpp"

#define TAPNEXT_MODEL_SIZE 256
#define TAPNEXT_NUM_BLOCKS 12
#define TAPNEXT_NUM_CACHES (TAPNEXT_NUM_BLOCKS * 2)  // 24 = 12 rg_lru + 12 conv1d

/// TAPNext++ two-graph recurrent point tracker (C++ BMRT/BMCV port).
///
/// init graph : (frame, query_points)           -> tracks, vis, 24 caches
/// step graph : (frame, step, qp, *24 caches)   -> tracks, vis, 24 caches
///
/// The 24 cache tensors (12 blocks x (rg_lru_state, conv1d_state)) carry the
/// recurrent state and are fed back each step.  The two bmodels are loaded
/// sequentially (init is freed before step is loaded) because SE9's CPU RAM
/// cannot hold both graphs at once.  Cache feedback is zero-copy: output
/// device mems allocated by bmrt_launch_tensor are owned by the caller (see
/// bmruntime_interface.h) and stay valid across the bmrt reload, so each
/// step's cache outputs are bound directly as the next step's cache inputs —
/// no host round trip.
class TAPNext {
    bm_handle_t handle;
    bm_misc_info misc_info;
    int dev_id_;
    std::string init_bmodel_, step_bmodel_;

    // Only one graph is resident at a time (init, then step).
    void *bmrt = NULL;
    const bm_net_info_t *netinfo = NULL;
    std::string net_name;

    // BMCV preprocessing buffers (one frame at a time, batch=1)
    bm_image resized_img_{};
    bm_image converto_img_{};
    bool imgs_created_ = false;
    bmcv_convert_to_attr converto_attr_{};

    // device mems for non-image inputs (allocated once, reused across steps)
    bm_device_mem_t qp_dev_mem_ = bm_mem_null();    // query_points [1,Q,3]
    bm_device_mem_t step_dev_mem_ = bm_mem_null();  // step scalar [1]

    // 24 cache device mems carried between steps (zero-copy feedback).
    // Each is the previous launch's output device mem; owned here.
    std::vector<bm_device_mem_t> held_cache_mems_;

    // tensors of the currently loaded graph (device_mem filled per frame)
    std::vector<bm_tensor_t> input_tensors;
    std::vector<bm_tensor_t> output_tensors;

    // input/output indices of the currently loaded graph
    int in_frame_ = -1, in_step_ = -1, in_qp_ = -1;
    std::vector<int> in_cache_;
    int out_tracks_ = -1, out_vis_ = -1;
    std::vector<int> out_cache_;

    TimeStamp tmp_ts;

private:
    /// Load a bmodel, keep only network 0, resolve tensor indices.
    int load_net(const std::string& bmodel, bool is_init);
    void free_net();
    /// Find input tensor index by exact name match.
    int find_input(const std::string& name);
    /// Find output tensor index by prefix match (bmodel renames outputs).
    int find_output(const std::string& prefix);

    /// Preprocess one BGR/YUV bm_image to [1,3,256,256] float32 [-1,1].
    /// Returns the device mem handle (owned by the internal converto image).
    bm_device_mem_t pre_process(const bm_image& bgr);
    /// Launch the current graph and wait for completion.
    int forward();
    /// Copy a host float buffer into an input tensor's device mem.
    void set_input_host(int input_idx, const float* host_data, size_t count,
                        bm_device_mem_t& dev_mem);
    /// Read tracks/vis from output tensors to host; free + null their mems.
    void read_tracks_vis(int num_queries, float* tracks, float* vis);
    /// Take ownership of the 24 cache output mems (nulls the tensor fields so
    /// the next launch allocates fresh output buffers — no in-place aliasing).
    void hold_cache_outputs();
    /// Free the held cache mems (after the TPU has consumed them as inputs).
    void release_held_caches();

    float* get_cpu_data(bm_tensor_t* tensor, float scale);
    void release_cpu_data(bm_tensor_t* tensor, float* p);

public:
    TimeStamp* m_ts = NULL;
    /// \param init_bmodel  path to init graph bmodel
    /// \param step_bmodel  path to step graph bmodel
    /// \param dev_id       TPU device id
    TAPNext(const std::string& init_bmodel, const std::string& step_bmodel,
            int dev_id = 0);
    ~TAPNext();

    /// Track query points across a sequence of BGR bm_image frames.
    /// \param  frames       decoded BGR images (any resolution; resized internally)
    /// \param  query_points [Q, 3] float32 (t, y, x) in model pixels, row-major
    /// \param  num_queries  Q
    /// \param  tracks_out   [T, Q, 2] float32 — per-frame (y, x) in model pixels
    /// \param  vis_out      [T, Q] float32 — visibility logits
    int track(const std::vector<bm_image>& frames, const float* query_points,
              int num_queries, std::vector<float>& tracks_out,
              std::vector<float>& vis_out);
};

#endif  //! TAPNEXT_H
