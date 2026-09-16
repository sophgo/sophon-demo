//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef PI05_H
#define PI05_H

#include <cstdint>
#include <string>
#include <vector>

#include "bmdef.h"
#include "bmlib_runtime.h"
#include "bmruntime_interface.h"

// Single-shot inference for pi0.5: observation (camera views + task id) -> action chunk.
//
// One inference consists of four stages:
//   1. siglip        all views in one batch -> visual features (weights loaded once)
//   2. dkv x2        visual features + language prefix -> 36 KV tensors, kept resident on device
//   3. ddn x N       denoise loop; each step runs the 3 action-expert segments in order and
//                    the host performs the Euler integration
//   4. unnormalize   quantile affine transform, first 7 dimensions kept
class Pi05 {
public:
    // ---- camera slots ----------------------------------------------------------------
    // pi0.5 feeds PaliGemma three image slots, named after what they hold:
    //     base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
    // openpi's LIBERO policy zero-fills the third one and sets its image mask to False
    // (see openpi/src/openpi/policies/libero_policy.py), so its 256 tokens can never reach
    // the output -- that is the whole reason the prefix here is 536 tokens and not 968.
    //
    // The sample ships bmodels compiled for the first two slots. Everything downstream is
    // written in terms of kNumViews, so wiring up a real third camera is:
    //     1. set kNumViews = 3 here,
    //     2. re-export siglip with a [kNumViews,3,224,224] input and recompile it
    //        (scripts/gen_siglip_bmodel_mlir.sh, input_shapes in the mlir step),
    //     3. regenerate the prefix assets -- they carry one KV/rope set per prefix length
    //        (gen_kvseg_pertask.py -> gen_assets_536.py) and recompile dkv0_9 / dkv9_18.
    // Until then, a third view has nowhere to go and infer() rejects it rather than
    // silently dropping half the input.
    static constexpr int kNumViews = 2;
    static constexpr int kTokensPerView = 256;   // 224/14 = 16 patches per side, squared
    static constexpr int kVisTokens = kNumViews * kTokensPerView;
    static constexpr int kLangSlots = 24;        // longest LIBERO prompt is 21 tokens
    static constexpr int kPrefixLen = kVisTokens + kLangSlots;   // 536 at kNumViews == 2

    static constexpr int kActionHorizon = 10;
    static constexpr int kActionDim = 32;    // raw action width; only the first 7 are exposed
    static constexpr int kHidden = 2048;
    static constexpr int kHeadDim = 256;
    static constexpr int kNumLayers = 18;
    static constexpr int kKvCount = kNumLayers * 2;   // one K/V pair per layer
    static constexpr int kNumSteps = 2;               // default denoise steps

    // Unified error codes. Every failing entry point returns one of these so callers can tell
    // which stage failed without parsing message text.
    enum class Status {
        kOk = 0,
        kDeviceOpenFail,
        kRuntimeCreateFail,
        kModelLoadFail,
        kAssetLoadFail,
        kAssetShapeMismatch,
        kDeviceMemFail,
        kForwardFail,
        kInvalidArgument,
        kInferWithoutTask,
    };

    struct Timing {
        double preprocess_ms = 0;
        double siglip_ms = 0;
        double dkv_ms = 0;
        double ddn_ms = 0;      // total over all denoise steps
        double postprocess_ms = 0;
        double total_ms = 0;
    };

    // Constructs the engine and loads all six bmodels plus the unnormalization statistics.
    //
    // in:  dev_id    TPU device id
    //      model_dir directory holding the six bmodels
    Pi05(int dev_id, const std::string& model_dir);
    ~Pi05();

    Pi05(const Pi05&) = delete;
    Pi05& operator=(const Pi05&) = delete;

    // Whether construction succeeded. Check this before calling any other method.
    //
    // out: true if the engine is usable
    bool ok() const { return status_ == Status::kOk; }

    // Last error code, and the human-readable message that goes with it.
    //
    // out: error code / message of the most recent failure
    Status status() const { return status_; }
    const std::string& last_error() const { return err_; }

    // Loads the per-task prefix assets: p_amask / p_cos / p_sin / f4d / s_cos / s_sin / promptL.
    // Must be called before infer(); call it again to switch task.
    //
    // in:  task_id  task index 0..9, selects which prefix asset set to load
    //      data_dir dataset root containing prefix_assets/ and action_unnorm.npz
    // out: true on success
    bool load_task(int task_id, const std::string& data_dir);

    // Deterministic N(0,1) noise from a fixed seed, so the same observation always yields
    // exactly the same action and results are reproducible. Matches the official sampler's
    // distribution; uses SplitMix64 + Box-Muller rather than <random> to avoid differences
    // between standard library implementations.
    //
    // in:  seed  noise seed
    // out: kActionHorizon * kActionDim floats drawn from N(0,1)
    static std::vector<float> make_noise(uint64_t seed);

    // Runs one inference.
    //
    // in:  agentview    third-person view (slot base_0_rgb): 224*224*3 RGB bytes, row-major
    //      wrist       wrist view (slot left_wrist_0_rgb), same layout
    //      right_wrist second wrist view (slot right_wrist_0_rgb), same layout. Must be
    //                  empty while kNumViews is 2; when kNumViews is 3 an empty vector is
    //                  zero-filled, which is what the official LIBERO policy does for a
    //                  camera that does not exist.
    //      noise       kActionHorizon * kActionDim floats; use make_noise() for reproducibility
    //      num_steps   denoise steps, 1..50
    // out: action      kActionHorizon * 7 floats, already unnormalized
    //      timing      per-stage wall-clock times, may be nullptr
    //      return      true on success
    bool infer(const std::vector<uint8_t>& agentview, const std::vector<uint8_t>& wrist,
               const std::vector<uint8_t>& right_wrist,
               const std::vector<float>& noise, int num_steps,
               std::vector<float>& action, Timing* timing);

private:
    // One loaded network: its bmodel metadata plus reusable device-side I/O buffers.
    struct Net {
        const bm_net_info_t* info = nullptr;
        const bm_stage_info_t* stage = nullptr;
        char* name = nullptr;
        bm_tensor_t* in_t = nullptr;
        bm_tensor_t* out_t = nullptr;
        int nin = 0;
        int nout = 0;
        bool bufs_ready = false;
    };

    // Loads one bmodel and caches its network metadata.
    //
    // in:  path  bmodel file path
    // out: net   populated network handle
    //      return true on success
    bool load_net(const std::string& path, Net* net);

    // Allocates the device-side input/output tensors for one network, once, and reuses them
    // afterwards.
    //
    // in:  net  network whose buffers should be created
    // out: return true on success
    bool net_bufs_init(Net* net);

    // Runs one forward pass.
    //
    // in:  net         network to run
    //      in_bufs     host input pointers; an entry is ignored when in_dev[i] is set
    //      in_dev      if non-null, in_dev[i].size > 0 means input i is read straight from that
    //                  device memory, skipping the host-to-device copy
    //      keep_out    non-zero keeps all outputs on device and reports them via out_dev,
    //                  leaving out_bufs[i] as nullptr (ownership stays with net->out_t)
    // out: out_bufs    host output pointers, caller frees each non-null entry
    //      out_sizes   byte size of each output
    //      out_dev     device memory of each output when keep_out is set
    //      return      true on success
    bool net_forward(Net* net, void** in_bufs, const bm_device_mem_t* in_dev,
                     void** out_bufs, size_t* out_sizes, int keep_out,
                     bm_device_mem_t* out_dev);

    // Runs the three action-expert segments for one denoise step.
    //
    // in:  suffix_in  current action latent, kActionHorizon * kActionDim floats
    //      time       current flow-matching time
    // out: suffix_out velocity field v_t, kActionHorizon * kActionDim floats
    //      return     true on success
    bool denoise_step(const float* suffix_in, float time, float* suffix_out);

    // Runs the visual encoder on the two-view batch.
    //
    // in:  obs   2 * 3 * 224 * 224 floats, CHW, normalised to [-1, 1]
    // out: feats 2 * 256 * kHidden visual features
    //      return true on success
    bool run_siglip(const float* obs, std::vector<float>* feats);

    // Records a failure code and message. The first failure wins, so the root cause is kept.
    //
    // in:  status  error code to record
    //      msg     human-readable detail
    void fail(Status status, const std::string& msg);

    bm_handle_t handle_ = nullptr;
    void* bmrt_ = nullptr;
    Status status_ = Status::kOk;
    std::string err_;
    std::string model_dir_;
    int dev_id_ = 0;

    Net siglip_, dkv_[2], ddn_[3];

    // Device-resident KV produced by the dkv segments and shared with every ddn segment.
    // These alias the dkv segments' own output buffers; this class does not own them.
    bm_device_mem_t kv_dev_[kKvCount];

    // Host-side buffers, allocated once and reused across calls.
    std::vector<float> prefix_embs_;   // kPrefixLen * kHidden
    std::vector<float> p_amask_;       // kPrefixLen * kPrefixLen
    std::vector<float> p_cos_, p_sin_; // kPrefixLen * kHeadDim
    std::vector<float> f4d_;           // kActionHorizon * (kPrefixLen + kActionHorizon)
    std::vector<float> s_cos_, s_sin_; // kActionHorizon * kHeadDim
    std::vector<float> prompt_;        // L * kHidden, real language tokens only
    std::vector<float> unnorm_scale_;  // 7
    std::vector<float> unnorm_bias_;   // 7
    std::vector<float> x_t_;           // kActionHorizon * kActionDim
    std::vector<float> v_t_;           // kActionHorizon * kActionDim

    int task_id_ = -1;
    bool task_loaded_ = false;
};

#endif  // PI05_H
