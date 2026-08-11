"""Generate INT8 calibration data for TAPNext++ via a PyTorch recurrent rollout.

Outputs per-step inputs as ``.npz`` files (one per sample, keyed by the exact
ONNX input names) plus two ``data_list`` txt files consumed by
``run_calibration.py --data_list`` (see TPU-MLIR ``calibration/data_selector.py``:
case 2 — single ``.npz`` per sample, keys must match ``module.input_names``).

The 24 recurrent cache tensors are dumped from a *real* rollout (init on frame
0, step on frames 1..K, state fed back each step), not random values — this is
essential so the state-dependent RG-LRU / CausalConv1D layers see realistic
activations during calibration.

By default frames are random uniform [-1,1] NCHW float32 (same convention as
``export_onnx.validate``).  Pass ``--frames_npy`` to use real preprocessed video
frames (BGR->RGB, bilinear resize to model_size, ``x/127.5 - 1.0``); the npz
format is unchanged.  Real video is strongly recommended — random frames give
the ViT backbone wrong activation ranges, causing large INT8 quantization error
even with mixed-precision prediction heads.

Usage (in the torch 1.13 export venv)::

    # Real-video calibration (recommended):
    python tools/gen_cali_data.py \
        --ckpt models/tapnextpp_ckpt.pt \
        --frames_npy datasets/calib_frames.npy \
        --out-dir datasets/cali_data \
        --model-size 256 --num-queries 1 \
        --num-seqs 10 --seq-len 12

    # Random-frame calibration (POC only):
    python tools/gen_cali_data.py \
        --ckpt models/tapnextpp_ckpt.pt \
        --out-dir datasets/cali_data \
        --model-size 256 --num-queries 1 \
        --num-seqs 10 --seq-len 12

Produces ``datasets/cali_data/{init,step}/*.npz`` and
``datasets/cali_data/init_cali.txt`` / ``step_cali.txt``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# reuse the export helpers (load_model, flatten_state, cache_input_names, ...)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_onnx as eo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/tapnextpp_ckpt.pt")
    ap.add_argument("--out-dir", default="datasets/cali_data")
    ap.add_argument("--model-size", type=int, default=256)
    ap.add_argument("--num-queries", type=int, default=1)
    ap.add_argument("--num-seqs", type=int, default=10, help="number of independent rollouts")
    ap.add_argument("--seq-len", type=int, default=12, help="frames per rollout")
    ap.add_argument("--frames_npy", default=None,
                    help="preprocessed real-video frames [N,3,H,W] float32 [-1,1]; "
                         "if given, replaces random frames for calibration")
    args = ap.parse_args()

    H = args.model_size
    Q = args.num_queries
    out = Path(args.out_dir)
    (out / "init").mkdir(parents=True, exist_ok=True)
    (out / "step").mkdir(parents=True, exist_ok=True)

    # Load real-video frames if provided, else use random
    real_frames = None
    if args.frames_npy:
        real_frames = np.load(args.frames_npy)  # [N, 3, H, W] float32
        print(f"[cali] using {len(real_frames)} real-video frames from {args.frames_npy}")
    else:
        print("[cali] WARNING: using random frames — INT8 accuracy will likely degrade")

    model = eo.load_model(args.ckpt, H)
    cache_names = eo.cache_input_names()  # ["rg_lru_0","conv1d_0", ...] (24)
    assert len(cache_names) == 24, len(cache_names)

    init_lines, step_lines = [], []
    n_init = n_step = 0
    with torch.no_grad():
        for s in range(args.num_seqs):
            # a fresh rollout: new frames + a new query-point location
            torch.manual_seed(1000 + s)
            if real_frames is not None:
                # sliding window over real video, cycling if not enough frames
                n_real = len(real_frames)
                offset = (s * max(1, n_real // args.num_seqs)) % n_real
                frames = []
                for k in range(args.seq_len):
                    f = real_frames[(offset + k) % n_real]
                    frames.append(torch.from_numpy(f).unsqueeze(0).float())
            else:
                frames = [(torch.rand(1, 3, H, H) * 2 - 1).float() for _ in range(args.seq_len)]
            qp = torch.zeros(1, Q, 3).float()
            qp[..., 0] = 0.0                       # t = 0 (query born on frame 0)
            qp[..., 1] = float(torch.randint(0, H, (1,)))  # y (model pixels)
            qp[..., 2] = float(torch.randint(0, H, (1,)))  # x (model pixels)

            # --- init graph: frame 0 ---
            v0 = frames[0].permute(0, 2, 3, 1).unsqueeze(0)
            _, _, _, state = model(video=v0, query_points=qp, state=None)
            np.savez(out / "init" / f"seq{s}_f0.npz",
                     frame=frames[0].numpy(), query_points=qp.numpy())
            init_lines.append(str((out / "init" / f"seq{s}_f0.npz").resolve()))
            n_init += 1

            # --- step graph: frames 1..K-1, state fed back ---
            for k in range(1, args.seq_len):
                flat = eo.flatten_state(state)  # [step, query_points, *24 caches]
                step_t, qp_s = flat[0], flat[1]
                caches = flat[2:]               # 24 tensors
                assert len(caches) == 24
                step_npz = {str(nm): c.numpy() for nm, c in zip(cache_names, caches)}
                np.savez(out / "step" / f"seq{s}_f{k}.npz",
                         frame=frames[k].numpy(),
                         step=step_t.numpy(),
                         query_points=qp_s.numpy(),
                         **step_npz)
                step_lines.append(str((out / "step" / f"seq{s}_f{k}.npz").resolve()))
                n_step += 1
                # advance state
                vk = frames[k].permute(0, 2, 3, 1).unsqueeze(0)
                _, _, _, state = model(video=vk, state=state)
            print(f"[seq {s+1}/{args.num_seqs}] {n_init} init + {n_step} step samples")

    (out / "init_cali.txt").write_text("\n".join(init_lines) + "\n")
    (out / "step_cali.txt").write_text("\n".join(step_lines) + "\n")
    print(f"\n[done] {n_init} init samples -> {out/'init_cali.txt'}")
    print(f"[done] {n_step} step samples -> {out/'step_cali.txt'}")
    print("feed to run_calibration.py via --data_list <txt> --input_num N")


if __name__ == "__main__":
    main()
