#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""compare_acc.py — action-trajectory comparison against the official reference.

Metrics follow the convention shared by NVIDIA Jetson AI Lab, D-Robotics and FlashRT:
  - primary: cosine similarity, reported as overall plus per-timestep mean/min/max
  - gate: cos >= 0.999 is OK, >= 0.99 is the acceptable floor

Usage:
  # single case
  python3 compare_acc.py --pred results/action.npy --gt datasets/.../t00_init0.npy
  # whole directory, paired by file name
  python3 compare_acc.py --pred_dir results/ --gt_dir datasets/pi05_libero_sample/actions_gt/
"""
import argparse
import glob
import os
import sys

import numpy as np


def load(path):
    """Loads an action file as a 2-D [horizon, 7] float64 array.

    Args:
        path: path to a .npy file.

    Returns:
        Array of shape [time, 7].
    """
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 7)
    return arr.astype(np.float64)


def per_timestep_cos(pred, gt):
    """Computes the cosine similarity of each time step separately.

    A single overall cosine can hide degradation in the tail steps, so the
    per-step values are reported alongside it.

    Args:
        pred: predicted actions, shape [time, dim].
        gt: reference actions, shape [time, dim].

    Returns:
        Array of per-step cosine similarities.
    """
    out = []
    for t in range(min(len(pred), len(gt))):
        a, b = pred[t], gt[t]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out.append(float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0)
    return np.array(out)


def compare(pred, gt):
    """Compares one prediction against its reference.

    Args:
        pred: predicted actions, shape [time, dim].
        gt: reference actions, shape [time, dim].

    Returns:
        Dict with cos, rel, mae and the per-timestep mean/min/max.
    """
    n = min(pred.shape[0], gt.shape[0])
    pred, gt = pred[:n], gt[:n]
    flat_pred, flat_gt = pred.ravel(), gt.ravel()
    cos = float(flat_pred @ flat_gt / (np.linalg.norm(flat_pred) * np.linalg.norm(flat_gt)))
    rel = float(np.linalg.norm(flat_pred - flat_gt) / (np.linalg.norm(flat_gt) + 1e-12))
    mae = float(np.mean(np.abs(flat_pred - flat_gt)))
    per_step = per_timestep_cos(pred, gt)
    return dict(cos=cos, rel=rel, mae=mae,
                ts_mean=float(per_step.mean()),
                ts_min=float(per_step.min()),
                ts_max=float(per_step.max()))


def verdict(cos, warn, floor):
    """Maps a cosine value onto the pass/warn/fail verdict.

    Args:
        cos: overall cosine similarity.
        warn: threshold for a clean pass.
        floor: lowest acceptable value.

    Returns:
        One of "OK", "WARN" or "FAIL".
    """
    if cos >= warn:
        return "OK"
    if cos >= floor:
        return "WARN"
    return "FAIL"


def build_pairs(args):
    """Builds the list of (gt, pred, name) triples to compare.

    Args:
        args: parsed command line arguments.

    Returns:
        List of (gt_path, pred_path, display_name).
    """
    if args.pred and args.gt:
        return [(args.gt, args.pred, os.path.basename(args.pred))]
    pairs = []
    for gt in sorted(glob.glob(os.path.join(args.gt_dir, "*.npy"))):
        name = os.path.basename(gt)
        pred = os.path.join(args.pred_dir, name)
        if os.path.exists(pred):
            pairs.append((gt, pred, name))
        else:
            print(f"[skip] no prediction for {name}")
    return pairs


def main():
    """Entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="single prediction .npy")
    ap.add_argument("--gt", help="single reference .npy")
    ap.add_argument("--pred_dir", help="directory of predictions")
    ap.add_argument("--gt_dir", help="directory of references")
    ap.add_argument("--warn", type=float, default=0.999, help="cos threshold for OK")
    ap.add_argument("--floor", type=float, default=0.99, help="lowest acceptable cos")
    args = ap.parse_args()

    if not (args.pred and args.gt) and not (args.pred_dir and args.gt_dir):
        ap.error("give either --pred/--gt or --pred_dir/--gt_dir")
    pairs = build_pairs(args)
    if not pairs:
        print("no pairs to compare", file=sys.stderr)
        return 2

    header = (f"{'case':<20} {'cos':>10} {'rel L2':>10} {'MAE':>10} "
              f"{'ts mean':>10} {'ts min':>10} {'ts max':>10}  status")
    print(header)
    print("-" * len(header))

    worst = None
    results = []
    for gt_path, pred_path, name in pairs:
        try:
            result = compare(load(pred_path), load(gt_path))
        except (OSError, ValueError) as exc:
            print(f"{name:<20} load/compare failed: {exc}")
            continue
        status = verdict(result["cos"], args.warn, args.floor)
        print(f"{name:<20} {result['cos']:>10.6f} {result['rel'] * 100:>9.3f}% "
              f"{result['mae']:>10.5f} {result['ts_mean']:>10.6f} {result['ts_min']:>10.6f} "
              f"{result['ts_max']:>10.6f}  {status}")
        results.append(result)
        if worst is None or result["cos"] < worst[1]["cos"]:
            worst = (name, result, status)

    if worst is None:
        return 2
    name, result, status = worst

    # The headline numbers for the README table: cosine averaged over cases, plus the
    # worst single timestep seen anywhere in the set.
    cos_all = np.array([r["cos"] for r in results])
    rel_all = np.array([r["rel"] for r in results])
    ts_mean_all = np.array([r["ts_mean"] for r in results])
    ts_min_all = np.array([r["ts_min"] for r in results])
    ts_max_all = np.array([r["ts_max"] for r in results])
    print()
    print(f"summary over {len(results)} cases:")
    print(f"  overall cos        mean {cos_all.mean():.6f}   min {cos_all.min():.6f}   "
          f"max {cos_all.max():.6f}")
    print(f"  per-timestep cos   mean {ts_mean_all.mean():.6f}   min {ts_min_all.min():.6f}   "
          f"max {ts_max_all.max():.6f}")
    print(f"  rel L2             mean {rel_all.mean() * 100:.3f}%")
    print(f"worst case: {name}  cos={result['cos']:.6f}  -> {status}")
    print(f"gate: cos >= {args.warn} OK / >= {args.floor} WARN / below FAIL")
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
