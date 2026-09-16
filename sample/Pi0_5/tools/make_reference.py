#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""make_reference.py — produces the official-policy ground truth for the Pi0_5 example.

Consumes the `obs/` set written by make_sample_dataset.py and adds the other two halves
of the dataset:

    <out>/noise/tXX_initY.npy        [10,32] float32, the initial noise fed in
    <out>/actions_gt/tXX_initY.npy   [10,7]  float32, official policy action chunk

The reference is the official pi0.5 PyTorch checkpoint run through openpi's own policy
stack: the same model code, the same converted weights and the same 2-step Euler sampler
the bmodels were exported from. Three things are pinned so that the number measures the
port rather than the setup:

  * float32 - the precision the ONNX graph was exported at;
  * the initial noise - written to noise/ so pi05_bmcv.soc --noise can be given the exact
    same draw; comparing two different draws would measure the sampler, not the port;
  * the step count - must equal --num_steps on the device side.

Both sides then produce unnormalized [10,7] action chunks and are directly comparable.

The two halves are separate scripts because they need different environments: this one
wants openpi (torch, jax, sentencepiece), make_sample_dataset.py wants LIBERO (robosuite,
MuJoCo).

Usage:
    python3 make_reference.py --dataset ../datasets/pi05_libero_sample --steps 2
"""
import argparse
import json
import os
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import datetime

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

import numpy as np

PI05_ROOT = os.environ.get("PI05_ROOT")
if not PI05_ROOT:
    raise SystemExit("PI05_ROOT is not set; see tools/export/README.md")
sys.path.insert(0, f"{PI05_ROOT}/openpi-ref/src")
sys.path.insert(0, f"{PI05_ROOT}/openpi-ref/packages/openpi-client/src")

import torch  # noqa: E402
from openpi.models_pytorch import pi0_pytorch as _pp  # noqa: E402

# The ONNX graph is exported in float32, so the reference is taken in float32 too.
_pp.get_safe_dtype = lambda t, d: torch.float32

from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

NOISE_SEED = 20260916


def noise_for(case_idx):
    """Draws the initial noise for one case.

    Deterministic in the case index so the whole reference is reproducible, and N(0,1)
    because that is what the official sampler draws.

    Args:
        case_idx: position of the case in index.json.

    Returns:
        float32 array of shape [10, 32].
    """
    return np.random.RandomState(NOISE_SEED + case_idx).standard_normal((10, 32)).astype(np.float32)


def main():
    """Entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="dataset dir holding index.json and obs/")
    ap.add_argument("--out", default=None, help="defaults to --dataset")
    ap.add_argument("--cases", type=int, default=0, help="limit to the first N cases (0 = all)")
    ap.add_argument("--steps", type=int, default=2,
                    help="denoise steps; must match --num_steps on the device side "
                         "(the sample default is 2, the official evaluation default is 10)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = library default)")
    ap.add_argument("--policy-dir", default=f"{PI05_ROOT}/models/pi05_libero_official_pt",
                    help="PyTorch checkpoint directory produced by the official converter")
    args = ap.parse_args()
    out = args.out or args.dataset

    if args.threads:
        torch.set_num_threads(args.threads)

    with open(os.path.join(args.dataset, "index.json"), encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    if args.cases:
        cases = cases[:args.cases]

    pol = policy_config.create_trained_policy(
        _config.get_config("pi05_libero"), args.policy_dir, default_prompt=None,
        pytorch_device="cpu", sample_kwargs={"num_steps": args.steps})
    model = pol._model
    model.float()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    os.makedirs(os.path.join(out, "actions_gt"), exist_ok=True)
    os.makedirs(os.path.join(out, "noise"), exist_ok=True)

    for i, case in enumerate(cases):
        name = case["case"]
        obs_dir = os.path.join(args.dataset, "obs", name)
        element = {
            "observation/image": np.load(os.path.join(obs_dir, "agentview.npy")),
            "observation/wrist_image": np.load(os.path.join(obs_dir, "wrist.npy")),
            # pi0.5 does not take the robot state as an input (openpi only embeds it for
            # the older pi0), so the value here cannot affect the reference.
            "observation/state": np.zeros(8, np.float32),
            "prompt": case["prompt"],
        }
        noise = noise_for(i)
        actions = np.asarray(pol.infer(element, noise=noise)["actions"], dtype=np.float32)
        np.save(os.path.join(out, "actions_gt", name + ".npy"), actions)
        np.save(os.path.join(out, "noise", name + ".npy"), noise)
        print(f"[{i + 1}/{len(cases)}] {name} {actions.shape}", flush=True)

    print(f"done: {len(cases)} cases -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
