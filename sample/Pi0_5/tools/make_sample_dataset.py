#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""make_sample_dataset.py — renders the fixed-seed observation set for the Pi0_5 example.

Produces the `obs/` half of the dataset:

    datasets/pi05_libero_sample/
    ├── obs/tXX_initY/{agentview.npy,wrist.npy}   # uint8 [224,224,3] RGB
    └── index.json                                # task_id / init_state / seed / prompt per case

The environment protocol is the one openpi's own LIBERO evaluation uses, so the
observations match what the official policy was measured on: seed the environment,
reset, restore the recorded initial state, step a dummy action ten times (the simulator
drops objects on the first frames), then rotate 180 degrees and resize_with_pad to 224.

The matching `noise/` and `actions_gt/` come from make_reference.py, which needs the
openpi environment rather than the LIBERO one.

Usage (inside the LIBERO evaluation environment):
    python3 make_sample_dataset.py --out ../datasets/pi05_libero_sample \
                                   --inits 5 --seed 7 \
                                   --prefix-assets /path/to/dkva536_npy
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

# openpi's LIBERO evaluation protocol.
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_WAIT_STEPS = 10
RENDER_RESOLUTION = 256
OBS_SIZE = 224
TASK_SUITE = "libero_spatial"
NUM_TASKS = 10


def make_env(task, seed):
    """Creates the offscreen LIBERO environment for one task.

    Args:
        task: a LIBERO benchmark task.
        seed: environment seed; it affects object placement even with a fixed initial state.

    Returns:
        The environment object.
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RENDER_RESOLUTION,
                             camera_widths=RENDER_RESOLUTION)
    env.seed(seed)
    return env


def main():
    """Entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--inits", type=int, default=5,
                    help="initial states per task (5 gives 50 cases in total)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prefix-assets", default=None,
                    help="directory holding the per-task prefix assets; copied verbatim "
                         "into the dataset when given")
    args = ap.parse_args()

    from libero.libero import benchmark
    from openpi_client import image_tools

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()

    index = []
    for task_id in range(NUM_TASKS):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env = make_env(task, args.seed)
        for init in range(args.inits):
            env.reset()
            obs = env.set_init_state(np.asarray(init_states[init]))
            for _ in range(NUM_WAIT_STEPS):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            # Stored as .npy so the example needs no image decoding library on the target.
            case = f"t{task_id:02d}_init{init}"
            case_dir = os.path.join(args.out, "obs", case)
            os.makedirs(case_dir, exist_ok=True)
            for name, key in (("agentview", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")):
                img = np.ascontiguousarray(obs[key][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, OBS_SIZE, OBS_SIZE))
                np.save(os.path.join(case_dir, name + ".npy"), img)

            index.append({"case": case, "task_id": task_id, "init_state": init,
                          "seed": args.seed, "prompt": task.language})
            print(f"[{case}] {task.language[:60]}", flush=True)
        env.close()

    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "inits_per_task": args.inits, "task_suite": TASK_SUITE,
                   "action_horizon": 10, "action_dim": 32, "cases": index},
                  f, ensure_ascii=False, indent=2)

    if args.prefix_assets:
        import shutil
        dst = os.path.join(args.out, "prefix_assets")
        os.makedirs(dst, exist_ok=True)
        for name in sorted(os.listdir(args.prefix_assets)):
            shutil.copy2(os.path.join(args.prefix_assets, name), os.path.join(dst, name))
        print(f"prefix assets copied from {args.prefix_assets}")

    print(f"done: {len(index)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
