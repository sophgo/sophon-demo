#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/usr/bin/env python3
"""fix_unnorm_quantile.py — 修正 D 链动作反归一化口径(设备端 /data2/pi05s/demo_work/action_unnorm.npz).

背景(2026-09-13 定位):
  官方 openpi(即 C 列)的反归一化是**分位数仿射**:
      action = x * (q99 - q01) / 2 + (q01 + q99) / 2
  (x 为模型输出的 [-1,1] 归一化动作; 见 openpi transforms.Unnormalize + 官方 norm_stats.json 的 q01/q99)
  而 D 链此前用的是 mean/std:
      action = x * std + mean
  两者在 LIBERO 前 6 维相差 2.1~3.5 倍尺度, gripper 维还有约 -0.13 常量偏置 ——
  表现为 D 列闭环步数约为 C 列 1.8~2 倍、t3-t5 在 max220 步耗尽而失败。

修正后 D 列 libero_spatial task0-5 init0 闭环 6/6, 步数与 C 列逐项一致
(t0 107/107, t1 111/111, t2 130/121, t3 87/88, t4 118/119, t5 96/96)。

用法(在 SE7 上执行):  python3 fix_unnorm_quantile.py
旧资产会自动备份为 action_unnorm.npz.bak_meanstd。修改后需重启 tcp_serve_dkv。
"""
import os as _os

# All inputs and outputs live under PI05_ROOT, the working tree that holds the
# official openpi checkout (openpi-ref/), the official checkpoint (ckpt_official/)
# and the generated artifacts. Set it before running:
#     export PI05_ROOT=/path/to/pi05-work
_PI05_ROOT = _os.environ.get("PI05_ROOT")
if not _PI05_ROOT:
    raise SystemExit("PI05_ROOT is not set; see tools/export/README.md")

import os
import shutil

import numpy as np

P = _os.environ.get("PI05_UNNORM_NPZ")
if not P:
    raise SystemExit("PI05_UNNORM_NPZ is not set; point it at the action_unnorm.npz to fix "
                     "(see tools/export/README.md)")
Q01 = np.array([-0.747375, -0.796125, -0.9375, -0.115803, -0.16943, -0.194502, -1.0])
Q99 = np.array([0.937125, 0.8595, 0.937125, 0.140226, 0.181035, 0.311546, 0.9996])

if not os.path.exists(P + ".bak_meanstd"):
    shutil.copy(P, P + ".bak_meanstd")

d = np.load(P)
mu = d["mean"].astype(np.float32).copy()
sg = d["std"].astype(np.float32).copy()
mu[:7] = ((Q01 + Q99) / 2).astype(np.float32)
sg[:7] = ((Q99 - Q01) / 2).astype(np.float32)
np.savez(P, mean=mu, std=sg)

d2 = np.load(P)
print("新 mean[:7]", np.round(d2["mean"][:7], 6).tolist())
print("新 std [:7]", np.round(d2["std"][:7], 6).tolist())
print("FIX_SAVED")
