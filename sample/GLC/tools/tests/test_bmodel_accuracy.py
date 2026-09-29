#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""bmodel 输出与 PyTorch 的精度对齐。

用 tpu-mlir 的 model_runner.py 在 cmodel（模拟器）上跑 bmodel，不需要真机
就能拿到 bmodel 的实际数值输出。

判据按 BF16 定：BF16 只有 8 bit 尾数，相对误差量级约 2^-8 ≈ 4e-3，
不能用 fp32 的 1e-5 去要求它。

依赖 cmodel，跑得慢，因此与 ONNX 对账分开成两个脚本。

用法（先跑 tools/make_bmodel_probes.py 生成输入 npz）：
    PATH=$PWD/.venv-mlir/bin:$PATH .venv-mlir/bin/python tests/test_bmodel_accuracy.py
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # tools/tests -> GLC

PROBE = os.path.join(PROJ, "models", "onnx", "probe_glc_prior_init.npz")
CHIP = os.environ.get("GLC_CHIP", "BM1684X")  # BM1684X(SE7-32) / BM1688(SE9-16)
BMODEL = os.path.join(PROJ, "models", CHIP, "glc_prior_init_bf16.bmodel")
REF = os.path.join(PROJ, "models", "onnx", "ref_glc_prior_init.npz")
RUNNER = os.path.join(PROJ, ".venv-mlir", "bin", "model_runner.py")

# 五个输出与 PyTorch 的对应关系（bmodel 输出名含算子后缀，按顺序对齐）
OUTPUT_NAMES = ["q_enc", "q_dec", "scales", "means", "common"]
_failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def main():
    for path, what in ((PROBE, "输入 probe"), (BMODEL, "bmodel"), (REF, "参考输出")):
        if not os.path.isfile(path):
            print(f"缺少{what}：{path}")
            return 2

    out_path = os.path.join("/tmp", "bmodel_prior_init_out.npz")
    cmd = [RUNNER, "--input", PROBE, "--model", BMODEL, "--output", out_path]
    print("运行 cmodel（可能数分钟）...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.isfile(out_path):
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        print("cmodel 未产出结果")
        return 2

    got = np.load(out_path)
    ref = np.load(REF)
    keys = list(got.files)
    check("bmodel 输出张量个数与子图一致", len(keys) == len(OUTPUT_NAMES),
          f"{len(keys)} 个：{keys}")

    # 判据统一为「绝对误差 / 该张量自身的量级」。不逐元素算相对误差，
    # 是因为接近 0 的元素会把相对误差放大成假失败，而 BF16 的误差本来就是
    # 随量级缩放的（尾数 8 bit）。
    for idx, name in enumerate(OUTPUT_NAMES):
        if idx >= len(keys):
            break
        g = got[keys[idx]]
        r = ref[name]
        if g.shape != r.shape:
            check(f"{name} 形状一致", False, f"bmodel {g.shape} vs torch {r.shape}")
            continue
        scale = float(np.abs(r).max())
        abs_err = float(np.abs(g.astype(np.float64) - r.astype(np.float64)).max())
        rel = abs_err / max(scale, 1e-6)
        check(f"{name} BF16 误差（相对量级）< 2%", rel < 2e-2,
              f"最大绝对误差 {abs_err:.3e}，参考量级 {scale:.3f}，比值 {rel:.3e}")

    print()
    if _failures:
        print(f"失败 {len(_failures)} 项: {_failures}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
