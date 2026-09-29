#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""生成 bmodel 精度验收所需的输入与 PyTorch 参考输出。

输入的取法与 tests/test_onnx_alignment.py 一致：走真实图像链路，不用 randn——
这些子图含 GroupNorm，非真实分布的输入会产生 1e10 量级激活，测不出真实精度。

输出：
  models/onnx/probe_glc_prior_init.npz   bmodel 的输入
  models/onnx/ref_glc_prior_init.npz     PyTorch 的参考输出

用法：PYTHONPATH=tools/vendor:. python3 tools/make_bmodel_probes.py
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "python"))
sys.path.insert(0, os.path.join(PROJ, "tools", "tests"))

IMAGE = os.path.join(PROJ, "datasets", "kodak24", "kodim01.png")


def main():
    from model_loader import load_glc
    from test_onnx_alignment import build_real_chain

    net = load_glc(os.path.join(PROJ, "models", "GLC_image.pth.tar"),
                   os.path.join(PROJ, "GLC_src"), os.path.join(PROJ, "tools", "stubs"))
    chain = build_real_chain(net, IMAGE)

    out_dir = os.path.join(PROJ, "models", "onnx")
    os.makedirs(out_dir, exist_ok=True)

    z_hat = chain["z_hat"].numpy().astype(np.float32)
    np.savez(os.path.join(out_dir, "probe_glc_prior_init.npz"), z_hat=z_hat)

    params = chain["params"]
    quant_step, scales, means = params.chunk(3, 1)
    q_step = quant_step.clamp_min(0.5)
    np.savez(
        os.path.join(out_dir, "ref_glc_prior_init.npz"),
        q_enc=(1.0 / q_step).numpy().astype(np.float32),
        q_dec=q_step.numpy().astype(np.float32),
        scales=scales.numpy().astype(np.float32),
        means=means.numpy().astype(np.float32),
        common=chain["common"].numpy().astype(np.float32),
    )
    print(f"z_hat 值域 [{z_hat.min():.3f}, {z_hat.max():.3f}]")
    print(f"已写入 {out_dir}/probe_glc_prior_init.npz 与 ref_glc_prior_init.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
