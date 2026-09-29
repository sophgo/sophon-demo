#!/usr/bin/env python3
"""生成逐子图跨芯片一致性验证所需的探针输入。

用真实图像走一遍编码链，把每个子图的真实输入存成 npz：
  z_hat                  -> glc_prior_init 的输入
  stage{N}_y, stage{N}_common -> glc_stage_N 的输入（N=1,2,3）
  tail_y, tail_q_dec     -> glc_tail 的输入
  x                      -> glc_encoder 的输入

必须用真实链路的值，不能用 randn：这些子图含 GroupNorm，
随机输入会产生 1e10 量级激活，测不出真实精度差异。
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GLC = os.path.dirname(HERE)
sys.path.insert(0, GLC)
sys.path.insert(0, os.path.join(GLC, "python"))
sys.path.insert(0, os.path.join(GLC, "tools", "tests"))

IMAGE = os.path.join(GLC, "datasets", "kodak24", "kodim01.png")


def main():
    from model_loader import load_glc
    from test_onnx_alignment import build_real_chain

    net = load_glc(os.path.join(GLC, "models", "GLC_image.pth.tar"),
                   os.path.join(GLC, "GLC_src"),
                   os.path.join(GLC, "tools", "stubs"))
    chain = build_real_chain(net, IMAGE)

    out = {
        "z_hat": chain["z_hat"].numpy().astype(np.float32),
        "tail_y": chain["y_hat"].numpy().astype(np.float32),
        "tail_q_dec": chain["q_dec"].numpy().astype(np.float32),
    }
    # stage 1/2/3 的输入是前若干段重建之和，逐段落盘
    for i, si in enumerate(chain["stage_inputs"], start=1):
        out["stage%d_y" % i] = si.numpy().astype(np.float32)
    common = chain["common"].numpy().astype(np.float32)
    for i in (1, 2, 3):
        out["stage%d_common" % i] = common

    path = os.path.join(HERE, "mix_probes.npz")
    np.savez(path, **out)
    for k in sorted(out):
        a = out[k]
        print("%-16s %-22s range [%.4g, %.4g]" % (k, a.shape, a.min(), a.max()))
    print("已写入", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
