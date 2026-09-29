#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""从 GLC 权重导出设备端推理所需的派生资产。

产出（供 python/glc_infer.py 使用）：
    codebook.npy   z_vq 码本 (16384, 256)  —— 编码器与解码器之间的 CPU 最近邻查表
    q_enc.npy      编码端质量向量 (4, 256, 1, 1)
    q_dec.npy      解码端质量向量 (4, 256, 1, 1)

用法:
    PYTHONPATH=tools/stubs python3 tools/export_assets.py \
        --ckpt GLC_image.pth.tar --glc-src GLC_src --out models/
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--glc-src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.glc_src)
    from src.models.image_model import GLC_Image
    from src.utils.test_utils import get_state_dict

    os.makedirs(args.out, exist_ok=True)
    net = GLC_Image(inplace=False)
    net.load_state_dict(get_state_dict(args.ckpt), strict=False)
    net.eval()

    cb = net.z_vq.embedding.weight.detach().numpy().astype(np.float32)
    np.save(os.path.join(args.out, "codebook.npy"), cb)
    np.save(os.path.join(args.out, "q_enc.npy"),
            net.q_enc.detach().numpy().astype(np.float32))
    np.save(os.path.join(args.out, "q_dec.npy"),
            net.q_dec.detach().numpy().astype(np.float32))

    print("codebook %s  q_enc %s  q_dec %s" %
          (cb.shape, tuple(net.q_enc.shape), tuple(net.q_dec.shape)))
    print("已写入", args.out)


if __name__ == "__main__":
    main()
