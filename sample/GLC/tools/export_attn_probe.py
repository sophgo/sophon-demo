#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""AttnBlock 内部逐算子探测，用于定位 fp16 下第一个发散的算子。

对指定的 AttnBlock 复刻 forward_patch 的每一步并分别输出，编译成 fp16 后
在设备上跑一次就能看出是 norm / bmm / softmax 哪一步先出 NaN。

用法:
    PYTHONPATH=tools/stubs python3 tools/export_attn_probe.py --blocks 14 16 18 \
        --image images/kodim01.png --out onnx_attn
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc-src", default="GLC_src")
    ap.add_argument("--ckpt", default="GLC_image.pth.tar")
    ap.add_argument("--out", default="onnx_attn")
    ap.add_argument("--blocks", type=int, nargs="+", default=[14, 16, 18])
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.glc_src)
    from src.models.image_model import GLC_Image
    from src.utils.test_utils import get_state_dict
    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    net = GLC_Image(inplace=False)
    net.load_state_dict(get_state_dict(args.ckpt), strict=True)
    net.eval()

    blks = net.vqgan.encoder.blocks
    for i in args.blocks:
        assert type(blks[i]).__name__ == "AttnBlock", \
            "blocks[%d] 是 %s，不是 AttnBlock" % (i, type(blks[i]).__name__)

    class AttnProbe(nn.Module):
        """复刻 AttnBlock.forward_patch，并把每步中间量作为输出。"""

        def __init__(self, block):
            super().__init__()
            self.norm = block.norm
            self.q, self.k, self.v = block.q, block.k, block.v
            self.proj_out = block.proj_out
            self.patch = block.patch

        def forward(self, x):
            h_ = self.norm(x)
            Q, K, V = self.q(h_), self.k(h_), self.v(h_)
            H_ = torch.zeros_like(x)
            d = self.patch
            b, c, H, W = Q.shape
            for i in range(0, H, d):
                for j in range(0, W, d):
                    q, k, v = Q[:, :, i:i+d, j:j+d], K[:, :, i:i+d, j:j+d], V[:, :, i:i+d, j:j+d]
                    bb, cc, hh, ww = q.shape
                    q = q.reshape(bb, cc, hh*ww).permute(0, 2, 1)
                    k = k.reshape(bb, cc, hh*ww)
                    w_ = torch.bmm(q, k)
                    w_ = w_ * (int(c) ** (-0.5))
                    w_ = F.softmax(w_, dim=2)
                    v = v.reshape(bb, cc, hh*ww)
                    w_ = w_.permute(0, 2, 1)
                    h_ = torch.bmm(v, w_).reshape(bb, cc, hh, ww)
                    H_[:, :, i:i+d, j:j+d] = h_
            out = self.proj_out(H_)
            return h_, H_, out, x + out, Q, K, V, w_

    # 前置块：跑到目标块之前
    probe = AttnProbe(blks[args.blocks[0]]).eval()

    img = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    x = np.ascontiguousarray(((img.astype(np.float32)/255.0 - 0.5)*2.0).transpose(2, 0, 1)[None])
    xt = torch.from_numpy(x)

    # 取第一个目标块的输入
    with torch.no_grad():
        h = xt
        for i in range(args.blocks[0]):
            h = blks[i](h)
        ref = probe(h)
    names = ["attn_h_bmm", "attn_H", "attn_proj", "attn_out", "Q", "K", "V", "w_softmax"]
    for nm, t in zip(names, ref):
        print("  %-12s %s max|.|=%.2f" % (nm, tuple(t.shape), t.abs().max().item()))

    torch.onnx.export(probe, (h,), os.path.join(args.out, "attn_probe.onnx"),
                      input_names=["x_in"], output_names=names,
                      opset_version=13, do_constant_folding=True, dynamo=False)
    p = os.path.join(args.out, "attn_probe.onnx")
    print("导出:", p, round(os.path.getsize(p)/1e6, 1), "MB")
    print("注：输入是 blocks[%d] 的输入张量，需另行保存" % args.blocks[0])
    np.save(os.path.join(args.out, "attn_probe_input.npy"), h.numpy().astype(np.float32))
    print("输入已保存: attn_probe_input.npy", h.shape)


if __name__ == "__main__":
    main()
