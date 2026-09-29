#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""导出带中间输出的编码器 ONNX，用于定位 fp16 下第一个出现 NaN 的层。

vq_enc.blocks 是一个 ModuleList，这里把每个子块的输出都挂成 ONNX 输出，
编译成 fp16 bmodel 后在设备上跑一次即可看出从哪一块开始发散。

用法:
    PYTHONPATH=tools/stubs python3 tools/export_probe.py --out onnx_probe
"""
import argparse
import os
import sys

import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc-src", default="GLC_src")
    ap.add_argument("--ckpt", default="GLC_image.pth.tar")
    ap.add_argument("--out", default="onnx_probe")
    ap.add_argument("--img-h", type=int, default=512)
    ap.add_argument("--img-w", type=int, default=768)
    ap.add_argument("--image", default=None, help="真实测试图（必须用真实图，随机噪声会让 enc 发散）")
    ap.add_argument("--norm-fix", action="store_true", help="对 AttnBlock 的 GroupNorm 启用输入预缩放")
    args = ap.parse_args()

    sys.path.insert(0, args.glc_src)
    from src.models.image_model import GLC_Image
    from src.utils.test_utils import get_state_dict

    os.makedirs(args.out, exist_ok=True)
    H, W = args.img_h, args.img_w

    net = GLC_Image(inplace=False)
    net.load_state_dict(get_state_dict(args.ckpt), strict=True)
    net.eval()

    if args.norm_fix:
        class ScaledGroupNorm(nn.Module):
            SCALE = 32.0

            def __init__(self, norm):
                super().__init__()
                self.norm = norm

            def forward(self, x):
                return self.norm(x / self.SCALE)

        n = 0
        for m in net.vqgan.encoder.modules():
            if type(m).__name__ == "AttnBlock":
                m.norm = ScaledGroupNorm(m.norm)
                n += 1
        print("已对 %d 个 AttnBlock 启用 GroupNorm 输入预缩放" % n)

    vq_enc = net.vqgan.encoder
    blocks = vq_enc.blocks

    class Probe(nn.Module):
        """走真实 forward，用 hook 抓取 vq_enc 各级激活，外加最终 y、z。

        注意不能自己重写 forward —— Encoder.forward 末尾还有 quant_conv，
        漏掉它会让 enc 收到错误输入并直接发散成 NaN。
        """

        def __init__(self, net):
            super().__init__()
            self.net = net
            self.captured = []
            for blk in net.vqgan.encoder.blocks:
                blk.register_forward_hook(self._hook)

        def _hook(self, module, inp, out):
            self.captured.append(out)

        def forward(self, x, q_enc):
            self.captured = []
            n = self.net
            y = n.enc(n.vqgan.encoder(x), q_enc)
            z = n.hyper_enc(y)
            return tuple(self.captured) + (y, z)

    probe = Probe(net).eval()
    n_blk = len(blocks)
    names = ["blk_%02d" % i for i in range(n_blk)] + ["y", "z"]
    print("vq_enc.blocks 级数:", n_blk)

    if args.image:
        from PIL import Image
        import numpy as np
        img = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
        if img.shape[:2] != (H, W):
            img = np.asarray(Image.fromarray(img).resize((W, H), Image.BICUBIC))
        x = ((img.astype(np.float32) / 255.0 - 0.5) * 2.0).transpose(2, 0, 1)[None]
        x = torch.from_numpy(np.ascontiguousarray(x))
        print("使用真实图像:", args.image)
    else:
        x = torch.randn(1, 3, H, W)
        print("警告: 未指定 --image，随机噪声会让 enc 发散出 NaN")

    q = net.q_enc[0:1].detach().clone()
    with torch.no_grad():
        ref = probe(x, q)
    for nm, t in zip(names, ref):
        print("  %-8s %s  max|.|=%.1f" % (nm, tuple(t.shape), t.abs().max().item()))

    torch.onnx.export(
        probe, (x, q), os.path.join(args.out, "glc_probe.onnx"),
        input_names=["x", "q_enc"], output_names=names,
        opset_version=13, do_constant_folding=True, dynamo=False)
    p = os.path.join(args.out, "glc_probe.onnx")
    print("导出:", p, round(os.path.getsize(p) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
