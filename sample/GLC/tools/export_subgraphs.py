#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""导出 GLC 真熵编码所需的子图 ONNX。

新增子图：
  glc_prior_init    z_hat                 -> q_enc, q_dec, scales, means, common
  glc_stage_1/2/3   y_hat_so_far + common -> scales, means
  glc_tail          y_hat + q_dec         -> x_hat

glc_encoder（x -> y, z）已有导出脚本，这里不重复。

尺寸固定 512x768（沿用现有 bmodel 的静态 shape）：
  x       (1, 3, 512, 768)
  y       (1, 256, 32, 48)      = x / 16
  z_hat   (1, 256, 8, 12)       = x / 64

用法：
    PYTHONPATH=tools/vendor:. .venv-mlir/bin/python export_subgraphs.py --ckpt models/GLC_image.pth.tar
"""

import argparse
import os
import sys

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "python"))

H, W = 512, 768
# y 与 z 的空间尺寸由 vqgan 下采样 16 倍、hyper 再下采样 4 倍决定
Y_H, Y_W = H // 16, W // 16
Z_H, Z_W = H // 64, W // 64
N = 256


class PriorInitExport(nn.Module):
    """z_hat -> 初始 scales/means + common。

    separate_prior 拆成显式输出（官方是 chunk(3,1)）：ONNX 里 5 个输出比
    「一个大张量再在设备端切」更好对账，也避免 chunk 在某些后端被拆成多余算子。
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, z_hat):
        params = self.net.y_prior_fusion(self.net.hyper_dec(z_hat))
        quant_step, scales, means = params.chunk(3, 1)
        quant_step = quant_step.clamp_min(0.5)
        q_enc = 1.0 / quant_step
        q_dec = quant_step
        common = self.net.y_spatial_prior_reduction(params)
        return q_enc, q_dec, scales, means, common


class StageExport(nn.Module):
    """第 N 段：y_hat_so_far + common -> scales, means。"""

    def __init__(self, net, stage):
        super().__init__()
        adaptors = (net.y_spatial_prior_adaptor_1, net.y_spatial_prior_adaptor_2,
                    net.y_spatial_prior_adaptor_3)
        self.adaptor = adaptors[stage - 1]
        self.prior = net.y_spatial_prior

    def forward(self, y_hat_so_far, common):
        scales, means = self.prior(self.adaptor(
            torch.cat((y_hat_so_far, common), dim=1))).chunk(2, 1)
        return scales, means


class TailExport(nn.Module):
    """y_hat + q_dec -> x_hat。clamp 与官方一致（image_model.py:287 附近）。"""

    def __init__(self, net):
        super().__init__()
        self.dec = net.dec
        self.generator = net.vqgan.generator

    def forward(self, y_hat, q_dec):
        return self.generator(self.dec(y_hat, q_dec)).clamp(-1, 1)


def export(module, args, path, input_names, output_names):
    torch.onnx.export(
        module, args, path, input_names=input_names, output_names=output_names,
        opset_version=13, do_constant_folding=True, dynamo=False,
    )
    size = os.path.getsize(path) / 1e6
    print(f"  {os.path.basename(path)}: {size:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PROJ, "models", "GLC_image.pth.tar"))
    ap.add_argument(
        "--out",
        default=os.path.normpath(os.path.join(HERE, "..", "models", "onnx")))
    ap.add_argument("--glc-src", default=os.path.join(PROJ, "GLC_src"))
    ap.add_argument("--stubs", default=os.path.join(HERE, "stubs"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from model_loader import load_glc

    net = load_glc(args.ckpt, args.glc_src, args.stubs)
    print(f"权重已加载，尺寸锁定 {H}x{W}：y=({N},{Y_H},{Y_W}) z=({N},{Z_H},{Z_W})")

    # 固定 shape 的示例输入，形状即 bmodel 的静态 shape
    z_hat = torch.randn(1, N, Z_H, Z_W)
    y_hat_so_far = torch.zeros(1, N, Y_H, Y_W)
    y_hat = torch.randn(1, N, Y_H, Y_W)
    q_dec = net.q_dec[0:1].detach().clone()

    print("导出子图：")
    export(PriorInitExport(net).eval(), (z_hat,),
           os.path.join(args.out, "glc_prior_init.onnx"),
           ["z_hat"], ["q_enc", "q_dec", "scales", "means", "common"])

    # 先拿真实 common 造 stage 输入，保证形状与真实推理一致
    with torch.no_grad():
        params = net.y_prior_fusion(net.hyper_dec(z_hat))
        common = net.y_spatial_prior_reduction(params)
    for stage in (1, 2, 3):
        export(StageExport(net, stage).eval(), (y_hat_so_far, common),
               os.path.join(args.out, f"glc_stage_{stage}.onnx"),
               ["y_hat_so_far", "common"], ["scales", "means"])

    export(TailExport(net).eval(), (y_hat, q_dec),
           os.path.join(args.out, "glc_tail.onnx"),
           ["y_hat", "q_dec"], ["x_hat"])

    print(f"完成，输出位于 {args.out}")


if __name__ == "__main__":
    main()
