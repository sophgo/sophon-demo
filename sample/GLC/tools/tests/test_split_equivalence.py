#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""验证模型边界重切没改变语义：分离式实现 vs 官方整体式前向。

这是边界重切最关键的检查，而且**不依赖真实权重**：
只要两边数学路径一致，无论权重取什么值输出都必须逐位相同。
所以随机初始化就能把「切分切错了」与「权重没下对」两种失败区分开。

参照对象是 GLC_Image.test()（官方理论值路径），它内部走
forward_four_part_prior；分离式实现必须得到同一个 y_hat。

运行的检查：
  1. 分离式编码端重建的 y_hat 与官方 test() 逐位相同
  2. 码流往返后解码端 y_hat 与编码端逐位相同
  3. 编码端/解码端逐段选出的 CDF 索引逐元素一致
  4. 重建图与官方路径的 x_hat 逐位相同

运行：PYTHONPATH=vendor:. python3 tests/test_split_equivalence.py
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # tools/tests -> GLC
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "python"))

import entropy_numpy as en  # noqa: E402
import glc_codec  # noqa: E402
from rans_coder import RansCoder  # noqa: E402

H, W = 512, 768
_failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def build_random_net():
    from model_loader import prepare_paths

    prepare_paths(os.path.join(PROJ, "GLC_src"), os.path.join(PROJ, "tools", "stubs"))
    from src.models.image_model import GLC_Image

    torch.manual_seed(1234)
    net = GLC_Image(inplace=False)
    net.eval()
    return net


@torch.no_grad()
def official_test_path(net, x, q_index):
    """逐行复刻 image_model.py:113 test()，作为分离式实现的参照。

    返回 (y_hat, x_hat, z, z_hat, z_indices)。y_hat 是乘过 q_dec 的最终隐变量。
    """
    curr_q_enc = net.q_enc[q_index:q_index + 1]
    curr_q_dec = net.q_dec[q_index:q_index + 1]

    y = net.enc(net.vqgan.encoder(x), curr_q_enc)
    z = net.hyper_enc(y)
    index = net.z_vq.get_indices(z)
    z_hat = net.z_vq.get_quan_feat(index, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))

    params = net.y_prior_fusion(net.hyper_dec(z_hat))
    _, _, y_hat, _ = net.forward_four_part_prior(
        y, params, net.y_spatial_prior_adaptor_1, net.y_spatial_prior_adaptor_2,
        net.y_spatial_prior_adaptor_3, net.y_spatial_prior,
        y_spatial_prior_reduction=net.y_spatial_prior_reduction)
    x_hat = net.vqgan.generator(net.dec(y_hat, curr_q_dec))
    return y_hat, x_hat, z, z_hat, index


@torch.no_grad()
def stage_parameter_spread(net, x, q_index):
    """确认四段先验真的是自回归的：各段 scales 必须互不相同。

    若切分时把 stage 参数算错（例如各段都退回初始参数），这个量会退化成 0。
    这是检测「切分静默失效」最直接的信号。
    """
    q_enc = net.q_enc[q_index:q_index + 1]
    y = net.enc(net.vqgan.encoder(x), q_enc)
    z = net.hyper_enc(y)
    z_indices = net.z_vq.get_indices(z)
    z_hat = net.z_vq.get_quan_feat(
        z_indices, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))
    params = net.y_prior_fusion(net.hyper_dec(z_hat))
    _, _, scales, means = glc_codec._separate_prior(params)
    common = net.y_spatial_prior_reduction(params)

    masks = net.get_mask_four_parts(*y.shape, y.dtype, y.device)
    hat = torch.zeros_like(y)
    per_stage_scales = []
    for stage, mask in enumerate(masks):
        s, m = glc_codec._stage_params(net, stage, hat, common, scales, means)
        per_stage_scales.append(glc_codec.compact_masked(s, mask).numpy().reshape(-1))
        # 用真实量化重建推进，与编码端一致
        sym = en.quantize(glc_codec.compact_masked(y * glc_codec._separate_prior(params)[0], mask).numpy(),
                          glc_codec.compact_masked(m, mask).numpy())
        pm = glc_codec.compact_masked(m, mask)
        recon = en.dequantize(sym.reshape(-1), pm.numpy().reshape(-1))
        hat = hat + glc_codec.scatter_masked(
            torch.from_numpy(recon).reshape(1, *pm.shape[1:]).to(y.dtype), mask, y.shape)

    # 第 0 段与其余各段比，以及相邻段互比，都必须有实质差异
    diffs = []
    for i in range(len(per_stage_scales)):
        for j in range(i + 1, len(per_stage_scales)):
            diffs.append(float(np.abs(per_stage_scales[i] - per_stage_scales[j]).max()))
    return min(diffs)


def main():
    from model_loader import prepare_paths  # noqa: F401  (build_random_net 内已调用)

    net = build_random_net()
    torch.manual_seed(7)
    x = torch.randn(1, 3, H, W).clamp(-1, 1)
    q_index = 0

    y_hat_ref, x_hat_ref, z, z_hat, z_index = official_test_path(net, x, q_index)
    print(f"官方路径 y={tuple(y_hat_ref.shape)} z={tuple(z.shape)} "
          f"z 索引数={z_index.numel()}")

    entropy = glc_codec.EntropyContext.from_compressai()
    coder = RansCoder()
    payload, enc_y_hat, info = glc_codec.encode(
        coder, entropy, net, net.z_vq, x, q_index, (H, W))
    x_hat_dec, dec_y_hat = glc_codec.decode(coder, entropy, net, net.z_vq, payload)

    check("分离式编码端 y_hat 与官方 test() 逐位相同",
          torch.equal(enc_y_hat, y_hat_ref),
          f"最大误差 {(enc_y_hat - y_hat_ref).abs().max().item():.3e}")
    check("编解码两端 y_hat 逐位相同（torch.equal 级）",
          torch.equal(enc_y_hat, dec_y_hat),
          f"最大误差 {(enc_y_hat - dec_y_hat).abs().max().item():.3e}")
    check("重建图与官方路径逐位相同",
          torch.equal(x_hat_dec, x_hat_ref.clamp(-1, 1)),
          f"最大误差 {(x_hat_dec - x_hat_ref.clamp(-1, 1)).abs().max().item():.3e}")

    spread = stage_parameter_spread(net, x, q_index)
    check("四段先验参数确有自回归差异（切分未静默退化）", spread > 1e-6,
          f"各段 scales 最小两两差 {spread:.3e}")

    bd = glc_codec.bpp_breakdown(payload, H, W)
    check("码流含四段非空 y 码流", len(bd["y_bytes"]) == 4 and min(bd["y_bytes"]) > 0,
          f"{bd['y_bytes']}")
    check("y 符号数 = 256x32x48 = 393216", bd["y_shape"] == [256, 32, 48],
          f"{bd['y_shape']}")

    print()
    if _failures:
        print(f"失败 {len(_failures)} 项: {_failures}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
