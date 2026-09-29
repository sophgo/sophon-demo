#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""真熵编码往返一致 + 与理论值路径对比。

检验项：
  1. 编解码分离：编码端只产出 bytes，解码端只消费 bytes
  2. 往返一致性：解码端重建的 y_hat 与编码端逐位相同（torch.equal 级）
  3. 实测真实 bpp：与现有理论 bpp 并排对比

需要真实权重 GLC_image.pth.tar。缺权重时直接报缺失并返回非 0，
不用随机权重糊弄——随机权重的 bpp 与 PSNR 都没有参考价值。

运行：PYTHONPATH=tools/vendor:python:. python3 tools/tests/test_roundtrip.py
"""

import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # tools/tests -> GLC
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "python"))

import glc_codec  # noqa: E402
from rans_coder import RansCoder  # noqa: E402

CKPT = os.path.join(PROJ, "models", "GLC_image.pth.tar")
GLC_SRC_PARENT = os.path.join(PROJ, "GLC_src")
STUBS = os.path.join(PROJ, "tools", "stubs")

H, W = 512, 768
_failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def load_input(image_path):
    """有真实图就用真实图，否则退回随机噪声。

    真实图才让 bpp 有意义：随机噪声的分布接近均匀，码流长度不能代表实际场景。
    测试输出会明示用的是哪一种，避免把噪声结果当成性能结论。
    """
    if image_path and os.path.isfile(image_path):
        from PIL import Image

        import numpy as np

        im = Image.open(image_path).convert("RGB")
        if im.size != (W, H):
            raise ValueError(f"输入图尺寸 {im.size} 与目标 {(W, H)} 不符，本例程是静态 shape")
        arr = np.asarray(im, dtype=np.float32) / 255.0 * 2.0 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1)[None]
        return x, os.path.basename(image_path)

    torch.manual_seed(0)
    return torch.randn(1, 3, H, W).clamp(-1, 1), "random-noise"


def main(image_path=None):
    if not os.path.isfile(CKPT):
        print(f"缺少权重 {CKPT}，无法跑端到端验收")
        return 2

    from model_loader import load_glc

    net = load_glc(CKPT, GLC_SRC_PARENT, STUBS)
    x, source = load_input(image_path)
    print(f"输入: {source}  {tuple(x.shape)}")

    entropy = glc_codec.EntropyContext.from_compressai()
    coder = RansCoder()

    payload, enc_y_hat, info = glc_codec.encode(
        coder, entropy, net, net.z_vq, x, 0, (H, W)
    )
    bd = glc_codec.bpp_breakdown(payload, H, W)
    theory_bpp = (
        info["theory_y_bits"]
        + info["z_elements"] * math.log2(info["codebook_size"])
    ) / (H * W)

    print(f"码流总长 {len(payload)} 字节 = header {bd['header_bytes']} "
          f"+ z {bd['z_bytes']} + 四段 y {bd['y_bytes']}")
    print(f"真实 bpp {bd['total']:.6f} = y {bd['y']:.6f} + z {bd['z']:.6f} "
          f"+ header {bd['header']:.6f}")
    print(f"理论 bpp（现有例程口径）{theory_bpp:.6f}，"
          f"差值 {bd['total'] - theory_bpp:+.6f} "
          f"({(bd['total'] - theory_bpp) / theory_bpp * 100:+.2f}%)")

    check("码流头固定 76 字节", bd["header_bytes"] == 76)
    check("含四段 y 码流且每段非空", len(bd["y_bytes"]) == 4 and min(bd["y_bytes"]) > 0,
          f"各段字节 {bd['y_bytes']}")
    check("z 码流非空", bd["z_bytes"] > 0, f"{bd['z_bytes']} 字节")
    check("y 符号数等于 256x32x48", bd["y_shape"] == [256, 32, 48],
          f"y 形状 {bd['y_shape']}")

    x_hat, dec_y_hat = glc_codec.decode(coder, entropy, net, net.z_vq, payload)

    # 验收标准 3：算术编码成立的前提是两端逐位相同
    same = torch.equal(enc_y_hat, dec_y_hat)
    max_err = (enc_y_hat - dec_y_hat).abs().max().item()
    check("编解码两端 y_hat 逐位相同", same, f"torch.equal={same}, 最大误差={max_err}")
    check("重建图形状与输入一致", tuple(x_hat.shape) == (1, 3, H, W),
          f"{tuple(x_hat.shape)}")

    # 验收标准 4：真实 bpp 不低于理论下界（低于是不可能的，说明口径错了）
    check("真实 bpp 不低于理论下界", bd["total"] >= theory_bpp - 1e-9,
          f"真实 {bd['total']:.6f} vs 理论 {theory_bpp:.6f}")
    check("真实 bpp 超出理论值在合理范围（< 15%）",
          (bd["total"] - theory_bpp) / theory_bpp < 0.15,
          f"{(bd['total'] - theory_bpp) / theory_bpp * 100:+.2f}%")

    print()
    if _failures:
        print(f"失败 {len(_failures)} 项: {_failures}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="GLC 真熵编码端到端验收")
    ap.add_argument("--image", default=None,
                    help="512x768 真实图；不给则用随机噪声（bpp 无参考价值）")
    sys.exit(main(ap.parse_args().image))
