#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""实测真熵编码的开销与码率。

rANS 是逐符号串行编解码，393216 个符号的耗时是本方案最大的性能不确定项。
本脚本给三组数字：

  1. 每张图的真实 bpp 与理论 bpp
  2. rANS 编/解码的逐段耗时，及其在总耗时里的占比
  3. 与现有 BF16 端到端基线（约 688ms）的对比

量的是主机 x86_64 上的耗时。SE7 的 aarch64 主频更低，绝对耗时会更大，
占比结构可以参照；SE7/SE9 真机数字见 docs/GLC_Entropy_Coding.md 第七、八节。

用法：
    PYTHONPATH=tools/vendor:. python3 tools/bench_entropy.py --data datasets --limit 6
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "python"))

import entropy_numpy as en  # noqa: E402
import glc_codec  # noqa: E402
from rans_coder import RansCoder  # noqa: E402

CKPT = os.path.join(PROJ, "models", "GLC_image.pth.tar")
H, W = 512, 768


def load_image(path):
    """读图并校验尺寸。

    本例程是静态 shape 512x768（高x宽），竖构图（如 Kodak 的 04/06）不适用，
    与现有 glc_eval.py 的处理方式一致：跳过而不是拉伸。
    """
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if im.size != (W, H):
        raise ValueError(f"{os.path.basename(path)} 尺寸 {im.size} 不是 {W}x{H}（宽x高）")
    arr = np.asarray(im, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJ, "datasets"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--q_index", type=int, default=0)
    args = ap.parse_args()

    from model_loader import load_glc

    net = load_glc(CKPT, os.path.join(PROJ, "GLC_src"), os.path.join(PROJ, "tools", "stubs"))
    entropy = glc_codec.EntropyContext.from_compressai()
    coder = RansCoder()

    images = sorted(glob.glob(os.path.join(args.data, "*.png")))
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"{args.data} 下没有 png", file=sys.stderr)
        return 2

    rows = []
    skipped = []
    for path in images:
        try:
            x = load_image(path)
        except ValueError as exc:
            skipped.append(os.path.basename(path))
            continue

        t0 = time.perf_counter()
        payload, enc_y_hat, info = glc_codec.encode(
            coder, entropy, net, net.z_vq, x, args.q_index, (H, W))
        enc_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        _, dec_y_hat = glc_codec.decode(coder, entropy, net, net.z_vq, payload)
        dec_ms = (time.perf_counter() - t0) * 1000

        bd = glc_codec.bpp_breakdown(payload, H, W)
        theory = (info["theory_y_bits"]
                  + info["z_elements"] * np.log2(info["codebook_size"])) / (H * W)
        rows.append({
            "name": os.path.basename(path),
            "bytes": len(payload),
            "real": bd["total"], "theory": theory,
            "delta_pct": (bd["total"] - theory) / theory * 100,
            "y": bd["y"], "z": bd["z"], "header": bd["header"],
            "enc_ms": enc_ms, "dec_ms": dec_ms,
            "roundtrip_ok": bool(torch.equal(enc_y_hat, dec_y_hat)),
        })

    for s in skipped:
        print(f"跳过 {s}")

    print(f"{'图':<14}{'字节':>7}{'真实bpp':>10}{'理论bpp':>10}"
          f"{'差值%':>8}{'编码ms':>9}{'解码ms':>9}  往返")
    for r in rows:
        print(f"{r['name']:<14}{r['bytes']:>7}{r['real']:>10.5f}{r['theory']:>10.5f}"
              f"{r['delta_pct']:>+8.2f}{r['enc_ms']:>9.1f}{r['dec_ms']:>9.1f}"
              f"  {'OK' if r['roundtrip_ok'] else 'FAIL'}")

    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n
    print()
    print(f"平均：真实 bpp {avg('real'):.5f} / 理论 {avg('theory'):.5f} "
          f"(差值 {avg('delta_pct'):+.2f}%)")
    print(f"      y {avg('y'):.5f} + z {avg('z'):.5f} + header {avg('header'):.5f}")
    print(f"平均耗时：编码端 {avg('enc_ms'):.1f} ms，解码端 {avg('dec_ms'):.1f} ms")
    print(f"往返一致：{sum(r['roundtrip_ok'] for r in rows)}/{n}")

    # rANS 自身的串行耗时：393216 个符号，是本次主要新增开销
    print()
    if skipped:
        print(f"跳过 {len(skipped)} 张尺寸不符的图:")
        for name in skipped:
            print(f"  {name}")

    if rows:
        bench_rans(net, entropy, coder,
                   os.path.join(args.data, rows[0]["name"]))
    return 0


def bench_rans(net, entropy, coder, image_path):
    """单独量 rANS 编解码本身的耗时，与上面含网络的端到端耗时区分开。"""
    x = load_image(image_path)
    with torch.no_grad():
        y = net.enc(net.vqgan.encoder(x), net.q_enc[0:1])
        z = net.hyper_enc(y)
        z_hat = net.z_vq.get_quan_feat(
            net.z_vq.get_indices(z), (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))
        params = net.y_prior_fusion(net.hyper_dec(z_hat))
        q_enc, q_dec, scales, means = glc_codec._separate_prior(params)
        common = net.y_spatial_prior_reduction(params)
        y_scaled = y * q_enc
        masks = net.get_mask_four_parts(*y.shape, y.dtype, y.device)

        streams, idx_list, sym_list = [], [], []
        hat = torch.zeros_like(y)
        for stage, mask in enumerate(masks):
            ss, sm = glc_codec._stage_params(net, stage, hat, common, scales, means)
            payload, symbols, packed_means, packed_scales = glc_codec._encode_stage(
                coder, entropy, y_scaled, ss, sm, mask)
            streams.append(payload)
            idx_list.append(glc_codec._indexes_for(packed_scales.numpy(), entropy).reshape(-1))
            sym_list.append(symbols.reshape(-1))
            recon = en.dequantize(
                symbols.reshape(-1), packed_means.numpy().reshape(-1))
            hat = hat + glc_codec.scatter_masked(
                torch.from_numpy(recon).reshape(1, *packed_means.shape[1:]).to(y.dtype),
                mask, y.shape)

    total_syms = sum(s.size for s in sym_list)
    t0 = time.perf_counter()
    for _ in range(3):
        for s, i in zip(sym_list, idx_list):
            coder.encode(s, i, entropy.flat, entropy.lengths, entropy.starts,
                         entropy.offsets)
    enc_ms = (time.perf_counter() - t0) / 3 * 1000

    t0 = time.perf_counter()
    for _ in range(3):
        for s, i, p in zip(sym_list, idx_list, streams):
            coder.decode(p, s.size, i, entropy.flat, entropy.lengths,
                         entropy.starts, entropy.offsets)
    dec_ms = (time.perf_counter() - t0) / 3 * 1000

    print(f"rANS 串行耗时（{total_syms} 符号，四段合计，3 次平均）：")
    print(f"  编码 {enc_ms:.1f} ms  ({total_syms / enc_ms / 1000:.2f} M符号/s)")
    print(f"  解码 {dec_ms:.1f} ms  ({total_syms / dec_ms / 1000:.2f} M符号/s)")


if __name__ == "__main__":
    sys.exit(main())
