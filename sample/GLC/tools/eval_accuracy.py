#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""精度对比：设备端 bmodel 结果 vs PyTorch 基线。

用法:
    python3 tools/eval_accuracy.py --baseline baseline_kodak24.json \
        --device ../out/eval_fp32.json ../out/eval_bf16.json [../out/eval_int8.json]
"""
import argparse
import json
import os


def load_device(path):
    with open(path) as f:
        d = json.load(f)
    return d["meta"]["precision"], d["results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--device", nargs="+", required=True)
    args = ap.parse_args()

    with open(args.baseline) as f:
        base = json.load(f)["results"]

    devices = [load_device(p) for p in args.device]

    # 取所有设备结果都覆盖的图片交集（各精度评估的图集应一致）
    for q in sorted(base.keys(), key=lambda x: int(x[1:])):
        imgs = set(base[q].keys()) - {"_avg"}
        for _, res in devices:
            imgs &= set(res.get(q, {}).keys()) - {"_avg"}
        if not imgs:
            continue
        imgs = sorted(imgs)
        n = len(imgs)
        print("\n=== 质量档位 %s（%d 张图）===" % (q, n))
        print("%-8s %-10s %-12s %-12s %-10s %-9s" %
              ("精度", "bpp", "PSNR(dB)", "SSIM", "PSNR差", "状态"))
        print("-" * 66)

        # 基线 _avg 覆盖的是完整数据集，必须在公共图集上重算才能对齐
        def avg_of(res, key):
            return sum(res[q][i][key] for i in imgs) / len(imgs)

        b_bpp, b_psnr, b_ssim = avg_of(base, "bpp"), avg_of(base, "psnr"), avg_of(base, "ssim")
        print("%-8s %-10.4f %-12.2f %-12.4f %-10s %-9s" %
              ("PyTorch", b_bpp, b_psnr, b_ssim, "-", "基线"))

        for prec, res in devices:
            a_bpp, a_psnr, a_ssim = avg_of(res, "bpp"), avg_of(res, "psnr"), avg_of(res, "ssim")
            d_psnr = a_psnr - b_psnr
            # 判据：FP32 应 <0.1dB；BF16/FP16 <0.5dB；INT8 <1.5dB
            if prec == "fp32":
                ok = abs(d_psnr) < 0.1
            elif prec in ("bf16", "fp16"):
                ok = abs(d_psnr) < 0.5
            else:
                ok = abs(d_psnr) < 1.5
            print("%-8s %-10.4f %-12.2f %-12.4f %+-10.2f %-9s" %
                  (prec, a_bpp, a_psnr, a_ssim, d_psnr,
                   "OK" if ok else "偏差偏大"))

        # 逐图最大 PSNR 偏差
        for prec, res in devices:
            worst = max(((abs(res[q][i]["psnr"] - base[q][i]["psnr"]), i)
                         for i in imgs), key=lambda t: t[0])
            print("  %s 单图最大 PSNR 偏差: %.2f dB (%s)"
                  % (prec, worst[0], worst[1]))

    print("\n=== 性能 ===")
    print("%-8s %-12s %-10s %-10s %-10s" %
          ("精度", "端到端(ms)", "编码(ms)", "解码(ms)", "吞吐(FPS)"))
    print("-" * 54)
    for prec, res in devices:
        a = res["q0"]["_avg"]
        print("%-8s %-12.0f %-10.0f %-10.0f %-10.2f" %
              (prec, a["latency_ms"], a["encoder_ms"], a["decoder_ms"], a["fps"]))


if __name__ == "__main__":
    main()
