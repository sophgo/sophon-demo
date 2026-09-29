#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""GLC 数据集批量评估（SE7 设备端）。

对指定精度遍历数据集 × 质量档位，输出与 GLC 官方 test_image.py 同口径的
bpp / PSNR / SSIM，并给出分阶段耗时。结果写入 JSON，供 tools/eval_accuracy.py
与 PyTorch 基线对比。

用法:
    python3 glc_eval.py --precision bf16 \
        --data ../datasets/kodak24 --out ../out/eval_bf16.json
"""
import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bmrt import BmrtEngine                                    # noqa: E402
from glc_infer import (CODEBOOK_SIZE, codebook_lookup, gaussian_bits,  # noqa: E402
                       load_image, pad_to_multiple, postprocess, preprocess,
                       run_once)
from metrics import evaluate                                   # noqa: E402

EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.normpath(os.path.join(here, "..", "models", "BM1684X")))
    ap.add_argument("--precision", default="bf16",
                    choices=["fp32", "fp16", "bf16", "int8"])
    ap.add_argument("--data", required=True, help="图像目录")
    ap.add_argument("--out", required=True, help="结果 JSON 路径")
    ap.add_argument("--q-indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--loops", type=int, default=5, help="每张图计时循环次数")
    ap.add_argument("--dev_id", type=int, default=0)
    args = ap.parse_args()
    # 分阶段耗时用 logging.info 输出，root logger 默认 WARNING 会把它丢掉
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    enc_p = os.path.join(args.models, "glc_encoder_%s.bmodel" % args.precision)
    dec_p = os.path.join(args.models, "glc_decoder_%s.bmodel" % args.precision)
    for p in (enc_p, dec_p):
        if not os.path.exists(p):
            raise SystemExit("模型不存在: %s" % p)

    codebook = np.load(os.path.join(args.models, "codebook.npy"))
    q_all = np.load(os.path.join(args.models, "q_enc.npy"))
    qd_all = np.load(os.path.join(args.models, "q_dec.npy"))

    files = sorted(f for f in os.listdir(args.data) if f.lower().endswith(EXTS))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit("目录下没有图像: %s" % args.data)

    enc = BmrtEngine(enc_p, args.dev_id)
    dec = BmrtEngine(dec_p, args.dev_id)

    # bmodel 是静态 shape，只评估 padding 后能对上模型输入尺寸的图
    want = tuple(enc.info.input_shapes[0])          # (1,3,H,W)
    kept, skipped = [], []
    for fn in files:
        img = load_image(os.path.join(args.data, fn))
        padded, _ = pad_to_multiple(img)
        h, w = padded.shape[:2]
        (kept if (1, 3, h, w) == want else skipped).append(fn)
    if skipped:
        print("跳过 %d 张尺寸不匹配的图（模型输入固定为 %dx%d）: %s"
              % (len(skipped), want[2], want[3], ", ".join(skipped)), flush=True)
    files = kept
    if not files:
        raise SystemExit("没有尺寸匹配的图像（模型输入 %dx%d）" % (want[2], want[3]))

    results = {}
    try:
        for q in args.q_indexes:
            q_enc = np.ascontiguousarray(q_all[q:q + 1])
            q_dec = np.ascontiguousarray(qd_all[q:q + 1])
            per_img, bpps, psnrs, ssims = {}, [], [], []
            tsum = {"preprocess": 0.0, "encoder": 0.0, "vq_lookup": 0.0,
                    "decoder": 0.0, "postprocess": 0.0, "total": 0.0}
            for fn in files:
                tp0 = time.perf_counter()
                img = load_image(os.path.join(args.data, fn))
                img_pad, (pl, pr, pt, pb) = pad_to_multiple(img)
                x = preprocess(img_pad)
                tsum["preprocess"] += time.perf_counter() - tp0
                h, w = img.shape[:2]

                # 预热一次，再计时
                run_once(enc, dec, codebook, q_enc, q_dec, x)
                t0 = time.perf_counter()
                for _ in range(args.loops):
                    x_hat, y_q, scales_hat, z_shape, t = run_once(
                        enc, dec, codebook, q_enc, q_dec, x)
                total = (time.perf_counter() - t0) / args.loops
                for k in ("encoder", "vq_lookup", "decoder"):
                    # t[k] 已是单次迭代耗时，不能再除 loops
                    tsum[k] += t[k]
                tsum["total"] += total

                tq0 = time.perf_counter()
                bit_y = gaussian_bits(y_q, scales_hat)
                bit_z = z_shape[2] * z_shape[3] * math.log2(CODEBOOK_SIZE)
                bpp = (bit_y + bit_z) / (h * w)

                rec = postprocess(x_hat)[pt:pt + h, pl:pl + w]
                tsum["postprocess"] += time.perf_counter() - tq0
                # 指标计算不计入耗时（纯 numpy 的 SSIM 很慢，会淹没真实后处理开销）
                p, s = evaluate(rec, img)
                per_img[fn] = {"bpp": bpp, "bit_y": bit_y, "bit_z": bit_z,
                               "psnr": p, "ssim": s, "latency_ms": total * 1e3}
                bpps.append(bpp)
                psnrs.append(p)
                ssims.append(s)
                print("  q%d %-14s bpp=%.4f psnr=%.2f ssim=%.4f %.0fms"
                      % (q, fn, bpp, p, s, total * 1e3), flush=True)

            n = len(files)
            per_img["_avg"] = {
                "bpp": float(np.mean(bpps)), "psnr": float(np.mean(psnrs)),
                "ssim": float(np.mean(ssims)),
                "preprocess_ms": tsum["preprocess"] / n * 1e3,
                "encoder_ms": tsum["encoder"] / n * 1e3,
                "vq_ms": tsum["vq_lookup"] / n * 1e3,
                "decoder_ms": tsum["decoder"] / n * 1e3,
                "postprocess_ms": tsum["postprocess"] / n * 1e3,
                "latency_ms": tsum["total"] / n * 1e3,
                "fps": n / tsum["total"],
            }
            results["q%d" % q] = per_img
            a = per_img["_avg"]
            print("q%d 平均: bpp=%.4f psnr=%.2f ssim=%.4f  %.0fms  %.2f FPS"
                  % (q, a["bpp"], a["psnr"], a["ssim"], a["latency_ms"], a["fps"]),
                  flush=True)
            # 汇总行，便于人工核对与脚本解析
            print("q%d average_bpp:  %.4f" % (q, a["bpp"]), flush=True)
            print("q%d average_psnr:  %.2f" % (q, a["psnr"]), flush=True)
            print("q%d average_ssim:  %.4f" % (q, a["ssim"]), flush=True)
            if q == args.q_indexes[0]:
                # 分阶段耗时汇总，对应第一个质量档位
                logging.info("preprocess_time(ms): {:.2f}".format(a["preprocess_ms"]))
                logging.info("encoder_time(ms): {:.2f}".format(a["encoder_ms"]))
                logging.info("vq_time(ms): {:.2f}".format(a["vq_ms"]))
                logging.info("decoder_time(ms): {:.2f}".format(a["decoder_ms"]))
                logging.info("inference_time(ms): {:.2f}".format(
                    a["encoder_ms"] + a["vq_ms"] + a["decoder_ms"]))
                logging.info("postprocess_time(ms): {:.2f}".format(a["postprocess_ms"]))
    finally:
        enc.close()
        dec.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": {"precision": args.precision, "num_images": len(files),
                            "images": files, "loops": args.loops},
                   "results": results}, f, indent=2)
    print("结果已写入", args.out)


if __name__ == "__main__":
    main()
