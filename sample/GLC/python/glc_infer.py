#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""GLC 图像压缩 SE7/SE9 (BM1684X/BM1688) 推理例程。

流程与 GLC 官方 test_image.py 一致，只是把重计算部分换成 TPU bmodel：

    image -> [pre] -> encoder.bmodel -> y, z
          -> [CPU] 码本最近邻 (argmin) -> z_hat
          -> decoder.bmodel -> x_hat, y_q, scales_hat
          -> [post] 重建图 + bpp/PSNR/SSIM

编码器与解码器之间必须插一段 CPU 码本查找：z_vq 的 argmin 是数据依赖的离散选择，
TPU 上展开不划算（且 GLC 官方也在 CPU/GPU 上做这一步）。

用法:
    python3 glc_infer.py --precision bf16 \
        --input ../datasets/kodak24/kodim01.png --q_index 0
"""
import argparse
import logging
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bmrt import BmrtEngine          # noqa: E402
from metrics import evaluate         # noqa: E402

PADDING_SIZE = 64
CODEBOOK_SIZE = 16384
EMB_DIM = 256


# ---------------- 预处理 / 后处理 ----------------

def load_image(path):
    """读 RGB 图，返回 uint8 HWC。"""
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("需要 Pillow 读图: pip3 install pillow")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def pad_to_multiple(img, p=PADDING_SIZE):
    """replicate 填充到 p 的整数倍，返回 (padded, (l, r, t, b))。"""
    h, w = img.shape[:2]
    pl, pr, pt, pb = GLC_Image_get_padding_size(h, w, p)
    if pl or pr or pt or pb:
        img = np.pad(img, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
    return img, (pl, pr, pt, pb)


def GLC_Image_get_padding_size(height, width, p=64):
    """与 GLC_Image.get_padding_size 等价：向最近的下界取整，居中填充。"""
    if height % p:
        pad_h = p - height % p
        pt = pad_h // 2
        pb = pad_h - pt
    else:
        pt = pb = 0
    if width % p:
        pad_w = p - width % p
        pl = pad_w // 2
        pr = pad_w - pl
    else:
        pl = pr = 0
    return pl, pr, pt, pb


def preprocess(img_uint8):
    """uint8 HWC -> float32 1CHW，值域 [-1,1]（同 from_0_1_to_minus1_1）。"""
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - 0.5) * 2.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def postprocess(x_hat):
    """float32 1CHW [-1,1] -> uint8 HWC（同 write_image）。"""
    v = x_hat[0].transpose(1, 2, 0)
    v = np.clip(np.rint(((v / 2.0) + 0.5) * 255.0), 0, 255)
    return v.astype(np.uint8)


# ---------------- 熵模型比特估算 ----------------

def _norm_cdf(z):
    """标准正态 CDF。用 erfc 形式，避免大 |z| 时 1-erf 的相消误差。"""
    x = np.abs(z) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
                 t * (-1.453152027 + t * 1.061405429))))
    erfc = poly * np.exp(-x * x)
    return np.where(z >= 0, 1.0 - 0.5 * erfc, 0.5 * erfc)


def gaussian_bits(y_q, scales_hat):
    """等价于 GLC_Image.get_y_gaussian_bits：-log2(P(y-0.5 < Y < y+0.5))。"""
    sigma = np.clip(scales_hat, 1e-5, 1e10)
    probs = _norm_cdf((y_q + 0.5) / sigma) - _norm_cdf((y_q - 0.5) / sigma)
    bits = np.clip(-np.log2(probs + 1e-5), 0.0, None)
    return float(bits.sum())


def codebook_lookup(z, codebook):
    """z_vq 最近邻量化：返回 z_hat (1,C,H,W)。"""
    b, c, h, w = z.shape
    zz = z.transpose(0, 2, 3, 1).reshape(-1, c)
    # (z-e)^2 = z^2 + e^2 - 2 z·e
    d = (zz ** 2).sum(1, keepdims=True) + (codebook ** 2).sum(1) - 2.0 * zz @ codebook.T
    idx = np.argmin(d, axis=1)
    z_hat = codebook[idx].reshape(b, h, w, c).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(z_hat.astype(np.float32)), idx


# ---------------- 主流程 ----------------

def run_once(enc, dec, codebook, q_enc, q_dec, x):
    """一次完整前向，返回 (x_hat, y_q, scales_hat, z_shape, 分阶段耗时 dict)。"""
    t = {}
    t0 = time.perf_counter()
    eo = enc.infer({"x": x, "q_enc": q_enc})
    t["encoder"] = time.perf_counter() - t0

    # bmodel 输出名由 TPU-MLIR 改写（y_Add / z_Conv），按导出顺序取更稳
    ev = list(eo.values())
    y, z = ev[0], ev[1]

    t0 = time.perf_counter()
    z_hat, _ = codebook_lookup(z, codebook)
    t["vq_lookup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    do = dec.infer({"y": y, "z_hat": z_hat, "q_dec": q_dec})
    t["decoder"] = time.perf_counter() - t0

    dv = list(do.values())
    x_hat, y_q, scales_hat = dv[0], dv[1], dv[2]
    return x_hat, y_q, scales_hat, z.shape, t


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_models = os.path.normpath(os.path.join(here, "..", "models", "BM1684X"))

    ap = argparse.ArgumentParser(description="GLC on Sophon SE7/SE9 (BM1684X/BM1688)")
    ap.add_argument("--models", default=default_models, help="模型/资产目录")
    ap.add_argument("--precision", default="fp32",
                    choices=["fp32", "fp16", "bf16", "int8"])
    ap.add_argument("--input", required=True, help="输入图像路径")
    ap.add_argument("--q_index", type=int, default=0, choices=[0, 1, 2, 3],
                    help="质量档位 0..3")
    ap.add_argument("--dev_id", type=int, default=0)
    ap.add_argument("--loops", type=int, default=1, help="重复推理次数（性能测试）")
    ap.add_argument("--out", default=None, help="重建图输出路径（.png）")
    args = ap.parse_args()
    # 分阶段耗时用 logging.info 输出，root logger 默认 WARNING 会把它丢掉
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    enc_path = os.path.join(args.models, "glc_encoder_%s.bmodel" % args.precision)
    dec_path = os.path.join(args.models, "glc_decoder_%s.bmodel" % args.precision)
    for p in (enc_path, dec_path):
        if not os.path.exists(p):
            raise SystemExit("模型不存在: %s" % p)

    codebook = np.load(os.path.join(args.models, "codebook.npy"))
    q_all = np.load(os.path.join(args.models, "q_enc.npy"))
    qd_all = np.load(os.path.join(args.models, "q_dec.npy"))
    q_enc = np.ascontiguousarray(q_all[args.q_index:args.q_index + 1])
    q_dec = np.ascontiguousarray(qd_all[args.q_index:args.q_index + 1])

    t0 = time.perf_counter()
    img = load_image(args.input)
    img_pad, pads = pad_to_multiple(img)
    x = preprocess(img_pad)
    pl, pr, pt, pb = pads
    preprocess_time = time.perf_counter() - t0

    print("精度: %s | 质量档位: q%d | 输入: %s %dx%d (pad 后 %dx%d)"
          % (args.precision, args.q_index, os.path.basename(args.input),
             img.shape[1], img.shape[0], img_pad.shape[1], img_pad.shape[0]))

    enc = BmrtEngine(enc_path, args.dev_id)
    dec = BmrtEngine(dec_path, args.dev_id)
    try:
        # 预热（首帧含 bmodel 权重搬运）
        run_once(enc, dec, codebook, q_enc, q_dec, x)

        acc = {"encoder": 0.0, "vq_lookup": 0.0, "decoder": 0.0}
        t_all = time.perf_counter()
        for _ in range(args.loops):
            x_hat, y_q, scales_hat, z_shape, t = run_once(
                enc, dec, codebook, q_enc, q_dec, x)
            for k in acc:
                acc[k] += t[k]
        total = time.perf_counter() - t_all
    finally:
        enc.close()
        dec.close()

    for k in acc:
        acc[k] /= args.loops
    avg_total = total / args.loops

    t0 = time.perf_counter()
    # bpp：bit_y 来自熵模型，bit_z 是码本索引的定长开销
    h, w = img.shape[:2]
    bit_y = gaussian_bits(y_q, scales_hat)
    bit_z = z_shape[2] * z_shape[3] * math.log2(CODEBOOK_SIZE)
    bpp = (bit_y + bit_z) / (h * w)

    rec = postprocess(x_hat)
    rec = rec[pt:pt + h, pl:pl + w]
    postprocess_time = time.perf_counter() - t0
    # 指标计算不计入耗时（纯 numpy 的 SSIM 很慢，会淹没真实后处理开销）
    psnr, ssim = evaluate(rec, img)

    print("  bpp          : %.4f  (bit_y=%.0f, bit_z=%.0f, %.0fx%.0f 码本索引)"
          % (bpp, bit_y, bit_z, z_shape[2], z_shape[3]))
    print("  PSNR / SSIM  : %.2f dB / %.4f" % (psnr, ssim))
    print("  耗时 (ms)    : encoder %.1f | vq %.1f | decoder %.1f | 合计 %.1f"
          % (acc["encoder"] * 1e3, acc["vq_lookup"] * 1e3,
             acc["decoder"] * 1e3, avg_total * 1e3))
    print("  吞吐         : %.2f FPS" % (1.0 / avg_total))

    # 分阶段耗时，供性能统计使用，格式与其它 sample 保持一致
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("encoder_time(ms): {:.2f}".format(acc["encoder"] * 1000))
    logging.info("vq_time(ms): {:.2f}".format(acc["vq_lookup"] * 1000))
    logging.info("decoder_time(ms): {:.2f}".format(acc["decoder"] * 1000))
    logging.info("inference_time(ms): {:.2f}".format(avg_total * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))

    if args.out:
        try:
            from PIL import Image
            Image.fromarray(rec).save(args.out)
            print("  重建图已保存 : %s" % args.out)
        except ImportError:
            print("  (未安装 Pillow，跳过重建图保存)")


if __name__ == "__main__":
    main()
