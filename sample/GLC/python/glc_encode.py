#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""GLC 压缩端：图 -> 真熵编码码流（.glc）。

这是部署中的「压缩端」独立进程：只产出码流文件，不持有任何要交给
还原端的张量。真实部署时这个 .glc 就是要过无线信道传过去的东西。

与 glc_infer.py（理论值路径）的区别：那个脚本把 y 直接喂给解码 bmodel 并
按高斯熵模型就地估算码率；本脚本把 y 真正 rANS 编码成比特流，还原端手里
只有这个文件。

依赖（设备端）：numpy + pillow，不需要 torch / compressai / scipy。

用法:
    python3 glc_encode.py --precision bf16 \
        --input ../datasets/kodak24/kodim01.png --out out.glc --q_index 0

产物码流布局（与客户参考实现 test_image_complement2.py 的 HEADER 逐字段对齐）:
    header(76B) + z_payload(14-bit 定长) + 四段 y rANS 码流
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entropy_numpy as en          # noqa: E402
import glc_codec as gc              # noqa: E402
from bmrt import BmrtEngine         # noqa: E402
from glc_infer import (             # noqa: E402
    CODEBOOK_SIZE, EMB_DIM, load_image, pad_to_multiple, preprocess,
)
from rans_coder import RansCoder    # noqa: E402

# 静态 shape（与 bmodel 一致）；码流头会把实际尺寸写进去，解码端据此还原
Y_CH, Y_H, Y_W = 256, 32, 48
Z_CH, Z_H, Z_W = 256, 8, 12


def default_models():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "models", "BM1684X"))


class Encoder:
    """编码端各步骤绑到 bmodel 引擎。"""

    def __init__(self, models, precision="bf16", dev_id=0):
        self.precision = precision
        p = gc.resolve_precision(precision)
        self.profile = p
        self.encoder = BmrtEngine(
            os.path.join(models, "glc_encoder_%s.bmodel" % p["encoder"]), dev_id)
        self.prior = BmrtEngine(
            os.path.join(models, "glc_prior_init_%s.bmodel" % p["prior_init"]), dev_id)
        self.stages = {
            s: BmrtEngine(
                os.path.join(models, "glc_stage_%d_%s.bmodel" % (s, p["stage"])), dev_id)
            for s in (1, 2, 3)
        }
        self.codebook = np.load(os.path.join(models, "codebook.npy"))
        self.q_enc = np.load(os.path.join(models, "q_enc.npy"))
        self.entropy = gc.EntropyContext.load(os.path.join(models, "entropy_assets.npz"))
        self.rans = RansCoder()

    def close(self):
        self.encoder.close()
        self.prior.close()
        for e in self.stages.values():
            e.close()

    def _out(self, engine, feeds):
        return list(engine.infer(feeds).values())

    def _stage_params(self, pin, stage, y_hat_so_far):
        """第 0 段用 prior_init 的输出；第 N 段走 glc_stage_N。"""
        if stage == 0:
            return pin["scales"], pin["means"]
        o = self._out(self.stages[stage], {
            "y_hat_so_far": y_hat_so_far.astype(np.float32),
            "common": pin["common"].astype(np.float32),
        })
        return o[0], o[1]

    def encode(self, x, q_index, original_size):
        """图（1,3,H,W float32 [-1,1]）-> (码流 bytes, y_hat)。

        original_size 是 padding 前的 (h, w)：码流头要同时记录原图与 pad 后尺寸，
        解码端才能把重建图裁回原大小。这两个值不能混用。
        """
        q = self.q_enc[q_index:q_index + 1].astype(np.float32)
        y, z = self._out(self.encoder, {"x": x.astype(np.float32), "q_enc": q})

        # z 走码本 argmin，再 14-bit 定长打包（离散选择，TPU 上不划算）
        zf = z.transpose(0, 2, 3, 1).reshape(-1, EMB_DIM)
        d = ((zf ** 2).sum(1, keepdims=True) + (self.codebook ** 2).sum(1)
             - 2 * zf.dot(self.codebook.T))
        z_idx = d.argmin(1).astype(np.int32)
        z_payload = en.pack_fixed_width(z_idx.tolist(), gc.Z_INDEX_BITS)
        z_hat = self.codebook[z_idx].reshape(z.shape[0], z.shape[2], z.shape[3], z.shape[1])
        z_hat = z_hat.transpose(0, 3, 1, 2).copy()

        pv = self._out(self.prior, {"z_hat": z_hat.astype(np.float32)})
        pin = {"q_enc": pv[0], "q_dec": pv[1], "scales": pv[2],
               "means": pv[3], "common": pv[4]}

        y_scaled = y * pin["q_enc"]
        masks = en.four_part_masks(Y_CH, Y_H, Y_W)
        y_hat_so_far = np.zeros((1, Y_CH, Y_H, Y_W), dtype=np.float32)
        streams = []
        for s, mask in enumerate(masks):
            ss, sm = self._stage_params(pin, s, y_hat_so_far)
            packed_y = en.compact_mask(y_scaled, mask)
            packed_scales = en.compact_mask(ss, mask)
            packed_means = en.compact_mask(sm, mask)
            idxs = en.build_indexes(np.maximum(packed_scales, gc.SCALE_MIN),
                                    self.entropy.scale_table)
            syms = en.quantize(packed_y, packed_means)
            payload = self.rans.encode(
                np.ascontiguousarray(syms.reshape(-1), dtype=np.int32),
                np.ascontiguousarray(idxs.reshape(-1), dtype=np.int32),
                self.entropy.flat, self.entropy.lengths,
                self.entropy.starts, self.entropy.offsets)
            streams.append(payload)
            # 重建值与解码端同构：符号 + means（rANS 无损）
            recon = en.dequantize(syms.reshape(-1), packed_means.reshape(-1))
            recon = recon.reshape(1, Y_CH, Y_H, Y_W // 4)
            y_hat_so_far = y_hat_so_far + en.scatter_mask(recon, mask, (1, Y_CH, Y_H, Y_W))

        # 逐像素 q_dec 来自先验网络；tail 的 q_dec 输入是模型参数，两者不同
        y_hat = y_hat_so_far * pin["q_dec"]

        orig_h, orig_w = original_size
        header = gc.HEADER.pack(
            gc.MAGIC, gc.VERSION, q_index, 0, orig_h, orig_w,
            x.shape[2], x.shape[3],
            Y_CH, Y_H, Y_W, Z_CH, Z_H, Z_W,
            int(z_idx.size), len(z_payload), *(len(s) for s in streams), 0)
        return header + z_payload + b"".join(streams), y_hat


def main():
    ap = argparse.ArgumentParser(description="GLC 压缩端（真熵编码）")
    ap.add_argument("--models", default=default_models(), help="bmodel 与资产目录")
    ap.add_argument("--precision", default="bf16",
                    choices=["fp32", "bf16", "mixed"],
                    help="fp32/bf16 全程单精度；mixed 为逐子图混合精度"
                         "（跨芯片部署必须用 mixed 或 fp32，见文件头说明）")
    ap.add_argument("--input", required=True, help="输入图像路径")
    ap.add_argument("--out", required=True, help="输出 .glc 码流路径")
    ap.add_argument("--q_index", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--dev_id", type=int, default=0)
    ap.add_argument("--dump", default=None,
                    help="可选：把编码端重建的 y_hat 存成 npz，供分离性对账")
    args = ap.parse_args()

    img = load_image(args.input)
    h, w = img.shape[:2]
    padded, _ = pad_to_multiple(img)
    x = preprocess(padded)

    enc = Encoder(args.models, args.precision, args.dev_id)
    try:
        t0 = time.perf_counter()
        bitstream, y_hat = enc.encode(x, args.q_index, (h, w))
        elapsed = (time.perf_counter() - t0) * 1000
    finally:
        enc.close()

    with open(args.out, "wb") as f:
        f.write(bitstream)

    if args.dump:
        np.savez(args.dump, y_hat=y_hat.astype(np.float32))

    bd = gc.bpp_breakdown(bitstream, h, w)
    print("输入 %s  %dx%d (pad 后 %dx%d)" % (
        os.path.basename(args.input), w, h, padded.shape[1], padded.shape[0]))
    print("码流 %d 字节 -> %s" % (len(bitstream), args.out))
    print("  真实 bpp %.6f = y %.6f + z %.6f + header %.6f"
          % (bd["total"], bd["y"], bd["z"], bd["header"]))
    print("  四段 y 字节数 %s" % bd["y_bytes"])
    print("编码耗时 %.1f ms" % elapsed)


if __name__ == "__main__":
    main()
