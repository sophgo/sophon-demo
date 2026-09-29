#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""GLC 还原端：真熵编码码流（.glc）-> 重建图。

这是部署中的「还原端」独立进程：只读编码端产出的 .glc 文件，
不接收任何张量。真实部署时它与压缩端分处两地，中间只有无线信道。

本脚本所需的一切形状信息都从码流头读取（header 自带原图/pad 后尺寸、
y 与 z 的通道数及空间尺寸），因此不需要与编码端共享任何配置——
这正是「两端可独立部署」的前提。

依赖（设备端）：numpy + pillow，不需要 torch / compressai / scipy。

用法:
    python3 glc_decode.py --precision bf16 --input out.glc --out recon.png
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
from glc_infer import postprocess    # noqa: E402
from rans_coder import RansCoder    # noqa: E402


def _sanity_check(y_hat, stage_means_abs, bitstream):
    """拦住明显退化的解码结果。

    两端熵模型子图（prior_init / stage_1..3）精度不一致时，个别 scale 会落到不同的
    CDF 档，rANS 一旦失步就静默解出错误符号，最终产出全 -1 的垃圾图而不报错。
    这里做一次残差量级自检：解码出的 y_hat 若比先验给出的 means 量级大出好几个
    数量级，说明码流与当前模型精度不匹配，直接报错而不是交出坏图。
    """
    if not np.all(np.isfinite(y_hat)):
        raise ValueError(
            "解码出的 y_hat 含 NaN/Inf：码流与当前 --precision 模型不匹配")
    amp = float(np.abs(y_hat).max())
    scale = max(abs(stage_means_abs), 1.0)
    if amp > 1e4 * scale:
        raise ValueError(
            "解码结果量级异常（|y_hat|max=%.3g，先验 means 量级 %.3g）：\n"
            "  码流与当前 --precision 的模型不匹配。\n"
            "  两端用于熵模型的子图（prior_init / stage_1..3）必须精度一致，\n"
            "  否则 CDF 行选择会分歧、rANS 解出的符号全错。\n"
            "  跨芯片部署请两端都用 --precision mixed（见 README Q10）。"
            % (amp, stage_means_abs))


def default_models():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "models", "BM1684X"))


class Decoder:
    """还原端各步骤绑到 bmodel 引擎。

    只需要 glc_prior_init / glc_stage_1..3 / glc_tail 五个子图，
    不需要 encoder——两端模型集合本身就不重叠。
    """

    def __init__(self, models, precision="bf16", dev_id=0):
        self.precision = precision
        p = gc.resolve_precision(precision)
        self.profile = p
        self.prior = BmrtEngine(
            os.path.join(models, "glc_prior_init_%s.bmodel" % p["prior_init"]), dev_id)
        self.stages = {
            s: BmrtEngine(
                os.path.join(models, "glc_stage_%d_%s.bmodel" % (s, p["stage"])), dev_id)
            for s in (1, 2, 3)
        }
        self.tail = BmrtEngine(
            os.path.join(models, "glc_tail_%s.bmodel" % p["tail"]), dev_id)
        self.codebook = np.load(os.path.join(models, "codebook.npy"))
        self.q_dec = np.load(os.path.join(models, "q_dec.npy"))
        self.entropy = gc.EntropyContext.load(os.path.join(models, "entropy_assets.npz"))
        self.rans = RansCoder()

    def close(self):
        self.prior.close()
        for e in self.stages.values():
            e.close()
        self.tail.close()

    def _out(self, engine, feeds):
        return list(engine.infer(feeds).values())

    def _stage_params(self, pin, stage, y_hat_so_far):
        if stage == 0:
            return pin["scales"], pin["means"]
        o = self._out(self.stages[stage], {
            "y_hat_so_far": y_hat_so_far.astype(np.float32),
            "common": pin["common"].astype(np.float32),
        })
        return o[0], o[1]

    def decode(self, bitstream):
        """码流 bytes -> (x_hat, y_hat)。全程只用码流里带的信息。"""
        q_index, dims, streams = gc.parse_bitstream(bitstream)
        (orig_h, orig_w, pad_h, pad_w, y_ch, y_h, y_w,
         z_ch, z_h, z_w, z_count) = dims
        z_payload, y_streams = streams[0], streams[1:]

        if z_count != z_h * z_w:
            raise ValueError("超先验索引数 %d 与 %dx%d 不符" % (z_count, z_h, z_w))

        z_idx = np.array(en.unpack_fixed_width(z_payload, z_count, gc.Z_INDEX_BITS),
                         dtype=np.int64)
        z_hat = self.codebook[z_idx].reshape(1, z_h, z_w, z_ch)
        z_hat = z_hat.transpose(0, 3, 1, 2).copy()

        pv = self._out(self.prior, {"z_hat": z_hat.astype(np.float32)})
        pin = {"q_dec": pv[1], "scales": pv[2], "means": pv[3], "common": pv[4]}

        masks = en.four_part_masks(y_ch, y_h, y_w)
        y_hat_so_far = np.zeros((1, y_ch, y_h, y_w), dtype=np.float32)
        stage_means_abs = 0.0
        for s, (mask, stream) in enumerate(zip(masks, y_streams)):
            ss, sm = self._stage_params(pin, s, y_hat_so_far)
            packed_scales = en.compact_mask(ss, mask)
            packed_means = en.compact_mask(sm, mask)
            idxs = en.build_indexes(np.maximum(packed_scales, gc.SCALE_MIN),
                                    self.entropy.scale_table)
            flat_idx = np.ascontiguousarray(idxs.reshape(-1), dtype=np.int32)
            syms = self.rans.decode(stream, flat_idx.size, flat_idx,
                                    self.entropy.flat, self.entropy.lengths,
                                    self.entropy.starts, self.entropy.offsets)
            recon = en.dequantize(syms, packed_means.reshape(-1))
            recon = recon.reshape(1, y_ch, y_h, y_w // 4)
            y_hat_so_far = y_hat_so_far + en.scatter_mask(
                recon, mask, (1, y_ch, y_h, y_w))
            stage_means_abs = max(stage_means_abs or 0.0,
                                  float(np.abs(packed_means).max()))

        # 逐像素 q_dec 来自先验网络；tail 的 q_dec 输入是模型参数，两者不同
        y_hat = y_hat_so_far * pin["q_dec"]

        # 数值自检：跨精度（如编码端 bf16、解码端 fp32）会让个别 scale 落到
        # 不同 CDF 档，rANS 一旦失步就会静默解出错误符号，最终产出全 -1 的
        # 垃圾图而不报错。这里拦住明显退化的结果，宁可报错也不要交出坏图。
        _sanity_check(y_hat, stage_means_abs, bitstream)
        qd = self.q_dec[q_index:q_index + 1].astype(np.float32)
        x_hat = self._out(self.tail, {
            "y_hat": y_hat.astype(np.float32), "q_dec": qd})[0]

        if x_hat.shape[-2:] != (pad_h, pad_w):
            raise ValueError("解码输出尺寸 %s 与码流记录 (%d,%d) 不符"
                             % (x_hat.shape[-2:], pad_h, pad_w))
        return x_hat[:, :, :orig_h, :orig_w], y_hat


def main():
    ap = argparse.ArgumentParser(description="GLC 还原端（真熵解码）")
    ap.add_argument("--models", default=default_models(), help="bmodel 与资产目录")
    ap.add_argument("--precision", default="bf16",
                    choices=["fp32", "bf16", "mixed"],
                    help="fp32/bf16 全程单精度；mixed 为逐子图混合精度"
                         "（跨芯片部署必须用 mixed 或 fp32，见文件头说明）")
    ap.add_argument("--input", required=True, help="输入 .glc 码流路径")
    ap.add_argument("--out", default=None, help="重建图输出路径（.png）")
    ap.add_argument("--dev_id", type=int, default=0)
    ap.add_argument("--dump", default=None,
                    help="可选：把还原端解出的 y_hat 存成 npz，供分离性对账")
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        bitstream = f.read()

    dec = Decoder(args.models, args.precision, args.dev_id)
    try:
        t0 = time.perf_counter()
        x_hat, y_hat = dec.decode(bitstream)
        elapsed = (time.perf_counter() - t0) * 1000
    finally:
        dec.close()

    q_index, dims, streams = gc.parse_bitstream(bitstream)
    orig_h, orig_w = dims[0], dims[1]
    bd = gc.bpp_breakdown(bitstream, orig_h, orig_w)

    print("码流 %s  %d 字节" % (os.path.basename(args.input), len(bitstream)))
    print("  头部记录：q_index=%d  原图 %dx%d  pad 后 %dx%d"
          % (q_index, orig_h, orig_w, dims[2], dims[3]))
    print("  真实 bpp %.6f = y %.6f + z %.6f + header %.6f"
          % (bd["total"], bd["y"], bd["z"], bd["header"]))
    print("重建图 %s  范围 [%.3f, %.3f]" % (x_hat.shape, x_hat.min(), x_hat.max()))
    print("解码耗时 %.1f ms" % elapsed)

    if args.dump:
        np.savez(args.dump, y_hat=y_hat.astype(np.float32))

    # 输出退化检查：跨精度互解时曾观察到重建图退化成常量（整幅 -1），
    # 这种图不报错但完全无用，必须在写出前拦住。
    if float(x_hat.std()) < 1e-6:
        raise SystemExit(
            "重建图退化为常量（std=%.3e）：码流与当前 --precision 的模型不匹配。\n"
            "  编码端与还原端必须使用同一精度的模型。" % float(x_hat.std()))

    if args.out:
        from PIL import Image
        Image.fromarray(postprocess(x_hat)).save(args.out)
        print("已写出 %s" % args.out)


if __name__ == "__main__":
    main()
