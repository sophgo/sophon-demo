#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""SE7/SE9 真机端到端：bmodel 编码端 -> 真码流 -> bmodel 解码端。

即真实部署形态：编码端只产出 .glc 码流，解码端只读该码流，
两者之间不传递任何张量。
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # tools/tests -> GLC
sys.path.insert(0, os.path.join(PROJ, "python"))

import entropy_numpy as en
import glc_codec as gc
from bmrt import BmrtEngine

from rans_coder import RansCoder

CHIP = os.environ.get("GLC_CHIP", "BM1684X")  # BM1684X(SE7-32) / BM1688(SE9-16)
M = os.environ.get("GLC_MODELS", os.path.join(PROJ, "models", CHIP))
RANS = RansCoder()
SCALE_MIN = 0.11

entropy = gc.EntropyContext.load("%s/entropy_assets.npz" % M)
codebook = np.load("%s/codebook.npy" % M)
q_enc_all = np.load("%s/q_enc.npy" % M)
q_dec_all = np.load("%s/q_dec.npy" % M)

enc = BmrtEngine("%s/glc_encoder_bf16.bmodel" % M)
prior = BmrtEngine("%s/glc_prior_init_bf16.bmodel" % M)
stage = {i: BmrtEngine("%s/glc_stage_%d_bf16.bmodel" % (M, i)) for i in (1, 2, 3)}
tail = BmrtEngine("%s/glc_tail_bf16.bmodel" % M)


def stage_params(net_out, s, y_hat_so_far, common):
    """第 0 段用 prior_init 的输出，第 N 段走 glc_stage_N。"""
    if s == 0:
        return net_out["scales"], net_out["means"]
    o = stage[s].infer({"y_hat_so_far": y_hat_so_far, "common": common})
    ks = list(o.keys())
    return o[ks[0]], o[ks[1]]


def encode(x, q_index):
    q = q_enc_all[q_index:q_index + 1].astype(np.float32)
    o = enc.infer({"x": x.astype(np.float32), "q_enc": q})
    ks = list(o.keys())
    y, z = o[ks[0]], o[ks[1]]

    # z 码本 argmin + 14-bit 定长打包
    zf = z.transpose(0, 2, 3, 1).reshape(-1, 256)
    d = (zf ** 2).sum(1, keepdims=True) + (codebook ** 2).sum(1) - 2 * zf.dot(codebook.T)
    z_idx = d.argmin(1).astype(np.int32)
    z_payload = en.pack_fixed_width(z_idx.tolist(), gc.Z_INDEX_BITS)
    z_hat = codebook[z_idx].reshape(z.shape[0], z.shape[2], z.shape[3], z.shape[1])
    z_hat = z_hat.transpose(0, 3, 1, 2).copy()

    o = prior.infer({"z_hat": z_hat.astype(np.float32)})
    ks = list(o.keys())
    pin = {"q_enc": o[ks[0]], "q_dec": o[ks[1]], "scales": o[ks[2]],
           "means": o[ks[3]], "common": o[ks[4]]}

    y_scaled = y * pin["q_enc"]
    masks = en.four_part_masks(256, 32, 48)
    y_hat_so_far = np.zeros_like(y)
    streams = []
    for s, mask in enumerate(masks):
        ss, sm = stage_params(pin, s, y_hat_so_far.astype(np.float32), pin["common"])
        packed_y = en.compact_mask(y_scaled, mask)
        packed_scales = en.compact_mask(ss, mask)
        packed_means = en.compact_mask(sm, mask)
        idxs = en.build_indexes(np.maximum(packed_scales, SCALE_MIN), entropy.scale_table)
        syms = en.quantize(packed_y, packed_means)
        flat_syms = np.ascontiguousarray(syms.reshape(-1), dtype=np.int32)
        flat_idx = np.ascontiguousarray(idxs.reshape(-1), dtype=np.int32)
        payload = RANS.encode(flat_syms, flat_idx, entropy.flat, entropy.lengths,
                              entropy.starts, entropy.offsets)
        streams.append(payload)
        recon = en.dequantize(flat_syms, packed_means.reshape(-1))
        recon_t = recon.reshape(1, 256, 32, 12)
        y_hat_so_far = y_hat_so_far + en.scatter_mask(recon_t, mask, y.shape)

    # 逐像素 q_dec 来自 prior 网络输出；tail 的 q_dec 输入是模型参数，两者不同
    y_hat = y_hat_so_far * pin["q_dec"]
    return y_hat, z_payload, streams, pin


def decode(bitstream):
    q_index, dims, streams = gc.parse_bitstream(bitstream)
    z_payload, y_streams = streams[0], streams[1:]
    z_count = dims[10]
    z_idx = np.array(en.unpack_fixed_width(z_payload, z_count, gc.Z_INDEX_BITS), dtype=np.int64)
    z_hat = codebook[z_idx].reshape(1, dims[8], dims[9], dims[7]).transpose(0, 3, 1, 2).copy()

    o = prior.infer({"z_hat": z_hat.astype(np.float32)})
    ks = list(o.keys())
    pin = {"q_dec": o[ks[1]], "scales": o[ks[2]], "means": o[ks[3]], "common": o[ks[4]]}

    masks = en.four_part_masks(256, 32, 48)
    y_hat_so_far = np.zeros((1, 256, 32, 48), dtype=np.float32)
    for s, (mask, stream) in enumerate(zip(masks, y_streams)):
        ss, sm = stage_params(pin, s, y_hat_so_far, pin["common"])
        packed_scales = en.compact_mask(ss, mask)
        packed_means = en.compact_mask(sm, mask)
        idxs = en.build_indexes(np.maximum(packed_scales, SCALE_MIN), entropy.scale_table)
        flat_idx = np.ascontiguousarray(idxs.reshape(-1), dtype=np.int32)
        n = flat_idx.size
        syms = RANS.decode(stream, n, flat_idx, entropy.flat, entropy.lengths,
                           entropy.starts, entropy.offsets)
        recon = en.dequantize(syms, packed_means.reshape(-1)).reshape(1, 256, 32, 12)
        y_hat_so_far = y_hat_so_far + en.scatter_mask(recon, mask, (1, 256, 32, 48))

    qd = q_dec_all[q_index:q_index + 1].astype(np.float32)
    y_hat = y_hat_so_far * pin["q_dec"]
    o = tail.infer({"y_hat": y_hat.astype(np.float32), "q_dec": qd})
    ks = list(o.keys())
    return o[ks[0]], y_hat


if __name__ == "__main__":
    # 传入真实图片路径即可（内部按 glc_encode.py 的同一套预处理转成模型输入）；
    # 不传则用 512x768 的全零张量跑通链路。
    if len(sys.argv) > 1:
        from glc_encode import load_image, pad_to_multiple, preprocess
        padded, _ = pad_to_multiple(load_image(sys.argv[1]))
        x = preprocess(padded).astype(np.float32)
    else:
        x = np.zeros((1, 3, 512, 768), dtype=np.float32)
    H, W = x.shape[2], x.shape[3]
    q_index = 0

    t0 = time.perf_counter()
    enc_y_hat, z_payload, streams, pin = encode(x, q_index)
    enc_ms = (time.perf_counter() - t0) * 1000

    header = gc.HEADER.pack(gc.MAGIC, gc.VERSION, q_index, 0, H, W,
                            x.shape[2], x.shape[3], 256, 32, 48, 256, 8, 12,
                            z_payload and 96, len(z_payload),
                            *(len(s) for s in streams), 0)
    bitstream = header + z_payload + b"".join(streams)
    with open("/tmp/glc_e2e_out.glc", "wb") as f:
        f.write(bitstream)

    with open("/tmp/glc_e2e_out.glc", "rb") as f:
        received = f.read()

    t0 = time.perf_counter()
    x_hat, dec_y_hat = decode(received)
    dec_ms = (time.perf_counter() - t0) * 1000

    bd = gc.bpp_breakdown(received, H, W)
    print("=== 真机端到端（bmodel 编码 + 真熵编码 + bmodel 解码）===")
    print("  码流 %d 字节  = header %d + z %d + 四段 y %s"
          % (len(received), bd["header_bytes"], bd["z_bytes"], bd["y_bytes"]))
    print("  真实 bpp %.6f = y %.6f + z %.6f + header %.6f"
          % (bd["total"], bd["y"], bd["z"], bd["header"]))
    print("  编码端 %.1f ms   解码端 %.1f ms" % (enc_ms, dec_ms))
    same = bool(np.array_equal(enc_y_hat, dec_y_hat))
    print("  两端 y_hat 逐位相同: %s" % same)
    print("  x_hat 形状 %s  范围 [%.3f, %.3f]"
          % (x_hat.shape, float(x_hat.min()), float(x_hat.max())))

    # 存下重建图与 y_hat，供主机侧对齐比对
    np.savez("/tmp/glc_e2e_out.npz", x_hat=x_hat.astype(np.float32),
             y_hat=dec_y_hat.astype(np.float32))
    print("  已存 /tmp/glc_e2e_out.npz")

