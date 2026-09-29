#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""自实现 rANS + entropy model 层与 compressai 官方实现逐位对齐。

真熵编码路径的基础验收。检验项：
  1. pack/unpack_fixed_width 往返一致
  2. numpy build_indexes 与 torch 版逐元素相同
  3. rANS 编码字节流与 compressai BufferedRansEncoder 完全相同
  4. rANS 解码符号与压缩前符号完全相同
  5. 大规模随机符号（含逃逸路径）下两项仍然成立

运行：PYTHONPATH=tools/vendor:python:. python3 tools/tests/test_rans_parity.py
"""

import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # tools/tests -> GLC
sys.path.insert(0, os.path.join(PROJ, "python"))

from compressai.entropy_models import GaussianConditional  # noqa: E402

import entropy_numpy as en  # noqa: E402
from rans_coder import RansCoder  # noqa: E402

SCALE_MIN = 0.11
SCALE_MAX = 256.0
SCALE_LEVELS = 64

_failures = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def build_gaussian():
    scale_table = torch.exp(
        torch.linspace(math.log(SCALE_MIN), math.log(SCALE_MAX), SCALE_LEVELS)
    )
    model = GaussianConditional(None)
    model.update_scale_table(scale_table, force=True)
    return model, scale_table


def cdf_arrays(model):
    """把 compressai 的 CDF 表摊平成 C 侧需要的 (扁平表, 行起始, 有效长度, offsets)。

    _quantized_cdf 每行宽度是 max_length+2，实际有效长度由 _cdf_length 给出，
    必须按有效长度裁，否则解码端会在填充零上多做无谓比较。
    """
    q = model._quantized_cdf.numpy().astype(np.int32)
    lengths = model._cdf_length.numpy().astype(np.int32)
    offsets = model._offset.numpy().astype(np.int32)
    rows = [q[i, : int(lengths[i])] for i in range(q.shape[0])]
    flat = np.concatenate(rows).astype(np.int32)
    starts = np.zeros(len(rows), dtype=np.int32)
    acc = 0
    for i, r in enumerate(rows):
        starts[i] = acc
        acc += r.size
    return flat, starts, lengths, offsets


def official_encode(model, symbols, indexes, means):
    enc = model.entropy_coder
    return enc.encode_with_indexes(
        symbols.reshape(-1).int().tolist(),
        indexes.reshape(-1).int().tolist(),
        model._quantized_cdf.tolist(),
        model._cdf_length.reshape(-1).int().tolist(),
        model._offset.reshape(-1).int().tolist(),
    )


def test_fixed_width():
    values = [0, 1, 16383, 37, 8192, 4, 999]
    payload = en.pack_fixed_width(values, 14)
    back = en.unpack_fixed_width(payload, len(values), 14)
    check("14-bit 定长打包往返", back == values, f"{len(payload)} 字节")
    try:
        en.pack_fixed_width([16384], 14)
        check("越界值应报错", False)
    except ValueError:
        check("越界值应报错", True)


def test_build_indexes(model, scale_table):
    rng = np.random.default_rng(0)
    scales = np.exp(rng.uniform(math.log(0.05), math.log(400.0), 4096)).astype(np.float32)
    scales = np.maximum(scales, SCALE_MIN)
    ref = model.build_indexes(torch.from_numpy(scales)).numpy().astype(np.int32)
    mine = en.build_indexes(scales, scale_table.numpy().astype(np.float32))
    same = np.array_equal(ref, mine)
    check("numpy build_indexes 与 torch 版一致", same,
          f"4096 个 scale，取值范围 {scales.min():.4f}~{scales.max():.4f}")


def test_rans_small(model):
    """客户脚本自检用的那组符号，覆盖负值、大正值（逃逸）与 means。"""
    coder = RansCoder()
    means = torch.tensor([[[[0.25, -0.75, 1.5, -2.25, 0.0, 3.125]]]])
    residuals = torch.tensor([[[[0.0, 1.0, -2.0, 5000.0, -5001.0, 7.0]]]])
    values = means + residuals
    scales = torch.tensor([[[[0.11, 0.5, 1.0, 2.0, 16.0, 256.0]]]])
    indexes = model.build_indexes(scales)
    symbols = model.quantize(values, "symbols", means)

    flat, starts, lengths, offsets = cdf_arrays(model)
    ref = official_encode(model, symbols, indexes, means)
    mine = coder.encode(symbols.numpy(), indexes.numpy(), flat, lengths, starts, offsets)
    check("小样本码流与官方字节级一致", mine == ref,
          f"官方 {len(ref)} 字节 / 本实现 {len(mine)} 字节")

    dec = coder.decode(ref, symbols.numel(), indexes.numpy().reshape(-1), flat, lengths,
                       starts, offsets)
    want = symbols.numpy().reshape(-1)
    check("小样本解码符号完全还原", np.array_equal(dec, want), f"{dec.tolist()}")


def test_rans_bulk(model):
    """大规模随机：4000 个符号、随机 scale 索引、含大残差触发 bypass 逃逸。"""
    coder = RansCoder()
    rng = np.random.default_rng(1234)
    n = 4000
    scale_choices = rng.choice(SCALE_LEVELS, size=n).astype(np.float32)
    index_values = np.clip(scale_choices, 0, SCALE_LEVELS - 1)
    # 用 scale_table 反查真实 scales，保证 indexes 与 scales 自洽
    scale_table = model.scale_table.numpy()
    scales = scale_table[index_values.astype(np.int64)]
    means = rng.normal(0, 1.0, n).astype(np.float32)
    residual = rng.normal(0, 3.0, n).astype(np.float32)
    # 混入大值，逼出 bypass 逃逸路径
    residual[rng.choice(n, 20, replace=False)] = rng.normal(0, 50000, 20)

    values_t = torch.from_numpy(means + residual).reshape(1, 1, 1, n)
    scales_t = torch.from_numpy(scales).reshape(1, 1, 1, n)
    means_t = torch.from_numpy(means).reshape(1, 1, 1, n)
    indexes = model.build_indexes(scales_t)
    symbols = model.quantize(values_t, "symbols", means_t)

    flat, starts, lengths, offsets = cdf_arrays(model)
    ref = official_encode(model, symbols, indexes, means_t)
    mine = coder.encode(symbols.numpy(), indexes.numpy(), flat, lengths, starts, offsets)
    check("4000 符号码流与官方字节级一致", mine == ref,
          f"官方 {len(ref)} 字节 / 本实现 {len(mine)} 字节")

    dec = coder.decode(ref, n, indexes.numpy().reshape(-1), flat, lengths, starts, offsets)
    want = symbols.numpy().reshape(-1)
    mismatch = int((dec != want).sum())
    check("4000 符号（含逃逸）解码完全还原", mismatch == 0, f"不一致 {mismatch} 个")

    big = int((symbols.numpy().reshape(-1) > 1000).sum()) + int(
        (symbols.numpy().reshape(-1) < -1000).sum()
    )
    check("确有符号走到 bypass 逃逸路径", big > 0, f"|symbol|>1000 的有 {big} 个")


def test_realistic_bpp(model):
    """按 GLC 512x768 的 y 张量规模（256x32x48 = 393216）估算真实码流长度。

    用合成符号确认符号数与码流长度的量级关系；
    真实图像的端到端数字见 test_roundtrip 与 docs/GLC_Entropy_Coding.md。
    """
    coder = RansCoder()
    rng = np.random.default_rng(7)
    n = 393216
    idx = rng.integers(0, 24, size=n).astype(np.int32)  # 低 scale 区域为主
    scale_table = model.scale_table.numpy()
    scales = scale_table[idx.astype(np.int64)]
    means = rng.normal(0, 1.0, n).astype(np.float32)
    values = (means + rng.normal(0, 1.0, n)).astype(np.float32)

    values_t = torch.from_numpy(values).reshape(1, 1, 1, n)
    scales_t = torch.from_numpy(scales).reshape(1, 1, 1, n)
    means_t = torch.from_numpy(means).reshape(1, 1, 1, n)
    indexes = model.build_indexes(scales_t)
    symbols = model.quantize(values_t, "symbols", means_t)

    flat, starts, lengths, offsets = cdf_arrays(model)
    ref = official_encode(model, symbols, indexes, means_t)
    mine = coder.encode(symbols.numpy(), indexes.numpy(), flat, lengths, starts, offsets)
    check("393216 符号码流与官方字节级一致", mine == ref,
          f"官方 {len(ref)} / 本实现 {len(mine)} 字节")

    dec = coder.decode(mine, n, indexes.numpy().reshape(-1), flat, lengths, starts, offsets)
    check("393216 符号解码完全还原", np.array_equal(dec, symbols.numpy().reshape(-1)))


def test_customer_entry(model, scale_table):
    """走客户 ELICGaussianConditional 的完整入口：build_indexes + compress。

    前面几个用例直接调 entropy_coder.encode_with_indexes，跳过了一层；
    这里走客户脚本真正用的 GaussianConditional.compress()，把
    build_indexes / quantize / CDF 切片整条链一起盖上。
    """
    coder = RansCoder()
    rng = np.random.default_rng(2024)
    n = 20000
    idx = rng.integers(0, 24, size=n).astype(np.int32)
    st_np = scale_table.numpy() if hasattr(scale_table, "numpy") else np.asarray(scale_table)
    scales = np.maximum(st_np[idx.astype(np.int64)], 0.11).astype(np.float32)
    means = rng.normal(0, 1.0, n).astype(np.float32)
    values = (means + rng.normal(0, 1.5, n)).astype(np.float32)

    t = torch.from_numpy(values.reshape(1, 1, 1, n))
    ts = torch.from_numpy(scales.reshape(1, 1, 1, n))
    tm = torch.from_numpy(means.reshape(1, 1, 1, n))
    indexes = model.build_indexes(ts)
    ref = model.compress(t, indexes, means=tm)[0]

    flat, starts, lengths, offsets = cdf_arrays(model)
    sym = model.quantize(t, "symbols", tm)
    mine = coder.encode(sym.numpy(), indexes.numpy(), flat, lengths, starts, offsets)
    check("客户 compress() 入口码流与官方字节级一致", mine == ref,
          f"官方 {len(ref)} / 本实现 {len(mine)} 字节")

    dec = coder.decode(ref, n, indexes.numpy().reshape(-1), flat, lengths, starts, offsets)
    check("客户 compress() 入口解码完全还原",
          np.array_equal(dec, sym.numpy().reshape(-1)))


def main():
    model, scale_table = build_gaussian()
    print(f"scale_table: {SCALE_LEVELS} 档, "
          f"cdf_length {int(model._cdf_length.min())}~{int(model._cdf_length.max())}\n")
    test_fixed_width()
    test_build_indexes(model, scale_table)
    test_rans_small(model)
    test_rans_bulk(model)
    test_customer_entry(model, scale_table)
    test_realistic_bpp(model)
    print()
    if _failures:
        print(f"失败 {len(_failures)} 项: {_failures}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
