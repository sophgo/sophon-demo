#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""按客户样例给出的参数对齐验证。

客户样例 test_image_complement2.py 里的常量：
    SCALE_MIN=0.11  SCALE_MAX=256.0  SCALE_LEVELS=64
    Z_INDEX_BITS=14  MAGIC=b"GLC2"  VERSION=2  HEADER=">4sBBH17I"
样例没有写 compressai 版本号，因此这里验证参数侧的对齐：
  1. 按样例参数构造的 CDF 与我们下发的资产逐位相同（参数已对齐）
  2. 这些参数确实会影响码流（说明对齐是必要的，不是形式）

版本侧另行实测：compressai 1.1.9 / 1.2.0 / 1.2.4 / 1.2.7 / 1.2.8 五个版本，
相同参数下 CDF 与码流 md5 完全相同，即版本不影响码流。

运行：PYTHONPATH=tools/vendor:python:. python3 tests/test_param_alignment.py
"""
import hashlib
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # tools/tests -> GLC
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "python"))

import glc_codec as gc  # noqa: E402

# 客户样例里的常量，逐字抄自 test_image_complement2.py:29-35
SAMPLE_SCALE_MIN = 0.11
SAMPLE_SCALE_MAX = 256.0
SAMPLE_SCALE_LEVELS = 64
SAMPLE_Z_INDEX_BITS = 14
SAMPLE_MAGIC = b"GLC2"
SAMPLE_VERSION = 2
SAMPLE_HEADER = ">4sBBH17I"

_failures = []


def check(name, ok, detail=""):
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         (" - " + detail) if detail else ""))
    if not ok:
        _failures.append(name)


def _sample_cdf(tail_mass=1e-9, scale_min=SAMPLE_SCALE_MIN,
                scale_max=SAMPLE_SCALE_MAX, levels=SAMPLE_SCALE_LEVELS):
    """完全按样例 ELICGaussianConditional.__init__ 的写法构造模型。"""
    from compressai.entropy_models import GaussianConditional
    st = torch.exp(torch.linspace(math.log(scale_min), math.log(scale_max), levels))
    m = GaussianConditional(None, tail_mass=tail_mass)
    m.update_scale_table(st, force=True)
    return m, st


def _bitstream_fingerprint(model, scale_table):
    """用固定的符号/索引算一份码流指纹，供跨参数比较。"""
    from rans_coder import RansCoder
    rng = np.random.default_rng(2024)
    n = 20000
    idx = rng.integers(0, 24, size=n).astype(np.int32)
    st_np = scale_table.numpy() if hasattr(scale_table, "numpy") else np.asarray(scale_table)
    scales = st_np[idx].astype(np.float32)
    means = rng.normal(0, 1, n).astype(np.float32)
    values = (means + rng.normal(0, 1.5, n)).astype(np.float32)
    ts = torch.from_numpy(scales.reshape(1, 1, 1, n))
    tm = torch.from_numpy(means.reshape(1, 1, 1, n))
    tv = torch.from_numpy(values.reshape(1, 1, 1, n))
    indexes = model.build_indexes(ts)
    payload = model.compress(tv, indexes, means=tm)[0]
    return hashlib.md5(payload).hexdigest()[:10], len(payload)

def main():
    # --- 1. sample params vs shipped asset ---
    model, st = _sample_cdf()
    shipped = gc.EntropyContext.load(
        os.path.join(PROJ, "models", "BM1684X", "entropy_assets.npz"))
    sample = gc.EntropyContext(
        model._quantized_cdf.numpy(), model._cdf_length.numpy(),
        model._offset.numpy(), st.numpy().astype(np.float32))
    check("sample-params CDF == shipped asset (flat)",
          np.array_equal(sample.flat, shipped.flat))
    check("sample-params cdf_length == shipped",
          np.array_equal(sample.lengths, shipped.lengths))
    check("sample-params offsets == shipped",
          np.array_equal(sample.offsets, shipped.offsets))
    check("sample-params scale_table == shipped",
          np.array_equal(sample.scale_table, shipped.scale_table))

    # --- 2. header constants vs our module ---
    check("Z_INDEX_BITS matches sample", gc.Z_INDEX_BITS == SAMPLE_Z_INDEX_BITS,
          "ours=%d sample=%d" % (gc.Z_INDEX_BITS, SAMPLE_Z_INDEX_BITS))
    check("MAGIC matches sample", gc.MAGIC == SAMPLE_MAGIC, repr(gc.MAGIC))
    check("VERSION matches sample", gc.VERSION == SAMPLE_VERSION)
    check("HEADER struct matches sample",
          gc.HEADER.format == SAMPLE_HEADER,
          "ours=%s sample=%s" % (gc.HEADER.format, SAMPLE_HEADER))
    check("HEADER size == 76 bytes", gc.HEADER.size == 76,
          "%d" % gc.HEADER.size)


    # --- 3. parameters are bitstream-sensitive (so alignment matters) ---
    base_fp = _bitstream_fingerprint(model, st)
    variants = [
        ("SCALE_MIN=0.10", dict(scale_min=0.10)),
        ("SCALE_MAX=255", dict(scale_max=255.0)),
        ("SCALE_LEVELS=63", dict(levels=63)),
        ("tail_mass=1e-8", dict(tail_mass=1e-8)),
    ]
    for label, kw in variants:
        m2, st2 = _sample_cdf(**kw)
        fp = _bitstream_fingerprint(m2, st2)
        check("changing %s changes the bitstream" % label,
              fp != base_fp, "base=%s variant=%s" % (base_fp[0], fp[0]))

    print()
    if _failures:
        print("失败 %d 项: %s" % (len(_failures), _failures))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
