#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""生成设备端所需的熵编码静态资产。

CDF 表只依赖固定 64 档 scale_table（与图像无关），在主机用 compressai 预算一次，
存成 npz 随模型资产下发。设备端只读这个文件，不需要 compressai 也不需要 scipy。

同时把 64 档 scale_table 一起存进去，保证两端用的是同一张表——
这是两端 CDF 不一致导致码流无法互解的唯一防线。

用法：
    PYTHONPATH=tools/vendor:. python3 make_entropy_assets.py --out models/entropy_assets.npz
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "python"))

import numpy as np  # noqa: E402

import entropy_numpy as en  # noqa: E402
import glc_codec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "assets", "entropy_assets.npz"))
    ap.add_argument("--tail-mass", type=float, default=1e-9,
                    help="与客户参考实现一致；改这个值会改变 CDF，两端必须同值")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    entropy = glc_codec.EntropyContext.from_compressai(tail_mass=args.tail_mass)
    entropy.save(args.out)

    size = os.path.getsize(args.out)
    print(f"已写入 {args.out}（{size} 字节）")
    print(f"  scale_table : {entropy.scale_table.size} 档，"
          f"{entropy.scale_table.min():.4f} ~ {entropy.scale_table.max():.4f}")
    print(f"  CDF 行数    : {entropy.lengths.size}")
    print(f"  行有效长度  : {entropy.lengths.min()} ~ {entropy.lengths.max()}")
    print(f"  扁平 CDF    : {entropy.flat.size} 个 int32")

    # 往返自检：设备端读回来的东西必须与写入时逐字节等价
    back = glc_codec.EntropyContext.load(args.out)
    same = (np.array_equal(back.flat, entropy.flat)
            and np.array_equal(back.starts, entropy.starts)
            and np.array_equal(back.lengths, entropy.lengths)
            and np.array_equal(back.offsets, entropy.offsets)
            and np.array_equal(back.scale_table, entropy.scale_table))
    if not same:
        print("资产往返自检失败：读回与写入不一致", file=sys.stderr)
        return 1
    print("资产往返自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
