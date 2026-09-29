#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""numpy 四段掩码与官方 torch 版逐位对账。

掩码决定「每个符号属于哪一段」，两端不一致码流就解不出来——
与两端 CDF 口径不一致无法互解同源，单独锁住。

官方四行拼接不是循环移位：mask_1 用 (m3,m2,m1,m0) 逆序、
mask_2 用 (m2,m3,m0,m1)、mask_3 用 (m1,m0,m3,m2)，照抄才不会错位。

运行：PYTHONPATH=tools/vendor:python:. python3 tests/test_mask_parity.py
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # tools/tests -> GLC
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "python"))
sys.path.insert(0, os.path.join(PROJ, "GLC_src"))
sys.path.insert(0, os.path.join(PROJ, "tools", "stubs"))

import entropy_numpy as en  # noqa: E402

C, H, W = 256, 32, 48
_failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def main():
    from src.models.common_model import CompressionModel

    cm = CompressionModel()
    ref = cm.get_mask_four_parts(1, C, H, W, torch.float32, torch.device("cpu"))
    mine = en.four_part_masks(C, H, W)
    check("返回四段掩码", len(mine) == 4, f"{len(mine)}")

    total = C * H * W // 4
    for i, (r, m) in enumerate(zip(ref, mine)):
        rn = r.numpy().astype(bool)
        check(f"stage{i} 元素数 = {total}", int(m.sum()) == total, f"{int(m.sum())}")
        check(f"stage{i} 与 torch 版逐位一致", np.array_equal(rn, m))

    # 四段两两不重叠且并集覆盖全部元素——这是四段划分成立的前提
    union = np.zeros((C, H, W), dtype=bool)
    overlap = False
    for m in mine:
        mm = m[0]
        if np.any(union & mm):
            overlap = True
        union |= mm
    check("四段互不重叠", not overlap)
    check("四段并集覆盖全部元素", bool(union.all()), f"{int(union.sum())}")

    print()
    if _failures:
        print(f"失败 {len(_failures)} 项: {_failures}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
