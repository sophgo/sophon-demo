#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""逐子图精度档案（--precision fp32/bf16/mixed）的检查。

混合精度是跨芯片互解的前提，档案本身出错会导致「编译成功但码流互不可解」，
因此这里把切分规则固化成测试：

  1. 三个档案都存在，且 resolve_precision 展开正确
  2. mixed 的熵模型子图（prior_init / stage_1..3）必须是 FP32
     —— 这四个决定 CDF 行选择与 y_hat 重建，不逐位一致就会 rANS 失步
  3. mixed 的 encoder / tail 保持 BF16 —— 它们不影响可解性，且是耗时大头
  4. 编解码两端取到同一个档案（同一份代码，不允许各自定义）
  5. 未知精度必须报错，不能静默回退到某个默认值

运行：PYTHONPATH=tools/vendor:python:. python3 tools/tests/test_precision_profiles.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # tools/tests -> GLC
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "python"))

import glc_codec as gc  # noqa: E402

# 决定 CDF 行选择与 y_hat 重建的子图：必须跨芯片逐位一致，故 mixed 下用 FP32
ENTROPY_SUBGRAPHS = ("prior_init", "stage")
# 不影响可解性的子图：留 BF16（耗时大头）
COSMETIC_SUBGRAPHS = ("encoder", "tail")

_failures = []


def check(name, ok, detail=""):
    # detail 只在对账失败时展示，避免 PASS 行出现看似报错的说明
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         (" — " + detail) if (detail and not ok) else ""))
    if not ok:
        _failures.append(name)


def main():
    # 1. 档案存在且可展开
    for p in ("fp32", "bf16", gc.MIXED):
        try:
            profile = gc.resolve_precision(p)
            check("档案 %s 可展开" % p, isinstance(profile, dict),
                  str(sorted(profile)))
        except Exception as exc:  # noqa: BLE001
            check("档案 %s 可展开" % p, False, repr(exc))

    mixed = gc.resolve_precision(gc.MIXED)

    # 2. mixed 的熵模型子图必须 FP32
    for name in ENTROPY_SUBGRAPHS:
        check("mixed: %s 用 FP32" % name, mixed.get(name) == "fp32",
              "实际 %s" % mixed.get(name))

    # 3. mixed 的非关键子图保持 BF16
    for name in COSMETIC_SUBGRAPHS:
        check("mixed: %s 用 BF16" % name, mixed.get(name) == "bf16",
              "实际 %s" % mixed.get(name))

    # 4. 两端必须取到同一档案：glc_encode / glc_decode 都走 gc.resolve_precision
    for mod_name in ("glc_encode", "glc_decode"):
        path = os.path.join(PROJ, "python", mod_name + ".py")
        src = open(path, encoding="utf-8").read()
        check("%s 使用共用档案" % mod_name, "gc.resolve_precision" in src,
              "未调用 gc.resolve_precision")
        check("%s 不再自带档案表" % mod_name, "PRECISION_PROFILES" not in src,
              "模块内重复定义了 PRECISION_PROFILES")

    # 5. 未知精度必须报错
    try:
        gc.resolve_precision("int8")
        check("未知精度报错", False, "int8 被静默接受")
    except ValueError:
        check("未知精度报错", True, "ValueError")

    print()
    if _failures:
        print("失败 %d 项：%s" % (len(_failures), ", ".join(_failures)))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
