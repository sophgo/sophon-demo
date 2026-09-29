#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""码流协议边界检查：头部字段、截断、篡改都必须被明确拒绝。

解码端只吃 bytes，而 bytes 可能来自无线信道——截断或损坏是常态而非异常。
坏码流必须报错，不能静默产出错误图像。

只用到 glc_codec 的协议层，不需要权重，也不需要模型。

运行：PYTHONPATH=vendor:. python3 tests/test_bitstream_protocol.py
"""

import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # tools/tests -> GLC
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "python"))

import glc_codec  # noqa: E402

_failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def build_valid(y_bytes=(10, 11, 12, 13), z_bytes=4, q_index=2, dims=None):
    """按协议手工拼一个合法码流，避开对模型的依赖。"""
    if dims is None:
        dims = (512, 768, 512, 768, 256, 32, 48, 256, 8, 12, 96)
    z_payload = bytes(range(z_bytes))
    streams = [bytes((i,) * n) for i, n in enumerate(y_bytes)]
    header = glc_codec.HEADER.pack(
        glc_codec.MAGIC, glc_codec.VERSION, q_index, 0, *dims,
        len(z_payload), *(len(s) for s in streams), 0,
    )
    return header + z_payload + b"".join(streams), dims, streams


def expect_reject(name, payload):
    try:
        glc_codec.parse_bitstream(payload)
        check(name, False, "本应报错却成功解析")
    except (ValueError, struct.error) as exc:
        check(name, True, f"{type(exc).__name__}: {str(exc)[:50]}")


def main():
    check("头部长度固定 76 字节", glc_codec.HEADER.size == 76,
          f"{glc_codec.HEADER.size}")

    payload, dims, streams = build_valid()
    q_index, parsed_dims, parsed_streams = glc_codec.parse_bitstream(payload)
    check("合法码流解析出正确的 q_index", q_index == 2, f"q_index={q_index}")
    check("合法码流解析出全部维度", tuple(parsed_dims) == dims)
    check("合法码流解析出 1 段 z + 4 段 y", len(parsed_streams) == 5,
          f"{[len(s) for s in parsed_streams]}")

    # 逐项破坏，每一种都必须被拒绝
    expect_reject("魔数错误应拒绝", b"XXXX" + payload[4:])
    expect_reject("版本号错误应拒绝", payload[:4] + b"\x09" + payload[5:])
    expect_reject("截断的头部应拒绝", payload[:40])
    expect_reject("载荷截断应拒绝", payload[:-5])

    # 头部布局：4s(magic) B(ver) B(q_index) H(flags) + 17 个 I
    # flags 在偏移 6..7；保留字段是最后一个 I，偏移 72..75
    bad_flags = bytearray(payload)
    bad_flags[6:8] = struct.pack(">H", 1)
    expect_reject("flags 非 0 应拒绝", bytes(bad_flags))

    bad_reserved = bytearray(payload)
    off = 8 + 16 * 4
    bad_reserved[off:off + 4] = struct.pack(">I", 7)
    expect_reject("保留字段非 0 应拒绝", bytes(bad_reserved))

    expect_reject("尾部多余数据应拒绝", payload + b"\x00")

    # 长度为 0 的段：rANS 码流不可能是 0 长度。
    # 17 个 I = 11 个维度 + 5 个码流长度 + 1 个保留字段
    header0 = glc_codec.HEADER.pack(
        glc_codec.MAGIC, glc_codec.VERSION, 0, 0, *dims, 4,
        0, 10, 11, 12, 0)
    expect_reject("长度为 0 的码流段应拒绝", header0 + bytes(4))

    # bpp 分项应与实际字节数一致
    bd = glc_codec.bpp_breakdown(payload, 512, 768)
    px = 512 * 768
    check("bpp 分项加总等于总 bpp",
          abs(bd["y"] + bd["z"] + bd["header"] - bd["total"]) < 1e-12,
          f"{bd['total']:.6f}")
    check("header bpp 与 76 字节相符",
          abs(bd["header"] - 76 * 8 / px) < 1e-12, f"{bd['header']:.6f}")
    check("y_bytes 为四段", len(bd["y_bytes"]) == 4, f"{bd['y_bytes']}")

    print()
    if _failures:
        print(f"失败 {len(_failures)} 项: {_failures}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
