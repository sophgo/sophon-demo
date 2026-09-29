#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""ctypes 绑定：把 rans/glc_rans.c 的纯 C ABI 包成 Python 可调用接口。

设备端（SE7 / aarch64）与主机端共用同一份 C 源码，只是 .so 由不同工具链编出。
调用方只需要 numpy：不依赖 compressai，也不依赖 torch。
"""

import ctypes
import os

import numpy as np

_P32 = ctypes.POINTER(ctypes.c_int32)
_PU8 = ctypes.POINTER(ctypes.c_uint8)
_PSZ = ctypes.POINTER(ctypes.c_size_t)


def default_lib_name():
    import platform

    machine = platform.machine()
    if machine in ("aarch64", "arm64"):
        return "libglc_rans_aarch64.so"
    return "libglc_rans_x86_64.so"


def _find_lib():
    """在几个可能的位置找编好的 .so。

    C 源码在 <GLC>/tools/rans/，本模块在 <GLC>/python/，因此要往上一级找。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    name = default_lib_name()
    for d in (os.path.join(here, "..", "tools", "rans"),
              os.path.join(here, "rans"),
              os.path.join(here, "..", "rans"),
              here):
        cand = os.path.normpath(os.path.join(d, name))
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"未找到 {name}，先跑 tools/rans/build.sh 编译"
        "（它同时产出 x86_64 与 aarch64 两个版本）"
    )


class RansCoder:
    """rANS 编解码器，语义与 compressai 的 BufferedRansEncoder/RansDecoder 一致。

    cdf_len / cdf_off 必须是 CDF 的**有效长度与行起始偏移**，不是填充后的行宽：
    compressai 的 _quantized_cdf 每行留了 2 项填充，调用方需先用 _cdf_length 裁掉。
    """

    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_path = _find_lib()
        self.lib = ctypes.CDLL(lib_path)
        self.lib.glc_rans_encode.restype = _PU8
        self.lib.glc_rans_encode.argtypes = [
            _P32, _P32, ctypes.c_size_t, _P32, _P32, _P32, _P32, ctypes.c_size_t, _PSZ,
        ]
        self.lib.glc_rans_decode.restype = _P32
        self.lib.glc_rans_decode.argtypes = [
            _PU8, ctypes.c_size_t, ctypes.c_size_t, _P32, _P32, _P32, _P32, _P32,
            ctypes.c_size_t,
        ]
        self.lib.glc_rans_free.argtypes = [ctypes.c_void_p]
        self.lib.glc_int32_free.argtypes = [_P32]

    @staticmethod
    def _i32(arr):
        return np.ascontiguousarray(arr, dtype=np.int32)

    def encode(self, symbols, indexes, cdf, cdf_len, cdf_off, offsets):
        """返回 bytes。symbols/indexes 等长；cdf 为扁平 int32 CDF 表。"""
        sb = self._i32(symbols)
        ib = self._i32(indexes)
        cb = self._i32(cdf)
        lb = self._i32(cdf_len)
        ob = self._i32(cdf_off)
        fb = self._i32(offsets)
        n = sb.size
        if n == 0:
            return b""
        out_len = ctypes.c_size_t(0)
        ptr = self.lib.glc_rans_encode(
            sb.ctypes.data_as(_P32), ib.ctypes.data_as(_P32), n,
            cb.ctypes.data_as(_P32), lb.ctypes.data_as(_P32), ob.ctypes.data_as(_P32),
            fb.ctypes.data_as(_P32), lb.size, ctypes.byref(out_len),
        )
        if not ptr:
            raise RuntimeError("glc_rans_encode 失败：入参越界或内存不足")
        try:
            return bytes(bytearray(ptr[: out_len.value]))
        finally:
            self.lib.glc_rans_free(ctypes.cast(ptr, ctypes.c_void_p))

    def decode(self, payload, n, indexes, cdf, cdf_len, cdf_off, offsets):
        """返回 int32 numpy 数组（长度 n），已补回 offset。"""
        buf = bytes(payload)
        ib = self._i32(indexes)
        cb = self._i32(cdf)
        lb = self._i32(cdf_len)
        ob = self._i32(cdf_off)
        fb = self._i32(offsets)
        if ib.size != n:
            raise ValueError(f"indexes 长度 {ib.size} 与符号数 {n} 不符")
        cbuf = ctypes.create_string_buffer(buf, len(buf)) if buf else ctypes.create_string_buffer(1)
        ptr = self.lib.glc_rans_decode(
            ctypes.cast(cbuf, _PU8), len(buf), n,
            ib.ctypes.data_as(_P32), cb.ctypes.data_as(_P32), lb.ctypes.data_as(_P32),
            ob.ctypes.data_as(_P32), fb.ctypes.data_as(_P32), lb.size,
        )
        if not ptr:
            raise RuntimeError("glc_rans_decode 失败：入参越界或内存不足")
        try:
            return np.ctypeslib.as_array(ptr, shape=(n,)).copy()
        finally:
            self.lib.glc_int32_free(ptr)
