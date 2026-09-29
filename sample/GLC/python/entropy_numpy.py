#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""entropy model 层的 numpy 重写，供设备端（无 torch）使用。

compressai 的 GaussianConditional 在设备端要做的只有三件事：
  build_indexes        —— 64 档 scale_table 上二分/顺序比较，选出 CDF 行号
  quantize(symbols)    —— round(x - mean)
  dequantize           —— round(x - mean) + mean

CDF 表本身只依赖 scale_table（与图像无关），在主机用 compressai 预算一次、
随模型资产下发即可，设备端不必重算，也就绕开了 scipy 依赖。
"""

import math

import numpy as np

SCALE_MIN = 0.11
SCALE_MAX = 256.0
SCALE_LEVELS = 64


def make_scale_table():
    """scale_table 的 numpy 近似。

    **注意：不要用它生成下发的 CDF 资产。** numpy 的 exp/linspace 与 torch 的
    末位不同（64 档里 46 档差 1 ulp），会让 CDF 与客户 compressai 算出的不一致，
    导致码流无法互解。生成端必须用 tools/make_entropy_assets.py（走 torch），
    设备端读固化的 entropy_assets.npz，两端都用同一张表。

    这里保留它只是给纯 numpy 环境做量级估算，不作为权威来源。
    """
    return np.exp(
        np.linspace(math.log(SCALE_MIN), math.log(SCALE_MAX), SCALE_LEVELS)
    ).astype(np.float32)


def build_indexes(scales, scale_table=None):
    """等价于 GaussianConditional.build_indexes。

    官方实现是 indexes = (len-1) - #{s in scale_table[:-1] : scales <= s}，
    即「第一个 >= scales 的档位」，对应 searchsorted(side="left")；
    scale 恰好等于某档位时取该档，不是下一档。超出上限则 clamp 到最后一档。
    """
    if scale_table is None:
        scale_table = make_scale_table()
    idx = np.searchsorted(scale_table, np.asarray(scales), side="left")
    return np.clip(idx, 0, len(scale_table) - 1).astype(np.int32)


def quantize(values, means=None):
    """compressai GaussianConditional.quantize(mode="symbols")：round(x - mean) 取整。"""
    residual = np.asarray(values, dtype=np.float64)
    if means is not None:
        residual = residual - np.asarray(means, dtype=np.float64)
    return np.round(residual).astype(np.int32)


def dequantize(symbols, means=None, dtype=np.float32):
    out = np.asarray(symbols, dtype=dtype)
    if means is not None:
        out = out + np.asarray(means, dtype=dtype)
    return out


def standardized_cumulative(inputs):
    """等价于 compressai 的 0.5 * erfc(-x / sqrt(2))，用 math.erfc 避开 scipy。

    torch.erfc 对 fp32 输入按 fp32 算；这里用 float64 过关再回落，
    精度只多不少，且仅在主机重建 CDF 时用到。
    """
    arr = np.asarray(inputs, dtype=np.float64)
    const = -(2.0 ** -0.5)
    out = np.array([0.5 * math.erfc(const * v) for v in arr.ravel()], dtype=np.float64)
    return out.reshape(arr.shape)


def pack_fixed_width(values, bit_width):
    """定长位打包，与客户脚本 test_image_complement2.py 的实现逐位一致。

    逐值大端累积，满 8 bit 落 1 字节，末尾不足一字节左对齐补零。
    """
    limit = 1 << bit_width
    accumulator = 0
    buffered_bits = 0
    output = bytearray()
    for value in values:
        value = int(value)
        if not 0 <= value < limit:
            raise ValueError(f"值 {value} 无法用 {bit_width} bit 表示")
        accumulator = (accumulator << bit_width) | value
        buffered_bits += bit_width
        while buffered_bits >= 8:
            buffered_bits -= 8
            output.append((accumulator >> buffered_bits) & 0xFF)
            accumulator &= (1 << buffered_bits) - 1 if buffered_bits else 0
    if buffered_bits:
        output.append((accumulator << (8 - buffered_bits)) & 0xFF)
    return bytes(output)


def unpack_fixed_width(payload, count, bit_width):
    values = []
    accumulator = 0
    buffered_bits = 0
    for byte in payload:
        accumulator = (accumulator << 8) | byte
        buffered_bits += 8
        while buffered_bits >= bit_width and len(values) < count:
            buffered_bits -= bit_width
            values.append((accumulator >> buffered_bits) & ((1 << bit_width) - 1))
            accumulator &= (1 << buffered_bits) - 1 if buffered_bits else 0
    if len(values) != count:
        raise ValueError(f"定长码流被截断：期望 {count} 个，实得 {len(values)} 个")
    return values


def four_part_masks(channels, height, width):
    """四段先验掩码，等价于官方 common_model.get_mask_four_parts。

    官方实现（common_model.py:163）先把通道按 4 等分，每份配一个不同相位的
    micro mask，再拼接；这里直接用 numpy 生成同样形状的布尔掩码。

    返回 4 个 (1, C, H, W) 的布尔数组，各选中 C*H*W/4 个元素。
    掩码必须与 torch 版完全一致，否则两端逐段参数不同、码流解不出来。
    """
    if channels % 4 != 0:
        raise ValueError(f"通道数 {channels} 不是 4 的倍数")
    rows = np.arange(height)[:, None] % 2
    cols = np.arange(width)[None, :] % 2
    # 四个相位 (r, c)，顺序对应官方的 m0/m1/m2/m3
    micro = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # 四组通道各自的相位顺序，逐行抄自 common_model.py:170-174 的拼接式子。
    # 不是简单循环移位：mask_1 是 (m3,m2,m1,m0)、mask_2 是 (m2,m3,m0,m1)、
    # mask_3 是 (m1,m0,m3,m2)，照抄才不会与 torch 版错位。
    orders = [
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [2, 3, 0, 1],
        [1, 0, 3, 2],
    ]
    c4 = channels // 4
    masks = []
    for s in range(4):
        m = np.zeros((1, channels, height, width), dtype=bool)
        for g in range(4):
            r, c = micro[orders[s][g]]
            m[0, g * c4:(g + 1) * c4] = (rows == r) & (cols == c)
        masks.append(m)
    return masks


def compact_mask(tensor, mask):
    """按四段掩码取出该段的 1/4 元素，压成 (1, C, H, W//4)。

    numpy 版，对应 glc_codec.compact_masked 的 torch 实现。
    掩码形状 (1, C, H, W)，tensor 形状 (1, C, H, W)。
    """
    t = np.asarray(tensor)
    if t.ndim == 3:
        t = t[None]
    b, c, h, w = t.shape
    if b != 1:
        raise ValueError("码流编解码目前只支持 batch size 1")
    selected = t[np.broadcast_to(mask, t.shape)]
    expect = c * h * w // 4
    if selected.size != expect:
        raise ValueError(f"掩码选中 {selected.size} 个元素，预期 {expect}")
    return selected.reshape(1, c, h, w // 4)


def scatter_mask(compact, mask, shape):
    """compact_mask 的逆操作：把该段元素放回原张量。"""
    out = np.zeros(shape, dtype=np.asarray(compact).dtype)
    out[np.broadcast_to(mask, shape)] = np.asarray(compact).reshape(-1)
    return out
