#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""图像质量指标（纯 numpy 实现）。

主机基线与 SE7 设备端共用同一份实现，避免两侧指标口径不一致。
SSIM 采用与 skimage 一致的 11x11 高斯窗（sigma=1.5）、无偏方差归一化。
"""
import numpy as np


def _gaussian_kernel(size=11, sigma=1.5):
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return g


def _filter_sep(img, kernel):
    """可分离高斯滤波（输入为 2D 单通道），边界用 'reflect'（同 skimage 默认 mode）。"""
    r = len(kernel) // 2
    pad = np.pad(img, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    for i, k in enumerate(kernel):
        out += k * pad[i:i + img.shape[0], :]
    pad = np.pad(out, ((0, 0), (r, r)), mode="reflect")
    out2 = np.zeros_like(img, dtype=np.float64)
    for i, k in enumerate(kernel):
        out2 += k * pad[:, i:i + img.shape[1]]
    return out2


def psnr(img1, img2, data_range=255.0):
    """img1/img2: uint8 或 float 数组，形状 (H,W,3) 或 (H,W)。"""
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim(img1, img2, data_range=255.0):
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
        b = b[:, :, None]

    kernel = _gaussian_kernel()
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    vals = []
    for c in range(a.shape[2]):
        x = a[:, :, c]
        y = b[:, :, c]
        ux = _filter_sep(x, kernel)
        uy = _filter_sep(y, kernel)
        uxx = _filter_sep(x * x, kernel)
        uyy = _filter_sep(y * y, kernel)
        uxy = _filter_sep(x * y, kernel)

        # 无偏方差（同 skimage：用 N/(N-1) 修正）
        npix = kernel.size ** 2
        cov_norm = npix / (npix - 1.0)
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        # 只在有效区域统计，避免边界 reflect 造成的偏差
        r = len(kernel) // 2
        sl = (slice(r, -r), slice(r, -r))
        vx, vy, vxy = vx[sl], vy[sl], vxy[sl]
        ux, uy = ux[sl], uy[sl]

        s = ((2 * ux * uy + C1) * (2 * vxy + C2)) / \
            ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))
        vals.append(s.mean())
    return float(np.mean(vals))


def evaluate(rec_uint8, ref_uint8):
    """rec/ref 均为 uint8 HWC，返回 (psnr, ssim)。"""
    return psnr(rec_uint8, ref_uint8), ssim(rec_uint8, ref_uint8)
