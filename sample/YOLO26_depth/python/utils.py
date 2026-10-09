#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os

import cv2
import numpy as np


def is_img(file_name):
    """判断文件是否为可用图像."""
    fmt = os.path.splitext(file_name)[-1]
    return isinstance(fmt, str) and fmt.lower() in ('.jpg', '.png', '.jpeg', '.bmp', '.webp')


def is_video(file_name):
    """判断文件是否为可用视频."""
    fmt = os.path.splitext(file_name)[-1]
    return isinstance(fmt, str) and fmt.lower() in ('.avi', '.mp4', '.mkv', '.mov', '.m4v', '.flv', '.webm')


def depth_colorize(depth, min_depth=0.0, max_depth=None, colormap=cv2.COLORMAP_INFERNO):
    """把单通道深度图着色为 [H, W, 3] 的 BGR 图像。

    Args:
        depth (np.ndarray): [H, W] float32 深度图（单位米）。
        min_depth (float): 归一化下界，低于该值钳到位 0。
        max_depth (float): 归一化上界，默认取有效像素的 95% 分位，保证显色对比度。
        colormap (int): cv2 colormap 常量。
    Returns:
        np.ndarray: [H, W, 3] uint8 BGR 图像。
    """
    d = np.asarray(depth, dtype=np.float32)
    # 无效像素（<=0 或非有限）置 0，不参与分位统计
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    if max_depth is None:
        valid = d[d > 0]
        max_depth = float(np.percentile(valid, 95)) if valid.size else 1.0
    d = np.clip(d, min_depth, max_depth)
    d = (d - min_depth) / max((max_depth - min_depth), 1e-6)
    return cv2.applyColorMap((d * 255).astype(np.uint8), colormap)


def blend_depth(image, depth_vis, alpha=0.5):
    """把深度伪彩色图叠加到原图上。

    Args:
        image (np.ndarray): [H, W, 3] BGR 原图。
        depth_vis (np.ndarray): [H, W, 3] 深度伪彩色图（与 image 同尺寸）。
        alpha (float): 叠加权重。
    Returns:
        np.ndarray: [H, W, 3] uint8 叠加图。
    """
    return cv2.addWeighted(image, 1 - alpha, depth_vis, alpha, 0)