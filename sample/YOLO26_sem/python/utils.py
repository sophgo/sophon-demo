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

# Cityscapes 19 类
CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')

# Cityscapes 官方调色板 (BGR 顺序，与官方约定一致)
PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)

# Cityscapes gtFine labelIds -> trainIds 映射，index 为原始 labelId
# -1 及 255 表示 ignore_label
CITYSCAPES_LABEL_MAP = [
    255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 255, 255, 255,
    5, 255, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 16, 17, 18,
]


def is_img(file_name):
    """判断文件是否为可用图像."""
    fmt = os.path.splitext(file_name)[-1]
    return isinstance(fmt, str) and fmt.lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.webp']


def palette_map(class_map, palette=None):
    """把 [H, W] 的类别图着色为 [H, W, 3] 的 BGR 图像.

    Args:
        class_map (np.ndarray): [H, W] uint8，像素值为类别 id（0..nc-1），255 视为忽略（黑色）。
        palette (np.ndarray): [nc, 3] BGR 调色板，默认 Cityscapes。
    Returns:
        np.ndarray: [H, W, 3] uint8 BGR 图像。
    """
    if palette is None:
        palette = PALETTE
    color = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)
    for label, c in enumerate(palette):
        color[class_map == label] = c
    # 忽略像素填黑色
    color[class_map == 255] = (0, 0, 0)
    return color


def blend_seg(image, class_map, palette=None, alpha=0.5):
    """把着色后的分割结果叠加到原图上.

    Args:
        image (np.ndarray): [H, W, 3] BGR 原图。
        class_map (np.ndarray): [H, W] uint8 类别图。
        palette (np.ndarray): [nc, 3] BGR 调色板。
        alpha (float): 叠加权重。
    Returns:
        np.ndarray: [H, W, 3] uint8 叠加图。
    """
    color_seg = palette_map(class_map, palette)
    res = cv2.addWeighted(image, 1 - alpha, color_seg, alpha, 0)
    return res


def label_ids_to_train_ids(label_gt, label_map=CITYSCAPES_LABEL_MAP):
    """把 Cityscapes gtFine labelIds 标签图转换为 trainIds.

    Args:
        label_gt (np.ndarray): [H, W] uint8/long，原始 labelId。
        label_map (list): 长度 34 的映射表。
    Returns:
        np.ndarray: [H, W] uint8，trainId，忽略像素为 255。
    """
    lut = np.array(label_map, dtype=np.uint8)
    # 负值与 >=34 的值统一视为忽略
    out = np.full_like(label_gt, 255, dtype=np.uint8)
    mask = (label_gt >= 0) & (label_gt < len(label_map))
    out[mask] = lut[label_gt[mask]]
    return out