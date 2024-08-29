# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
import logging

import cv2
import numpy as np

rng = np.random.default_rng(2)
colors = rng.uniform(100, 255, size=(80, 3))

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def create_mask_from_polygons(polygons, image_shape):
    # 创建一个空白的mask
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    for polygon in polygons:
        # polygon 需要转换为 int32 类型的 numpy 数组
        np_polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [np_polygon], 1)
    return mask


def scale_segmentation(segmentation, scale_x, scale_y):
    # 缩放segmentation坐标
    scaled_segmentation = []
    for seg in segmentation:
        scaled_seg = np.array(seg).reshape((-1, 2))
        scaled_seg[:, 0] = (scaled_seg[:, 0] * scale_x).astype(np.int64)  # 缩放x坐标
        scaled_seg[:, 1] = (scaled_seg[:, 0] * scale_y).astype(np.int64)  # 缩放y坐标
        scaled_segmentation.append(scaled_seg.flatten().tolist())
    return scaled_segmentation


def draw_masks(
    image: np.ndarray, masks, alpha: float = 0.5, draw_border: bool = True
) -> np.ndarray:
    if image is None:
        return np.empty(0)
    mask_image = image.copy()
    for label_id, label_masks in enumerate(masks):
        # 随机数组的第一项固定为红色，在truck上标注不明显，故加了个10
        color = colors[label_id + 50]
        mask_image = draw_mask(
            mask_image, label_masks, (color[0], color[1], color[2]), alpha, draw_border
        )
    return mask_image


def draw_mask(
    image: np.ndarray,
    masks: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.5,
    draw_border: bool = True,
) -> np.ndarray:

    mask = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
    for m in masks[0, :, :, :]:
        mask[m > 0.5] = color
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    visualized = cv2.addWeighted(image, 1, mask, 0.5, 0)
    return visualized


def bytes_to_megabytes(bytes_size):
    return bytes_size / (1024 * 1024)


def simplify_contours(contours, epsilon=1.0):
    simplified = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified.append(approx)
    return simplified


def mask_to_coco_segmentation(mask, shape=None, simplify=True):
    # 删除1的维度，并进行二值化
    mask = mask.squeeze()
    binary_mask = (mask > 0.5).astype(np.uint8)

    if shape != binary_mask.shape:
        binary_mask = cv2.resize(binary_mask, (shape[1], shape[0]))

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if simplify:
        contours = simplify_contours(contours)

    segmentation = []
    for contour in contours:
        # 将轮廓坐标扁平化并转换为列表
        contour = contour.flatten().tolist()
        segmentation.append(contour)
    return segmentation
