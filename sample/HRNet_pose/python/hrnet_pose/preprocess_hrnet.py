#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import cv2
import numpy as np
import random
import math

import logging
logging.basicConfig(level=logging.INFO)

from typing import Tuple

def adjust_box(xmin: float, ymin: float, w: float, h: float, fixed_size: Tuple[float, float]):

    xmax = xmin + w
    ymax = ymin + h

    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:
        # 需要在w方向padding
        wi = h / hw_ratio
        pad_w = (wi - w) / 2
        xmin = xmin - pad_w
        xmax = xmax + pad_w
    else:
        # 需要在h方向padding
        hi = w * hw_ratio
        pad_h = (hi - h) / 2
        ymin = ymin - pad_h
        ymax = ymax + pad_h

    return xmin, ymin, xmax, ymax


# 获取仿射变换矩阵
def get_affine_transform(box, scale=None, rotation=None, fixed_size=(256, 192), inv=1):

    xmin, ymin, xmax, ymax = box
    w = xmax - xmin
    h = ymax - ymin
    box = [xmin, ymin, w, h]

    src_xmin, src_ymin, src_xmax, src_ymax = adjust_box(*box, fixed_size=fixed_size)
    src_w = src_xmax - src_xmin
    src_h = src_ymax - src_ymin
    src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
    src_p2 = src_center + np.array([0, -src_h / 2])
    src_p3 = src_center + np.array([src_w / 2, 0])

    dst_center = np.array([(fixed_size[1] - 1) / 2, (fixed_size[0] - 1) / 2])
    dst_p2 = np.array([(fixed_size[1] - 1) / 2, 0])
    dst_p3 = np.array([fixed_size[1] - 1, (fixed_size[0] - 1) / 2])

    if scale is not None:
        scale = random.uniform(*scale)
        src_w = src_w * scale
        src_h = src_h * scale
        src_p2 = src_center + np.array([0, -src_h / 2])
        src_p3 = src_center + np.array([src_w / 2, 0])

    if rotation is not None:
        angle = random.randint(*rotation)
        angle = angle / 180 * math.pi
        src_p2 = src_center + np.array([src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)])
        src_p3 = src_center + np.array([src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)])

    src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
    dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

    if inv:
        trans = cv2.getAffineTransform(src, dst)
    else:
        dst /= 4
        trans = cv2.getAffineTransform(dst, src)

    return trans


def normalize_image(image, mean=None, std=None):

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    if len(mean) != image.shape[2] or len(std) != image.shape[2]:
        raise ValueError("Mean and std must match the number of channels in the image")

    for i in range(image.shape[2]):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]

    return image


def preprocess(image, box, pose_resize_hw):

    trans = get_affine_transform(box, scale=None, rotation=None, fixed_size=pose_resize_hw)

    image = cv2.warpAffine(image, trans, tuple(pose_resize_hw[::-1]), flags=cv2.INTER_LINEAR)

    image = np.ascontiguousarray(image / 255.0)

    image = normalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    image = image.transpose((2, 0, 1))

    return image