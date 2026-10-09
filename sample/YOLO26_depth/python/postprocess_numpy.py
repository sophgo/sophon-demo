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


class PostProcess:

    def __init__(self, net_w=768, net_h=768):
        self.net_w = net_w
        self.net_h = net_h

    def __call__(self, outputs, im0_shapes, ratios=None):
        """后处理入口.

        bmodel 直接输出 [batch, 1, net_h, net_w] float32 度量深度图（米），与官方
        Depth head 导出口径一致（4x bilinear 上采样 + cal_a/cal_b 标定已烘焙进图）。
        ultralytics 官方 val 协议把 GT 深度也 resize 到 768x768（INTER_NEAREST）后
        逐像素比对，因此这里原样返回 net 分辨率深度图，不进原图尺寸缩放。

        Args:
            outputs (list[np.ndarray]): 模型输出，outputs[0] 为深度图 [batch, 1, net_h, net_w]。
            im0_shapes (list[(h, w)]): 每张原图尺寸（此处仅占位，保持接口一致）。
            ratios (list | None): 保留参数，stretch 预处理无 letterbox 缩放比。
        Returns:
            list[np.ndarray]: 每张图对应的 [net_h, net_w] float32 深度图。
        """
        results = []
        batch = outputs[0].shape[0]
        for i in range(batch):
            depth = outputs[0][i]
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            results.append(self.postprocess(depth))
        return results

    def postprocess(self, depth):
        """单图后处理：提取 [net_h, net_w] 浮点深度图.

        Args:
            depth (np.ndarray): [net_h, net_w] 或 [1, net_h, net_w] float32 深度图。
        Returns:
            np.ndarray: [net_h, net_w] float32 深度图（米）。
        """
        return np.ascontiguousarray(depth).astype(np.float32)