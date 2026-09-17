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

    def __init__(self, net_w=2048, net_h=1024):
        self.net_w = net_w
        self.net_h = net_h

    def __call__(self, outputs, im0_shapes, ratios):
        """后处理入口.

        Args:
            outputs (list[np.ndarray]): 模型输出，outputs[0] 为类别图 [batch, 1, net_h, net_w]。
            im0_shapes (list[(h, w)]): 每张原图尺寸。
            ratios (list[(rw, rh)]): 每张图的 letterbox 缩放比。
        Returns:
            list[np.ndarray]: 每张图对应的 [h, w] uint8 类别图。
        """
        results = []
        batch = outputs[0].shape[0]
        for i in range(batch):
            class_map = outputs[0][i]  # [1, net_h, net_w] 或 [net_h, net_w]
            if class_map.ndim == 3 and class_map.shape[0] == 1:
                class_map = class_map[0]
            results.append(self.postprocess(class_map, im0_shapes[i], ratios[i]))
        return results

    def postprocess(self, class_map, im0_shape, ratio):
        """单图后处理：去 letterbox padding + 缩回原图.

        bilinear 上采样 + argmax 已烘焙进 bmodel（与 ultralytics 后处理顺序一致：
        先对 logits 做 8x bilinear 上采样再 argmax），bmodel 直接输出 net 分辨率的
        类别图，因此这里只需裁掉 letterbox padding 并最近邻缩回原图。

        Args:
            class_map (np.ndarray): [net_h, net_w] 类别图（int32/int64，值为类别 id）。
            im0_shape (tuple): (h, w) 原图尺寸。
            ratio (tuple): (rw, rh) letterbox 缩放比。
        Returns:
            np.ndarray: [h, w] uint8 类别图。
        """
        class_map = class_map.astype(np.uint8)
        orig_h, orig_w = im0_shape[0], im0_shape[1]
        r = ratio[0]
        scaled_w = int(round(r * orig_w))
        scaled_h = int(round(r * orig_h))
        x1 = (self.net_w - scaled_w) // 2
        y1 = (self.net_h - scaled_h) // 2
        crop = class_map[y1:y1 + scaled_h, x1:x1 + scaled_w]

        # 缩回原图尺寸
        class_map = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return class_map