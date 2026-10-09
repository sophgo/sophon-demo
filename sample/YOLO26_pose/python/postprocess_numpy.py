#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import numpy as np

# [x1, y1, x2, y2, score, class_id] 之后紧跟 17 * 3 个关键点
BOX_LEN = 6
KPT_NUM = 17


class PostProcess:

    def __init__(self, conf_thres=0.25):
        self.conf_threshold = conf_thres

    def __call__(self, outputs, im0_shapes, ratios, txys):
        results = []
        for i in range(outputs[0].shape[0]):
            results.append(self.postprocess(outputs[0][i][np.newaxis, :],
                                            im0_shapes[i], ratios[i],
                                            txys[i][0], txys[i][1]))
        return results

    def postprocess(self, preds, im0_shape, ratio, pad_w, pad_h):
        """post process

        Args:
            preds (numpy.ndarray): [batch, 300, 6 + 17 * 3], the end2end output of
                YOLO26-pose, already sorted by score, no NMS needed.
                Each row is [x1, y1, x2, y2, score, class_id, kpts...].
            im0_shape (tuple): (h, w) of the original image
            ratio (tuple): width, height ratios in letterbox
            pad_w (float): width padding in letterbox
            pad_h (float): height padding in letterbox

        Returns:
            (numpy.ndarray): [n, 6 + 17 * 3] in the original image coordinates
        """
        pred = preds[0]

        # Predictions filtering by conf-threshold
        pred = pred[pred[:, 4] > self.conf_threshold]

        if len(pred) == 0:
            return np.zeros((0, BOX_LEN + KPT_NUM * 3))

        # Rescale bounding boxes from model shape to the shape of original image
        pred[:, :4] -= [pad_w, pad_h, pad_w, pad_h]
        pred[:, :4] /= min(ratio)

        # Bounding boxes boundary clamp
        pred[:, [0, 2]] = pred[:, [0, 2]].clip(0, im0_shape[1] - 1)
        pred[:, [1, 3]] = pred[:, [1, 3]].clip(0, im0_shape[0] - 1)

        # Rescale keypoints from model shape to the shape of original image
        for k in range(KPT_NUM):
            pred[:, BOX_LEN + k * 3] = ((pred[:, BOX_LEN + k * 3] - pad_w) / ratio[0]).clip(0, im0_shape[1] - 1)
            pred[:, BOX_LEN + k * 3 + 1] = ((pred[:, BOX_LEN + k * 3 + 1] - pad_h) / ratio[1]).clip(0, im0_shape[0] - 1)

        return pred
