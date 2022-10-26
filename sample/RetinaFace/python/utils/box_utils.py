#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# -*- coding: utf-8 -*-
"""
Function: Box Utils
Author: jin.zhang
Create Time: 2022/01/18 13:48
"""

# import torch
from itertools import product as product
import numpy as np
from math import sqrt as sqrt
from math import ceil
import random
import time
import cv2

from loguru import logger

def draw_one_on_bmimage(bmcv, box, landmark, img, color = None, label = None, line_thickness = None):

    tl = (
            line_thickness or round(0.001 * (img.width() + img.height()) / 2) + 1
    )  # line/font thickness

    color = color or [random.randint(0, 255) for _ in range(3)]

    left = int(box[0])
    top = int(box[1])
    width = int(box[2] - box[0])
    height =  int(box[3] - box[1])
    
    bmcv.rectangle(img, left, top, width, height, color, tl)

# see https://github.com/wang-xinyu/tensorrtx/blob/master/retinaface/retinaface_trt.py
def draw_one(box, landmark, img, color = None, label = None, line_thickness = None):
    """
    description: Plots one bounding box and face landmark on cv2.mat img

    param:
        box:     a box likes [x1,y1,x2,y2,conf]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
            line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness

    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    cv2.circle(img, (int(landmark[0]), int(landmark[1])), 1, (0, 0, 255), 4)
    cv2.circle(img, (int(landmark[2]), int(landmark[3])), 1, (0, 255, 255), 4)
    cv2.circle(img, (int(landmark[4]), int(landmark[5])), 1, (255, 0, 255), 4)
    cv2.circle(img, (int(landmark[6]), int(landmark[7])), 1, (0, 255, 0), 4)
    cv2.circle(img, (int(landmark[8]), int(landmark[9])), 1, (255, 0, 0), 4)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def py_cpu_nms(dets, thresh):  
    """Pure Python NMS baseline."""  
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 4]  #bbox打分
  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    #打分从大到小排列，取index  
    order = scores.argsort()[::-1]  

    #keep为最后保留的边框  
    keep = []  
    while order.size > 0:  
    #order[0]是当前分数最大的窗口，肯定保留  
        i = order[0]  
        keep.append(i)  
        #计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        #交/并得到iou值  
        ovr = inter / (areas[i] + areas[order[1:]] - inter) 

        #inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收  
        inds = np.where(ovr <= thresh)[0]  
        #order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]  
  
    return keep

# see https://github.com/biubug6/Pytorch_Retinaface/blob/master/layers/functions/prior_box.py
class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [box * self.steps[k] / self.image_size[1] for box in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        # output = torch.Tensor(anchors).view(-1, 4) # for torch
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            # output.clamp_(max=1, min=0) # for torch
            np.clip(output, min=0, max=1)
        return output

# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/utils/box_utils.py
def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), 1)
    return landms
 
 # Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/utils/box_utils.py
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes