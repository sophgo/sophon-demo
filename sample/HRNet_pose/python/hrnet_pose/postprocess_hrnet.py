#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import math
import cv2
import numpy as np
from .utils_hrnet import NUM_KPTS, SKELETON, CocoColors
from .preprocess_hrnet import get_affine_transform

import logging
logging.basicConfig(level=logging.INFO)

def flip_images(img):
    assert len(img.shape) == 4, 'images has to be [batch_size, channels, height, width]'
    img = np.flip(img, axis=3)
    return img


def flip_back(output_flipped, matched_parts):
    assert len(output_flipped.shape) == 4, 'output_flipped has to be [batch_size, num_joints, height, width]'

    output_flipped = np.flip(output_flipped, axis=3)

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0]].copy()
        output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
        output_flipped[:, pair[1]] = tmp

    return output_flipped


def draw_pose(keypoints, img):

    assert keypoints.shape == (NUM_KPTS, 2)

    for i in range(len(SKELETON)):

        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]

        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]

        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)

        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def draw_bbox(box, img):

    x1, y1, x2, y2 = box
    box = [(int(x1), int(y1)), (int(x2), int(y2))]
    # box = list(zip(*[iter(box)] * 2))
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)

def get_max_preds(batch_heatmaps):

    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)

    maxvals = np.amax(heatmaps_reshaped, axis=2)
    idx = np.argmax(heatmaps_reshaped, axis=2)
    idx = idx.astype(np.float32)

    preds = np.zeros((batch_size, num_joints, 2))

    preds[:, :, 0] = idx % w
    preds[:, :, 1] = np.floor(idx / w)

    pred_mask = (maxvals > 0.0).reshape(batch_size, num_joints, 1)
    pred_mask = np.repeat(pred_mask, 2, axis=2)

    preds *= pred_mask

    return preds, maxvals


def transform_preds(coords, box):
    target_coords = np.zeros(coords.shape)

    trans = get_affine_transform(box, inv=0)

    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_points(coords[p, 0:2], trans)

    return target_coords


def affine_points(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T

    new_pt = np.dot(t, new_pt)

    return new_pt[:2]

def postprocess(batch_heatmaps: np.ndarray,
                    box: list = None):

    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array(
                    [
                        hm[py][px + 1] - hm[py][px - 1],
                        hm[py + 1][px] - hm[py - 1][px]
                    ]
                )
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], box)

    return preds, maxvals