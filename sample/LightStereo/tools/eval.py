#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import argparse
import os
import cv2

def d1_metric(disp_pred, disp_gt, mask):
    E = np.abs(disp_gt - disp_pred)
    err_mask = (E > 3) & (E / np.abs(disp_gt) > 0.05)

    err_mask = err_mask & mask
    num_errors = np.sum(err_mask, axis=(1, 2))
    num_valid_pixels = np.sum(mask, axis=(1, 2))

    d1_per_image = num_errors.astype(np.float32) / num_valid_pixels.astype(np.float32)
    d1_per_image = np.where(num_valid_pixels > 0, d1_per_image, np.zeros_like(d1_per_image))

    return d1_per_image

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--results_path', type=str, default='../python/results/images', help='path of results disparity images')
    parser.add_argument('--gt_path', type=str, default='../datasets/KITTI12/training/disp_occ', help='path of grounding truth')
    parser.add_argument('--MAX_DISP', type=int, default=192, help='maximum disparity')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    res_file_list = []
    gt_file_list = []
    if os.path.isdir(args.results_path):
        for root, dirs, filenames in os.walk(args.results_path):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                res_img_file = os.path.join(root, filename)
                gt_img_file = os.path.join(args.gt_path, filename)
                if os.path.exists(res_img_file):
                    res_file_list.append(res_img_file)
                    gt_file_list.append(gt_img_file)
                else:
                    print("cannot find gt_img_file for comparison: ", gt_img_file)
    d1_list = []
    for i in range(len(res_file_list)):
        res_img = cv2.imread(res_file_list[i])
        gt_img = cv2.imread(gt_file_list[i])
        if res_img.shape != gt_img.shape:
            print("{}'s resolution is not the same with gt img. {} vs {}".format(res_file_list[i], res_img.shape, gt_img.shape))
            continue
        mask = (gt_img < args.MAX_DISP) & (gt_img > 0)
        d1 = d1_metric(res_img, gt_img, mask)
        d1_list.extend(d1.tolist())
    print("avg_d1: ", np.array(d1_list).mean())