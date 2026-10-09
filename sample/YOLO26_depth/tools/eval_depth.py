#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""NYU Depth V2 (Eigen split) 单目深度精度评测。

复刻 ultralytics DepthMetrics 口径：
- 逐图 median 对齐（Depth Anything 协议）：scale = median(gt) / median(pred.clamp(min_depth))
- Eigen 协议：只统计 gt ∈ (min_depth, max_depth) 的像素，pred 钳位到该区间
- 全 val 集有效像素池化累加（非逐图平均）
- 指标：delta1/2/3、abs_rel、rmse、silog
"""
import os
import glob
import argparse
import logging
logging.basicConfig(level=logging.INFO)

import cv2
import numpy as np


class DepthMetrics:
    """与 ultralytics utils/metrics.py DepthMetrics 数值对齐的 numpy 实现."""

    def __init__(self, min_depth=0.001, max_depth=100.0, align="median"):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.align = align
        self._totals = np.zeros(7, dtype=np.float64)
        self._count = 0.0

    def update_stats(self, pred, gt):
        p = np.asarray(pred, dtype=np.float64).squeeze()
        g = np.asarray(gt, dtype=np.float64).squeeze()
        mask = (g > self.min_depth) & (g < self.max_depth) & np.isfinite(p)
        n = int(mask.sum())
        if n == 0:
            return
        pv = p[mask]
        gv = g[mask]
        if self.align == "median":
            scale = np.median(gv) / np.median(np.clip(pv, self.min_depth, None))
            pv = pv * scale
        pv = np.clip(pv, self.min_depth, self.max_depth)
        thresh = np.maximum(pv / gv, gv / pv)
        log_diff = np.log(pv) - np.log(gv)
        self._totals += np.array([
            (thresh < 1.25).sum(),
            (thresh < 1.25 ** 2).sum(),
            (thresh < 1.25 ** 3).sum(),
            (np.abs(pv - gv) / gv).sum(),
            ((pv - gv) ** 2).sum(),
            (log_diff ** 2).sum(),
            log_diff.sum(),
        ])
        self._count += float(n)

    def process(self):
        t = self._totals
        n = self._count
        if n == 0:
            return {"delta1": 0.0, "delta2": 0.0, "delta3": 0.0,
                    "abs_rel": 0.0, "rmse": 0.0, "silog": 0.0}
        d1, d2, d3, abs_rel, rmse_sq, silog_a, silog_b = (float(x) for x in t)
        silog = max((silog_a / n) - (silog_b / n) ** 2, 0.0) ** 0.5 * 100
        return {
            "delta1": d1 / n,
            "delta2": d2 / n,
            "delta3": d3 / n,
            "abs_rel": abs_rel / n,
            "rmse": (rmse_sq / n) ** 0.5,
            "silog": silog,
        }


def main(args):
    pred_dir = args.pred_dir
    gt_dir = args.gt_dir
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
    if not pred_files:
        raise FileNotFoundError("no pred .npy found in {}".format(pred_dir))
    logging.info("found {} prediction files".format(len(pred_files)))

    metrics = DepthMetrics(min_depth=args.min_depth, max_depth=args.max_depth, align=args.align)
    matched = 0
    for pf in pred_files:
        basename = os.path.splitext(os.path.basename(pf))[0]
        gf = os.path.join(gt_dir, basename + ".npy")
        if not os.path.exists(gf):
            logging.warning("skip {}: no gt {}".format(basename, gf))
            continue
        pred = np.load(pf).astype(np.float32)
        # 官方口径：GT 深度 nan/inf -> 0，再 resize 到与预测同分辨率（INTER_NEAREST）
        gt = np.nan_to_num(np.load(gf).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if gt.shape[:2] != pred.shape[-2:]:
            gt = cv2.resize(gt, (pred.shape[-1], pred.shape[-2]), interpolation=cv2.INTER_NEAREST)
        metrics.update_stats(pred, gt)
        matched += 1

    logging.info("evaluated {} images (gt matched)".format(matched))
    res = metrics.process()
    print("=" * 50)
    print("NYU Depth V2 (Eigen split) evaluation results:")
    print("-" * 50)
    print("delta1 : {:.4f}".format(res["delta1"]))
    print("delta2 : {:.4f}".format(res["delta2"]))
    print("delta3 : {:.4f}".format(res["delta3"]))
    print("abs_rel: {:.4f}".format(res["abs_rel"]))
    print("rmse   : {:.4f}".format(res["rmse"]))
    print("silog  : {:.4f}".format(res["silog"]))
    print("=" * 50)
    return res


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--pred_dir', type=str, default='./results/depth_maps', help='directory of predicted depth .npy')
    parser.add_argument('--gt_dir', type=str, default='./datasets/nyu-depth/depth/val', help='directory of gt depth .npy')
    parser.add_argument('--min_depth', type=float, default=0.001, help='min valid depth (m)')
    parser.add_argument('--max_depth', type=float, default=100.0, help='max valid depth (m)')
    parser.add_argument('--align', type=str, default='median', choices=['median', 'none'], help='per-image scale alignment')
    return parser.parse_args()


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')