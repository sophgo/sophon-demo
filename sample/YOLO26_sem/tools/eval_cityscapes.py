#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""Cityscapes 语义分割精度评测：mIoU / Pixel Accuracy.

用法：
    python3 eval_cityscapes.py --pred_dir ../python/results/segmaps \
        --gt_dir ../datasets/cityscapes/gtFine/val \
        --img_suffix _leftImg8bit.png --gt_suffix _gtFine_labelIds.png

说明：
    pred_dir 下是推理保存的灰度类别图（像素值 0..18，255 为忽略）。
    gt_dir 下是 Cityscapes gtFine 的 labelIds 标签图，文件名形如
    {city}_{seq}_{frame}_gtFine_labelIds.png。脚本会把 labelIds 映射为 trainIds，
    与预测类别图逐像素比对，输出 mIoU / 像素准确率 / 逐类 IoU。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))
from utils import CLASSES, CITYSCAPES_LABEL_MAP  # noqa: E402

IGNORE_LABEL = 255


def load_seg(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img.astype(np.uint8)


def main():
    import argparse
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--pred_dir', type=str, required=True, help='dir of predicted segmaps')
    parser.add_argument('--gt_dir', type=str, required=True, help='dir of gtFine labelIds')
    parser.add_argument('--img_suffix', type=str, default='_leftImg8bit.png')
    parser.add_argument('--gt_suffix', type=str, default='_gtFine_labelIds.png')
    args = parser.parse_args()

    nc = len(CLASSES)
    total_intersect = np.zeros(nc, dtype=np.float64)
    total_union = np.zeros(nc, dtype=np.float64)
    total_pred_label = np.zeros(nc, dtype=np.float64)
    total_label = np.zeros(nc, dtype=np.float64)

    pred_files = sorted(f for f in os.listdir(args.pred_dir) if f.endswith('.png'))
    if not pred_files:
        raise RuntimeError(f"no png found in {args.pred_dir}")

    # gtFine 标签按 city 分子目录存放（val/{city}/），这里递归建索引
    gt_index = {}
    for root, _, files in os.walk(args.gt_dir):
        for f in files:
            if f.endswith(args.gt_suffix):
                gt_index[f] = os.path.join(root, f)
    if not gt_index:
        raise RuntimeError(f"no {args.gt_suffix} found under {args.gt_dir}")

    matched = 0
    for pred_name in pred_files:
        stem = pred_name[: -len(args.img_suffix)] if pred_name.endswith(args.img_suffix) else os.path.splitext(pred_name)[0]
        gt_name = stem + args.gt_suffix
        gt_path = gt_index.get(gt_name)
        if gt_path is None:
            print(f"[skip] no gt for {pred_name} -> {gt_name}", file=sys.stderr)
            continue
        pred = load_seg(os.path.join(args.pred_dir, pred_name))
        gt_labels = load_seg(gt_path)
        gt = label_ids_to_train_ids(gt_labels)

        if pred.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = gt != IGNORE_LABEL
        pred_valid = pred[mask]
        gt_valid = gt[mask]

        area_intersect, _ = np.histogram(pred_valid[pred_valid == gt_valid], bins=np.arange(nc + 1))
        area_pred, _ = np.histogram(pred_valid, bins=np.arange(nc + 1))
        area_label, _ = np.histogram(gt_valid, bins=np.arange(nc + 1))
        area_union = area_pred + area_label - area_intersect

        total_intersect += area_intersect
        total_union += area_union
        total_pred_label += area_pred
        total_label += area_label
        matched += 1

    if matched == 0:
        raise RuntimeError("no matched gt, check --img_suffix/--gt_suffix")

    iou = total_intersect / np.where(total_union == 0, 1, total_union)
    iou[total_union == 0] = np.nan
    pixel_acc = total_intersect.sum() / total_label.sum()
    miou = np.nanmean(iou)

    print(f"matched images: {matched}")
    print(f"{'class':>16s} {'IoU':>8s}")
    for i, name in enumerate(CLASSES):
        print(f"{name:>16s} {iou[i]*100:8.2f}")
    print("-" * 30)
    print(f"mIoU        : {miou*100:.2f}")
    print(f"Pixel Acc   : {pixel_acc*100:.2f}")


def label_ids_to_train_ids(label_gt):
    lut = np.array(CITYSCAPES_LABEL_MAP, dtype=np.uint8)
    out = np.full_like(label_gt, 255, dtype=np.uint8)
    mask = (label_gt >= 0) & (label_gt < len(lut))
    out[mask] = lut[label_gt[mask]]
    return out


if __name__ == '__main__':
    main()