from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import argparse

TEMPLATE_D = {
    "image_id": 42,
    "category_id": 18,
    "bbox": [258.15, 41.29, 348.26, 243.78],
    "score": 0.236
    }
coco_class_map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def parse_result_file(result_path):
    d_list = []
    lines = []
    i = 0
    for l in open(result_path, 'r').readlines():
        if len(l) == 1:
            continue
        lines.append(l)
        i += 1
        if (i == 7):
            d = TEMPLATE_D.copy()
            d["image_id"] = int(lines[0][1:-6])
            d["category_id"] = coco_class_map[int(lines[1].split("=")[1])]
            x = float(lines[3].split("=")[1])
            y = float(lines[4].split("=")[1])
            w = float(lines[5].split("=")[1]) - x
            h = float(lines[6].split("=")[1]) - y
            d["bbox"] = [x, y, w, h]
            d["score"] = float(lines[2].split("=")[1])
            d_list.append(d)
            i = 0
            lines.clear()
    return coco_gt.loadRes(d_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truths', type=str, default='instances_val2017.json')
    parser.add_argument('--detections', type=str, default='result.txt')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    ann_file = args.ground_truths
    result_path = args.detections

    coco_gt = COCO(ann_file)

    coco_pred = parse_result_file(result_path)

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

    coco_eval.params.maxDets = [5, 20, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
