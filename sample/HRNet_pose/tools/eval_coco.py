#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import argparse

import logging
logging.basicConfig(level=logging.INFO)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main(args):

    coco_gt = COCO(args.gt_path)

    pred_keypoints = args.results_path
    coco_dt = coco_gt.loadRes(pred_keypoints)

    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0]
    print("mAP: {:.2f}".format(mAP))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)

    parser.add_argument('--results_path', type=str, default='./results/keypoints_results.json', help='Json file containing image info.')

    parser.add_argument('--gt_path', type=str, default='./datasets/coco/person_keypoints_val2017.json', help='Json file containing image info.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argsparser()
    main(args)