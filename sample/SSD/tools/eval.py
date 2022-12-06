#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import argparse
import json
import logging
logging.basicConfig(level=logging.DEBUG)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--ground_truths', type=str, default='../data/pascal_test2007.json', help='path of label json')
    parser.add_argument('--result_json', type=str, default='../cpp/ssd_bmcv/build/bmcv_cpp_result_b1.json', help='path of result json')
    args = parser.parse_args()
    return args
def main(args):
    with open(args.result_json,'r') as rj:
        results = json.load(rj)
    pred_list = []
    for i in results:
        pred_list.append(i)
    print(pred_list[0])
    anno_gts = args.ground_truths
    gt = COCO(anno_gts)
    pred = gt.loadRes(pred_list)
    eval = COCOeval(gt, pred, 'bbox')
    #eval.params.maxDets = [5, 20, 100]
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
if __name__ == '__main__':
    args = argsparser()
    main(args)