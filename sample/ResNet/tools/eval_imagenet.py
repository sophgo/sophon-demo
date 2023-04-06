#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import argparse
import json
import logging
logging.basicConfig(level=logging.DEBUG)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--gt_path', type=str, default='../datasets/imagenet_val_1k/label.txt', help='path of label')
    parser.add_argument('--result_json', type=str, default='../python/results/resnet50_fp32_1b.bmodel_img_opencv_python_result.json', help='path of result json')
    args = parser.parse_args()
    return args

def main(args):

    d_gt = dict([l[:-1].split('\t') for l in open(args.gt_path, 'r').readlines()])
    
    d_pred = {}
    with open(args.result_json, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        d_pred[res['filename'].split('/')[-1]] = res['prediction']

    correct = 0
    for k, gt in d_gt.items():
        prediction = d_pred[k]
        if int(gt)==prediction:
            correct += 1
    acc = correct / float(len(d_gt))

    logging.info('gt_path: {}'.format(args.gt_path))
    logging.info('pred_path: {}'.format(args.result_json))
    logging.info('ACC: {:.5f}%'.format(acc*100))


if __name__ == '__main__':
    args = argsparser()
    main(args)
