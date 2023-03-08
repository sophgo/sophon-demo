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
    parser.add_argument('--gt_path', type=str, default='../datasets/ground_truth.json', help='path of label json')
    parser.add_argument('--result_json', type=str, default='../cpp/c3d_bmcv/results/c3d_fp32_1b.bmodel_bmcv_cpp.json', help='path of result json')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.gt_path, 'r') as f:
        gt_json = json.load(f)
    with open(args.result_json, 'r') as f:
        res_json = json.load(f)
    total = 0
    correct = 0
    for res in res_json:
        if(res_json[res] == gt_json[res]):
            correct += 1
        total += 1
    logging.info("ACC = {}".format(correct / total))
    
if __name__ == '__main__':
    args = argsparser()
    main(args)