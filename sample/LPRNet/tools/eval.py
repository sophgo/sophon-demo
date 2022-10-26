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
    parser.add_argument('--label_json', type=str, default='../data/images/test_label.json', help='path of label json')
    parser.add_argument('--result_json', type=str, default='../python/results/test_bmcv_b4_result.json', help='path of result json')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.label_json,'r') as lj:
        label_dict = json.load(lj)
    with open(args.result_json,'r') as rj:
        result_dict = json.load(rj)
    num_label = len(label_dict)
    num_result = len(result_dict)
    if num_label != num_result:
        raise ValueError('num_label must be equal to num_result, num_label={}, num_result={}'.format(num_label, num_result))
    tp = 0
    for filename in label_dict.keys():
        label = label_dict[filename]
        result = result_dict[filename]
        if label == result:
            tp += 1
            
    logging.info("ACC = {}/{} = {}".format(tp, num_label, tp / num_label))
    
if __name__ == '__main__':
    args = argsparser()
    main(args)