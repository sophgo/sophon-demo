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
import numpy as np
import custom as dp


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    
    # parser.add_argument('--result_json', type=str, default='cpp/segformer_bmcv/results/segformer_fp32.b0.512x1024.city.160k.bmodel_cityscapes_small_bmcv_cpp_result.json', help='path of result json')
    # parser.add_argument('--result_json', type=str, default='cpp/segformer_sail/results/segformer_fp32.b0.512x1024.city.160k.bmodel_cityscapes_small_sail_cpp_result.json', help='path of result json')
    parser.add_argument('--result_json', type=str, default='python/results/segformer.b0.512x1024.city.160k_fp32_1b.bmodel_cityscapes_small_opencv_python_result.json', help='path of result json')
    args = parser.parse_args()
    return args

def main(args):

    with open(args.result_json, 'r') as f:
        res_json = json.load(f)
    res_cls=[]
    for img_res in res_json["img_info"]:
        res = img_res["res"]
        res_cls.append(res)
    
    dp.evaluate(results=res_cls,img_infos=res_json["img_info"],ann_dir=res_json["ann_dir"])

if __name__ == '__main__':
    args = argsparser()
    main(args)