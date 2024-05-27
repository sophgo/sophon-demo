#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import time
import os
import numpy as np
import cv2
import argparse
from skimage.metrics import structural_similarity as ssim

def img_size_type(arg):
    # 将字符串解析为列表类型
    img_sizes = arg.strip('[]').split('],[')
    img_sizes = [size.split(',') for size in img_sizes]
    img_sizes = [[int(width), int(height)] for width, height in img_sizes]
    return img_sizes

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--left_results', type=str, help='onnx results image dir')
    parser.add_argument('--right_results', type=str, help='bmodel results image dir')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    args = parse_opt()
    input_file_list = []
    output_file_list = []
    if os.path.isdir(args.right_results):
        for root, dirs, filenames in os.walk(args.right_results):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                input_img_file = os.path.join(root, filename)
                output_img_file = os.path.join(args.left_results, filename)
                if os.path.exists(output_img_file):
                    input_file_list.append(input_img_file)
                    output_file_list.append(output_img_file)
                else:
                    print("cannot file onnx_img_file for comparison: ", output_img_file)
    total_ssim = 0
    count = 0
    for i in range(len(input_file_list)):
        input_img = cv2.imread(input_file_list[i])
        output_img = cv2.imread(output_file_list[i])
        if input_img.shape != output_img.shape:
            print("{}-bmodel img's shape is not the same with onnx img. {} vs {}".format(input_file_list[i], input_img.shape, output_img.shape))
            continue
        tmp_ssim = ssim(input_img, output_img, channel_axis=-1)
        print("{} vs {}, ssim: {}".format(input_file_list[i], output_file_list[i], tmp_ssim))
        total_ssim += tmp_ssim
        count += 1
    avg_ssim = total_ssim / count 
    print("avg_ssim: ", avg_ssim)