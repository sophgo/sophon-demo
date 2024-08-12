#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
|    测试平台 |      测试程序       |        测试模型         |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | ----------------------- | --------- | ---------- | ----------- | ----------- |
|   SE5-16    |ppyolov3_opencv.py |   ppyolov3_fp32_1b.bmodel    |      15.24      |      27.63      |      86.30      |      97.82      |
|   SE5-16    |ppyolov3_opencv.py |   ppyolov3_int8_1b.bmodel    |      15.21      |      27.72      |      59.65      |      97.15      |
|   SE5-16    | ppyolov3_bmcv.py  |   ppyolov3_fp32_1b.bmodel    |      3.64       |      2.32       |      83.97      |     110.51      |
|   SE5-16    | ppyolov3_bmcv.py  |   ppyolov3_int8_1b.bmodel    |      3.63       |      2.31       |      57.35      |     110.27      |
|   SE5-16    | ppyolov3_bmcv.soc |   ppyolov3_fp32_1b.bmodel    |      4.86       |      1.56       |      78.13      |      16.70      |
|   SE5-16    | ppyolov3_bmcv.soc |   ppyolov3_int8_1b.bmodel    |      4.88       |      1.57       |      51.60      |      16.80      |
|   SE5-16    | ppyolov3_sail.soc |   ppyolov3_fp32_1b.bmodel    |      3.27       |      3.08       |      79.04      |      15.78      |
|   SE5-16    | ppyolov3_sail.soc |   ppyolov3_int8_1b.bmodel    |      3.27       |      3.08       |      52.49      |      15.76      |
|   SE7-32    |ppyolov3_opencv.py |   ppyolov3_fp32_1b.bmodel    |      15.07      |      28.61      |     158.26      |      95.17      |
|   SE7-32    |ppyolov3_opencv.py |   ppyolov3_fp16_1b.bmodel    |      15.25      |      29.19      |      23.83      |      96.63      |
|   SE7-32    |ppyolov3_opencv.py |   ppyolov3_int8_1b.bmodel    |      15.33      |      28.58      |      16.63      |      96.36      |
|   SE7-32    | ppyolov3_bmcv.py  |   ppyolov3_fp32_1b.bmodel    |      3.16       |      1.79       |     155.52      |     107.93      |
|   SE7-32    | ppyolov3_bmcv.py  |   ppyolov3_fp16_1b.bmodel    |      3.15       |      1.79       |      20.91      |     108.05      |
|   SE7-32    | ppyolov3_bmcv.py  |   ppyolov3_int8_1b.bmodel    |      3.13       |      1.79       |      13.79      |     106.55      |
|   SE7-32    | ppyolov3_bmcv.soc |   ppyolov3_fp32_1b.bmodel    |      4.34       |      0.66       |     148.99      |      16.73      |
|   SE7-32    | ppyolov3_bmcv.soc |   ppyolov3_fp16_1b.bmodel    |      4.34       |      0.65       |      14.33      |      16.73      |
|   SE7-32    | ppyolov3_bmcv.soc |   ppyolov3_int8_1b.bmodel    |      4.34       |      0.66       |      7.19       |      16.74      |
|   SE7-32    | ppyolov3_sail.soc |   ppyolov3_fp32_1b.bmodel    |      2.76       |      2.57       |     149.88      |      15.79      |
|   SE7-32    | ppyolov3_sail.soc |   ppyolov3_fp16_1b.bmodel    |      2.76       |      2.56       |      15.21      |      15.80      |
|   SE7-32    | ppyolov3_sail.soc |   ppyolov3_int8_1b.bmodel    |      2.76       |      2.56       |      8.08       |      15.77      |
|   SE9-16    |ppyolov3_opencv.py |   ppyolov3_fp32_1b.bmodel    |      19.61      |      36.50      |     746.66      |     132.59      |
|   SE9-16    |ppyolov3_opencv.py |   ppyolov3_fp16_1b.bmodel    |      19.61      |      37.35      |     102.32      |     133.42      |
|   SE9-16    |ppyolov3_opencv.py |   ppyolov3_int8_1b.bmodel    |      19.60      |      37.42      |      39.15      |     133.85      |
|   SE9-16    | ppyolov3_bmcv.py  |   ppyolov3_fp32_1b.bmodel    |      4.45       |      3.95       |     743.54      |     150.03      |
|   SE9-16    | ppyolov3_bmcv.py  |   ppyolov3_fp16_1b.bmodel    |      4.44       |      3.95       |      98.82      |     150.03      |
|   SE9-16    | ppyolov3_bmcv.py  |   ppyolov3_int8_1b.bmodel    |      4.44       |      3.95       |      35.57      |     148.29      |
|   SE9-16    | ppyolov3_bmcv.soc |   ppyolov3_fp32_1b.bmodel    |      5.82       |      1.74       |     734.97      |      23.37      |
|   SE9-16    | ppyolov3_bmcv.soc |   ppyolov3_fp16_1b.bmodel    |      5.80       |      1.73       |      90.38      |      23.31      |
|   SE9-16    | ppyolov3_bmcv.soc |   ppyolov3_int8_1b.bmodel    |      5.79       |      1.73       |      27.23      |      23.29      |
|   SE9-16    | ppyolov3_sail.soc |   ppyolov3_fp32_1b.bmodel    |      3.92       |      5.16       |     737.36      |      22.00      |
|   SE9-16    | ppyolov3_sail.soc |   ppyolov3_fp16_1b.bmodel    |      3.92       |      5.17       |      92.64      |      21.99      |
|   SE9-16    | ppyolov3_sail.soc |   ppyolov3_int8_1b.bmodel    |      3.88       |      5.14       |      29.47      |      21.90      |
|   SE9-16    |ppyolov3_opencv.py |ppyolov3_fp32_1b_2core.bmodel |      19.58      |      38.72      |     395.18      |     133.81      |
|   SE9-16    |ppyolov3_opencv.py |ppyolov3_fp16_1b_2core.bmodel |      19.58      |      37.38      |      64.96      |     134.18      |
|   SE9-16    |ppyolov3_opencv.py |ppyolov3_int8_1b_2core.bmodel |      19.60      |      38.01      |      32.05      |     133.43      |
|   SE9-16    | ppyolov3_bmcv.py  |ppyolov3_fp32_1b_2core.bmodel |      4.47       |      3.94       |     391.58      |     149.92      |
|   SE9-16    | ppyolov3_bmcv.py  |ppyolov3_fp16_1b_2core.bmodel |      4.43       |      3.95       |      61.30      |     150.10      |
|   SE9-16    | ppyolov3_bmcv.py  |ppyolov3_int8_1b_2core.bmodel |      4.46       |      3.95       |      28.60      |     148.18      |
|   SE9-16    | ppyolov3_bmcv.soc |ppyolov3_fp32_1b_2core.bmodel |      5.85       |      1.72       |     383.13      |      23.34      |
|   SE9-16    | ppyolov3_bmcv.soc |ppyolov3_fp16_1b_2core.bmodel |      5.84       |      1.73       |      53.06      |      23.27      |
|   SE9-16    | ppyolov3_bmcv.soc |ppyolov3_int8_1b_2core.bmodel |      5.81       |      1.73       |      20.09      |      23.27      |
|   SE9-16    | ppyolov3_sail.soc |ppyolov3_fp32_1b_2core.bmodel |      3.90       |      5.17       |     385.47      |      22.03      |
|   SE9-16    | ppyolov3_sail.soc |ppyolov3_fp16_1b_2core.bmodel |      3.92       |      5.15       |      55.32      |      22.01      |
|   SE9-16    | ppyolov3_sail.soc |ppyolov3_int8_1b_2core.bmodel |      3.90       |      5.14       |      22.34      |      21.95      |
|    SE9-8    |ppyolov3_opencv.py |   ppyolov3_fp32_1b.bmodel    |      32.64      |      38.33      |     756.09      |     133.57      |
|    SE9-8    |ppyolov3_opencv.py |   ppyolov3_fp16_1b.bmodel    |      25.44      |      38.47      |     109.32      |     133.84      |
|    SE9-8    |ppyolov3_opencv.py |   ppyolov3_int8_1b.bmodel    |      19.68      |      38.21      |      43.81      |     133.42      |
|    SE9-8    | ppyolov3_bmcv.py  |   ppyolov3_fp32_1b.bmodel    |      4.31       |      3.81       |     752.84      |     149.62      |
|    SE9-8    | ppyolov3_bmcv.py  |   ppyolov3_fp16_1b.bmodel    |      4.30       |      3.83       |     105.92      |     149.82      |
|    SE9-8    | ppyolov3_bmcv.py  |   ppyolov3_int8_1b.bmodel    |      4.28       |      3.82       |      40.29      |     147.92      |
|    SE9-8    | ppyolov3_bmcv.soc |   ppyolov3_fp32_1b.bmodel    |      5.80       |      1.71       |     744.18      |      23.43      |
|    SE9-8    | ppyolov3_bmcv.soc |   ppyolov3_fp16_1b.bmodel    |      5.67       |      1.72       |      97.52      |      23.33      |
|    SE9-8    | ppyolov3_bmcv.soc |   ppyolov3_int8_1b.bmodel    |      5.71       |      1.71       |      32.03      |      23.32      |
|    SE9-8    | ppyolov3_sail.soc |   ppyolov3_fp32_1b.bmodel    |      4.67       |      5.07       |     746.61      |      22.05      |
|    SE9-8    | ppyolov3_sail.soc |   ppyolov3_fp16_1b.bmodel    |      5.93       |      5.07       |      99.83      |      22.02      |
|    SE9-8    | ppyolov3_sail.soc |   ppyolov3_int8_1b.bmodel    |      3.80       |      5.05       |      34.31      |      21.97      |
"""
table_data = {
    "platform": [],
    "program": [],
    "bmodel": [],
    "decode": [],
    "preprocess": [],
    "inference": [],
    "postprocess": []
}

patterns_cpp = {
    'decode': re.compile(r'\[.*decode time.*\]  loops:.*avg: ([\d.]+) ms'),
    'preprocess': re.compile(r'\[.*preprocess.*\]  loops:.*avg: ([\d.]+) ms'),
    'inference': re.compile(r'\[.*inference.*\]  loops:.*avg: ([\d.]+) ms'),
    'postprocess': re.compile(r'\[.*postprocess.*\]  loops:.*avg: ([\d.]+) ms'),
}

patterns_python = {
    'decode': re.compile(r'decode_time\(ms\): ([\d.]+)'),
    'preprocess': re.compile(r'preprocess_time\(ms\): ([\d.]+)'),
    'inference': re.compile(r'inference_time\(ms\): ([\d.]+)'),
    'postprocess': re.compile(r'postprocess_time\(ms\): ([\d.]+)'),
}

def extract_times(text, patterns):
    results = {}
    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            results[key] = round(float(match.group(1)),2)
    return results


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--target', type=str, default='BM1684X', help='path of label json')
    parser.add_argument('--platform', type=str, default='soc', help='path of result json')
    parser.add_argument('--bmodel', type=str, default='yolov5s_v6.1_3output_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov5_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov5s_v6.1_3output_fp32_1b.bmodel_python_test.log')
    parser.add_argument('--use_cpu_opt', action="store_true", default=False, help='accelerate cpu postprocess')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    args = argsparser()
    benchmark_path = current_dir + "/benchmark.txt"
        
    for line in baseline.strip().split("\n")[2:]:
        match = re.search(r'\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', line)
        if match:
            table_data["platform"].append(match.group(1))
            table_data["program"].append(match.group(2))
            table_data["bmodel"].append(match.group(3))
            table_data["decode"].append(float(match.group(4)))
            table_data["preprocess"].append(float(match.group(5)))
            table_data["inference"].append(float(match.group(6)))
            table_data["postprocess"].append(float(match.group(7)))

    if args.platform == "soc":
        if args.target == "BM1684X":
            platform = "SE7-32"
        elif args.target == "BM1684":
            platform = "SE5-16"
        elif args.target == "BM1688":
            platform = "SE9-16"
            if multiprocessing.cpu_count() == 6:
                platform = "SE9-8"
        elif args.target == "CV186X":
            platform = "SE9-8"
    else:
        platform = args.target + " SoC" if args.platform == "soc" else args.target + " PCIe"
    min_width = 17
    
    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^13}|{:^19}|{:^30}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
           "platform", "program", "bmodel", "decode_time", "preprocess_time", "inference_time", "postprocess_time", width=min_width)
            f.write(benchmark_str)
            
    with open(args.input, "r") as f:
        data = f.read()
    if args.language == "python":    
        extracted_data = extract_times(data, patterns_python)
    elif args.language == "cpp":
        extracted_data = extract_times(data, patterns_cpp)
    else:
        print("unsupport code language")
    match_index = -1
    for i in range(0, len(table_data["platform"])):
        if platform == table_data["platform"][i] and args.program == table_data["program"][i] and args.bmodel == table_data["bmodel"][i]:
            match_index = i
            break
    baseline_data = {}
    if match_index == -1:
        print("Unmatched case.")
    else:
        baseline_data["decode"] = table_data["decode"][match_index]
        baseline_data["preprocess"] = table_data["preprocess"][match_index]
        baseline_data["inference"] = table_data["inference"][match_index]
        baseline_data["postprocess"] = table_data["postprocess"][match_index]
    for key, statis in baseline_data.items():
        if abs(statis - extracted_data[key]) / statis > 0.4:
            print("{:} time, diff ratio > 0.4".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^30}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
