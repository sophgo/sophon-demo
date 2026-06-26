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

baseline = """
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | --------  | ---------    | ---------     | ---------      |
|   SE7-32    |yoloworld_opencv.py|yoloworld_fp32_1b.bmodel |      9.56       |      22.52      |      41.79      |      4.48       |
|   SE7-32    |yoloworld_opencv.py|yoloworld_fp16_1b.bmodel |      7.41       |      22.83      |      13.55      |      4.50       |
|   SE7-32    |yoloworld_opencv.py|yoloworld_int8_1b.bmodel |      6.79       |      22.67      |      11.08      |      4.47       |
|   SE7-32    | yoloworld_bmcv.py |yoloworld_fp32_1b.bmodel |      3.03       |      2.30       |      44.70      |      4.52       |
|   SE7-32    | yoloworld_bmcv.py |yoloworld_fp16_1b.bmodel |      3.02       |      2.29       |      16.37      |      4.52       |
|   SE7-32    | yoloworld_bmcv.py |yoloworld_int8_1b.bmodel |      3.05       |      2.29       |      14.01      |      4.48       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_fp32_1b.bmodel |      19.44      |      29.54      |     192.39      |      5.66       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_fp16_1b.bmodel |      12.02      |      30.10      |      46.74      |      5.65       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_int8_1b.bmodel |      9.41       |      29.94      |      23.96      |      5.66       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_fp32_1b.bmodel |      4.23       |      4.71       |     195.66      |      5.66       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_fp16_1b.bmodel |      4.24       |      4.73       |      49.85      |      5.67       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_int8_1b.bmodel |      4.23       |      4.72       |      27.92      |      5.67       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_fp32_1b_2core.bmodel|      9.42       |      29.65      |     106.97      |      5.62       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_fp16_1b_2core.bmodel|      9.42       |      29.96      |      31.96      |      5.63       |
|   SE9-16    |yoloworld_opencv.py|yoloworld_int8_1b_2core.bmodel|      9.39       |      30.01      |      20.31      |      5.64       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_fp32_1b_2core.bmodel|      4.26       |      4.72       |     110.11      |      5.68       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_fp16_1b_2core.bmodel|      4.21       |      4.73       |      35.43      |      5.65       |
|   SE9-16    | yoloworld_bmcv.py |yoloworld_int8_1b_2core.bmodel|      4.24       |      4.71       |      23.66      |      5.68       |
|    SE9-8    |yoloworld_opencv.py|yoloworld_fp32_1b.bmodel |      13.81      |      30.34      |     192.50      |      5.73       |
|    SE9-8    |yoloworld_opencv.py|yoloworld_fp16_1b.bmodel |      14.24      |      29.72      |      46.66      |      5.71       |
|    SE9-8    |yoloworld_opencv.py|yoloworld_int8_1b.bmodel |      13.24      |      29.68      |      24.00      |      5.71       |
|    SE9-8    | yoloworld_bmcv.py |yoloworld_fp32_1b.bmodel |      7.84       |      4.62       |     195.78      |      5.75       |
|    SE9-8    | yoloworld_bmcv.py |yoloworld_fp16_1b.bmodel |      7.40       |      4.60       |      50.11      |      5.72       |
|    SE9-8    | yoloworld_bmcv.py |yoloworld_int8_1b.bmodel |      7.46       |      4.58       |      27.35      |      5.73       |
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
    parser.add_argument('--bmodel', type=str, default='yoloworld_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yoloworld_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yoloworld_fp32_1b.bmodel_python_test.log')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    benchmark_path = current_dir + "/benchmark.txt"
    args = argsparser()
    if args.platform == "soc":
        if args.target == "BM1684X":
            platform = "SE7-32"
        elif args.target == "BM1684":
            platform = "SE5-16"
        elif args.target == "BM1688":
            platform = "SE9-16"
        elif args.target == "CV186X":
            platform = "SE9-8"
    else:
        platform = args.target + " SoC" if args.platform == "soc" else args.target + " PCIe"
    min_width = 17
    
    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^13}|{:^19}|{:^25}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
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
        if statis < 1:
            if abs(statis - extracted_data[key]) > 0.5:
                print("{:} time, diff > 0.5".format(key))
                print("Baseline is:", statis)
                print("Now is: ", extracted_data[key])
                compare_pass = False
        elif abs(statis - extracted_data[key]) / statis > 0.4:
            print("{:} time, diff ratio > 0.4".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^25}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        