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
|    测试平台  |     测试程序  |      测试模型     |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------- | ---------------- | -------- | ---------   | ---------------| --------- |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_fp32_1b.bmodel|     128.55      |     535.26      |     280.79      |      0.28       |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_fp32_4b.bmodel|     128.06      |     597.35      |     284.46      |      0.14       |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_fp16_1b.bmodel|     129.23      |     534.43      |     114.50      |      0.28       |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_fp16_4b.bmodel|     127.40      |     595.62      |     121.54      |      0.14       |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_int8_1b.bmodel|     129.18      |     535.00      |     105.71      |      0.28       |
|   SE7-32    |slowfast_opencv.py |slowfast_bm1684x_int8_4b.bmodel|     128.09      |     596.31      |     114.01      |      0.13       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_fp32_1b.bmodel|      95.94      |     138.04      |     199.35      |      0.37       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_fp32_4b.bmodel|      96.54      |     137.84      |     193.76      |      0.35       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_fp16_1b.bmodel|      95.82      |     137.14      |      32.91      |      0.37       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_fp16_4b.bmodel|      95.93      |     136.93      |      31.75      |      0.35       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_int8_1b.bmodel|      95.68      |     138.00      |      24.19      |      0.37       |
|   SE7-32    |slowfast_opencv.soc|slowfast_bm1684x_int8_4b.bmodel|      95.88      |     137.42      |      23.81      |      0.34       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp32_1b.bmodel|     177.17      |     731.05      |     1256.12     |      0.41       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp32_4b.bmodel|     176.96      |     804.29      |     1254.24     |      0.20       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp16_1b.bmodel|     177.98      |     732.07      |     324.62      |      0.40       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp16_4b.bmodel|     177.27      |     803.49      |     329.01      |      0.20       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_int8_1b.bmodel|     176.73      |     730.13      |     171.32      |      0.39       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_int8_4b.bmodel|     177.92      |     805.44      |     177.44      |      0.19       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp32_1b.bmodel|     117.47      |     174.13      |     1155.48     |      0.62       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp32_4b.bmodel|     118.97      |     174.07      |     1142.85     |      0.65       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp16_1b.bmodel|     116.81      |     174.06      |     222.16      |      0.61       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp16_4b.bmodel|     119.03      |     174.26      |     217.44      |      0.53       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_int8_1b.bmodel|     117.55      |     174.38      |      69.07      |      0.59       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_int8_4b.bmodel|     118.12      |     174.30      |      66.01      |      0.55       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp32_1b_2core.bmodel|     177.62      |     731.70      |     1103.99     |      0.40       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp32_4b_2core.bmodel|     177.36      |     806.39      |     1095.97     |      0.20       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp16_1b_2core.bmodel|     177.74      |     730.70      |     299.17      |      0.41       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_fp16_4b_2core.bmodel|     176.96      |     806.46      |     305.09      |      0.20       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_int8_1b_2core.bmodel|     176.89      |     731.70      |     155.36      |      0.40       |
|   SE9-16    |slowfast_opencv.py |slowfast_bm1688_int8_4b_2core.bmodel|     177.62      |     804.52      |     162.30      |      0.19       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp32_1b_2core.bmodel|     118.24      |     173.65      |     998.84      |      0.60       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp32_4b_2core.bmodel|     119.54      |     173.59      |     984.25      |      0.91       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp16_1b_2core.bmodel|     117.01      |     172.96      |     197.01      |      0.60       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_fp16_4b_2core.bmodel|     119.55      |     173.87      |     193.42      |      0.56       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_int8_1b_2core.bmodel|     117.10      |     173.77      |      52.90      |      0.59       |
|   SE9-16    |slowfast_opencv.soc|slowfast_bm1688_int8_4b_2core.bmodel|     118.87      |     173.69      |      50.75      |      0.56       |
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
    'decode': re.compile(r'\[.*decode_time.*\]  loops:.*avg: ([\d.]+) ms'),
    'preprocess': re.compile(r'\[.*preprocess_time.*\]  loops:.*avg: ([\d.]+) ms'),
    'inference': re.compile(r'\[.*inference.*\]  loops:.*avg: ([\d.]+) ms'),
    'postprocess': re.compile(r'\[.*postprocess_time.*\]  loops:.*avg: ([\d.]+) ms'),
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
    parser.add_argument('--bmodel', type=str, default='slowfast_bm1684x_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='slowfast_opencv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/opencv_slowfast_bm1684x_fp32_1b.bmodel_python_debug.log')
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
        
