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
|   SE5-16    |   c3d_opencv.py   |   c3d_fp32_1b.bmodel    |      66.43      |      30.22      |      68.69      |      0.09       |
|   SE5-16    |   c3d_opencv.py   |   c3d_fp32_4b.bmodel    |      67.00      |      37.55      |      56.48      |      0.03       |
|   SE5-16    |   c3d_opencv.py   |   c3d_int8_1b.bmodel    |      66.39      |      30.39      |      34.65      |      0.09       |
|   SE5-16    |   c3d_opencv.py   |   c3d_int8_4b.bmodel    |      67.18      |      37.69      |      13.71      |      0.03       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_fp32_1b.bmodel    |      71.78      |      26.17      |      62.32      |      0.01       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_fp32_4b.bmodel    |      71.94      |      25.91      |      50.09      |      0.00       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_int8_1b.bmodel    |      71.73      |      26.07      |      28.22      |      0.01       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_int8_4b.bmodel    |      71.67      |      25.80      |      7.39       |      0.00       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_fp32_1b.bmodel    |      75.55      |      6.74       |      62.29      |      0.01       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_fp32_4b.bmodel    |      74.63      |      6.62       |      50.08      |      0.00       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_int8_1b.bmodel    |      74.72      |      6.72       |      28.21      |      0.01       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_int8_4b.bmodel    |      74.97      |      6.57       |      7.38       |      0.00       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp32_1b.bmodel    |      65.80      |      30.95      |      86.39      |      0.09       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp32_4b.bmodel    |      67.21      |      38.68      |      80.74      |      0.03       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp16_1b.bmodel    |      66.06      |      31.05      |      16.78      |      0.09       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp16_4b.bmodel    |      67.21      |      38.67      |      14.18      |      0.03       |
|   SE7-32    |   c3d_opencv.py   |   c3d_int8_1b.bmodel    |      65.83      |      30.88      |      12.88      |      0.09       |
|   SE7-32    |   c3d_opencv.py   |   c3d_int8_4b.bmodel    |      67.07      |      38.60      |      11.53      |      0.03       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp32_1b.bmodel    |      71.89      |      26.43      |      79.06      |      0.01       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp32_4b.bmodel    |      72.32      |      26.08      |      73.65      |      0.00       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp16_1b.bmodel    |      71.86      |      26.39      |      9.48       |      0.01       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp16_4b.bmodel    |      72.34      |      26.15      |      7.11       |      0.00       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_int8_1b.bmodel    |      72.16      |      26.40      |      5.57       |      0.01       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_int8_4b.bmodel    |      72.36      |      26.14      |      4.40       |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp32_1b.bmodel    |      74.45      |      3.64       |      79.03      |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp32_4b.bmodel    |      75.02      |      3.48       |      73.63      |      0.00       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp16_1b.bmodel    |      74.71      |      3.60       |      9.46       |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp16_4b.bmodel    |      74.66      |      3.49       |      7.10       |      0.00       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_int8_1b.bmodel    |      74.84      |      3.62       |      5.52       |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_int8_4b.bmodel    |      75.16      |      3.49       |      4.41       |      0.00       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp32_1b.bmodel    |      91.91      |      42.36      |     414.42      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp32_4b.bmodel    |      94.88      |      50.27      |     397.28      |      0.05       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp16_1b.bmodel    |      90.77      |      42.20      |      78.45      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp16_4b.bmodel    |      94.31      |      50.30      |      72.12      |      0.04       |
|   SE9-16    |   c3d_opencv.py   |   c3d_int8_1b.bmodel    |      91.94      |      42.05      |      34.79      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |   c3d_int8_4b.bmodel    |      93.83      |      50.44      |      31.97      |      0.04       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp32_1b.bmodel    |     132.62      |     387.37      |     405.13      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp32_4b.bmodel    |     132.72      |     387.10      |     388.00      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp16_1b.bmodel    |     132.28      |     387.39      |      69.19      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp16_4b.bmodel    |     132.06      |     387.35      |      62.73      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_int8_1b.bmodel    |     131.51      |     387.38      |      25.53      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_int8_4b.bmodel    |     132.48      |     387.16      |      22.49      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp32_1b.bmodel    |     142.28      |      11.06      |     405.10      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp32_4b.bmodel    |     142.33      |      10.91      |     387.99      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp16_1b.bmodel    |     143.40      |      11.16      |      69.17      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp16_4b.bmodel    |     142.57      |      11.13      |      62.73      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_int8_1b.bmodel    |     143.75      |      11.41      |      25.50      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_int8_4b.bmodel    |     142.03      |      10.91      |      22.48      |      0.01       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp32_1b_2core.bmodel |      92.42      |      42.31      |     413.57      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp32_4b_2core.bmodel |      94.70      |      50.44      |     397.63      |      0.05       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp16_1b_2core.bmodel |      92.57      |      41.98      |      64.42      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp16_4b_2core.bmodel |      94.33      |      50.32      |      58.00      |      0.04       |
|   SE9-16    |   c3d_opencv.py   |c3d_int8_1b_2core.bmodel |      92.22      |      42.16      |      31.64      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |c3d_int8_4b_2core.bmodel |      94.32      |      50.60      |      29.26      |      0.04       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp32_1b_2core.bmodel |     132.71      |     387.49      |     404.24      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp32_4b_2core.bmodel |     133.12      |     387.29      |     388.47      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp16_1b_2core.bmodel |     132.91      |     387.52      |      55.13      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp16_4b_2core.bmodel |     133.25      |     387.12      |      48.66      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |c3d_int8_1b_2core.bmodel |     132.66      |     387.34      |      22.35      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |c3d_int8_4b_2core.bmodel |     132.56      |     387.10      |      19.32      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp32_1b_2core.bmodel |     143.02      |      11.32      |     404.23      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp32_4b_2core.bmodel |     142.85      |      10.80      |     388.46      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp16_1b_2core.bmodel |     142.27      |      11.26      |      55.10      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp16_4b_2core.bmodel |     142.10      |      10.93      |      48.67      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_int8_1b_2core.bmodel |     142.53      |      11.21      |      22.31      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_int8_4b_2core.bmodel |     142.31      |      10.81      |      19.31      |      0.01       |
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
    parser.add_argument('--bmodel', type=str, default='c3d_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='c3d_opencv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/opencv_c3d_fp32_1b.bmodel_python_debug.log')
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
        elif abs(statis - extracted_data[key]) / statis > 0.2:
            print("{:} time, diff ratio > 0.2".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^25}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
