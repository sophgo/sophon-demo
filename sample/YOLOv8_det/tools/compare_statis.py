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
|   SE5-16    | yolov8_opencv.py  | yolov8s_fp32_1b.bmodel  |      18.98      |      21.50      |      31.77      |      5.11       |
|   SE5-16    | yolov8_opencv.py  | yolov8s_int8_1b.bmodel  |      15.02      |      21.35      |      20.80      |      4.85       |
|   SE5-16    | yolov8_opencv.py  | yolov8s_int8_4b.bmodel  |      15.06      |      23.84      |      14.39      |      5.15       |
|   SE5-16    |  yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel  |      3.63       |      2.80       |      28.88      |      4.99       |
|   SE5-16    |  yolov8_bmcv.py   | yolov8s_int8_1b.bmodel  |      3.67       |      2.83       |      18.08      |      4.97       |
|   SE5-16    |  yolov8_bmcv.py   | yolov8s_int8_4b.bmodel  |      3.51       |      2.67       |      11.33      |      4.37       |
|   SE5-16    |  yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel  |      4.86       |      1.54       |      26.41      |      8.53       |
|   SE5-16    |  yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel  |      4.89       |      1.55       |      15.58      |      8.53       |
|   SE5-16    |  yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel  |      4.75       |      1.49       |      9.34       |      8.50       |
|   SE5-16    |  yolov8_bmcv.soc  |yolov8s_opt_fp32_1b.bmodel|      4.94       |      1.54       |      26.62      |      2.65       |
|   SE5-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_1b.bmodel|      4.88       |      1.54       |      15.02      |      2.60       |
|   SE5-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_4b.bmodel|      4.82       |      1.49       |      7.37       |      2.66       |
|   SE7-32    | yolov8_opencv.py  | yolov8s_fp32_1b.bmodel  |      21.70      |      22.55      |      35.12      |      5.38       |
|   SE7-32    | yolov8_opencv.py  | yolov8s_fp16_1b.bmodel  |      14.95      |      22.53      |      11.42      |      5.39       |
|   SE7-32    | yolov8_opencv.py  | yolov8s_int8_1b.bmodel  |      14.96      |      22.87      |      9.21       |      5.36       |
|   SE7-32    | yolov8_opencv.py  | yolov8s_int8_4b.bmodel  |      15.00      |      24.62      |      8.98       |      5.45       |
|   SE7-32    |  yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel  |      3.16       |      2.38       |      31.87      |      5.42       |
|   SE7-32    |  yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel  |      3.16       |      2.37       |      8.13       |      5.45       |
|   SE7-32    |  yolov8_bmcv.py   | yolov8s_int8_1b.bmodel  |      3.16       |      2.37       |      5.96       |      5.43       |
|   SE7-32    |  yolov8_bmcv.py   | yolov8s_int8_4b.bmodel  |      2.97       |      2.17       |      5.52       |      4.87       |
|   SE7-32    |  yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel  |      4.36       |      0.74       |      29.15      |      8.65       |
|   SE7-32    |  yolov8_bmcv.soc  | yolov8s_fp16_1b.bmodel  |      4.39       |      0.74       |      5.46       |      8.67       |
|   SE7-32    |  yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel  |      4.38       |      0.74       |      3.25       |      8.67       |
|   SE7-32    |  yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel  |      4.25       |      0.71       |      3.33       |      8.59       |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s_opt_fp32_1b.bmodel|      4.41       |      0.74       |      29.35      |      2.63       |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s_opt_fp16_1b.bmodel|      4.36       |      0.74       |      5.58       |      2.62       |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s_opt_int8_1b.bmodel|      4.39       |      0.74       |      2.90       |      2.62       |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s_opt_int8_4b.bmodel|      4.24       |      0.71       |      2.89       |      2.72       |
|   SE9-16    | yolov8_opencv.py  | yolov8s_fp32_1b.bmodel  |      23.35      |      29.68      |     169.78      |      6.93       |
|   SE9-16    | yolov8_opencv.py  | yolov8s_fp16_1b.bmodel  |      19.39      |      30.28      |      41.80      |      6.95       |
|   SE9-16    | yolov8_opencv.py  | yolov8s_int8_1b.bmodel  |      19.36      |      29.60      |      18.25      |      6.88       |
|   SE9-16    | yolov8_opencv.py  | yolov8s_int8_4b.bmodel  |      19.27      |      33.23      |      17.60      |      7.40       |
|   SE9-16    |  yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel  |      4.45       |      4.99       |     165.95      |      6.98       |
|   SE9-16    |  yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel  |      4.40       |      4.96       |      37.91      |      6.97       |
|   SE9-16    |  yolov8_bmcv.py   | yolov8s_int8_1b.bmodel  |      4.41       |      4.92       |      14.41      |      6.95       |
|   SE9-16    |  yolov8_bmcv.py   | yolov8s_int8_4b.bmodel  |      4.29       |      4.64       |      13.41      |      6.14       |
|   SE9-16    |  yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel  |      5.80       |      1.83       |     162.12      |      12.11      |
|   SE9-16    |  yolov8_bmcv.soc  | yolov8s_fp16_1b.bmodel  |      5.92       |      1.82       |      34.29      |      12.09      |
|   SE9-16    |  yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel  |      5.85       |      1.83       |      10.82      |      12.10      |
|   SE9-16    |  yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel  |      5.70       |      1.74       |      10.57      |      12.05      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_fp32_1b.bmodel|      5.85       |      1.82       |     161.79      |      3.69       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_fp16_1b.bmodel|      5.88       |      1.82       |      34.19      |      3.67       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_1b.bmodel|      5.87       |      1.82       |      8.10       |      3.67       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_4b.bmodel|      5.67       |      1.74       |      7.86       |      3.70       |
|   SE9-16    | yolov8_opencv.py  |yolov8s_fp32_1b_2core.bmodel|      35.37      |      29.56      |     100.48      |      6.94       |
|   SE9-16    | yolov8_opencv.py  |yolov8s_fp16_1b_2core.bmodel|      19.39      |      30.15      |      29.73      |      6.93       |
|   SE9-16    | yolov8_opencv.py  |yolov8s_int8_1b_2core.bmodel|      19.32      |      29.63      |      16.04      |      6.88       |
|   SE9-16    | yolov8_opencv.py  |yolov8s_int8_4b_2core.bmodel|      19.33      |      33.28      |      13.36      |      7.65       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s_fp32_1b_2core.bmodel|      4.49       |      4.98       |      96.83      |      6.96       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s_fp16_1b_2core.bmodel|      4.42       |      4.92       |      25.98      |      6.94       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s_int8_1b_2core.bmodel|      4.39       |      4.90       |      11.99      |      6.95       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s_int8_4b_2core.bmodel|      4.28       |      4.59       |      9.10       |      6.12       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_fp32_1b_2core.bmodel|      5.90       |      1.83       |      93.00      |      12.10      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_fp16_1b_2core.bmodel|      5.89       |      1.82       |      22.28      |      12.10      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_int8_1b_2core.bmodel|      5.92       |      1.83       |      8.35       |      12.18      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_int8_4b_2core.bmodel|      5.78       |      1.75       |      6.24       |      12.08      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_fp32_1b_2core.bmodel|      5.92       |      1.84       |      92.15      |      3.72       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_fp16_1b_2core.bmodel|      5.90       |      1.83       |      22.06      |      3.66       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_1b_2core.bmodel|      5.89       |      1.83       |      7.25       |      3.67       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s_opt_int8_4b_2core.bmodel|      5.75       |      1.75       |      4.84       |      3.71       |
|    SE9-8    | yolov8_opencv.py  | yolov8s_fp32_1b.bmodel  |      25.22      |      30.41      |     169.19      |      7.01       |
|    SE9-8    | yolov8_opencv.py  | yolov8s_fp16_1b.bmodel  |      24.42      |      29.83      |      41.77      |      6.97       |
|    SE9-8    | yolov8_opencv.py  | yolov8s_int8_1b.bmodel  |      19.31      |      30.46      |      18.25      |      6.95       |
|    SE9-8    | yolov8_opencv.py  | yolov8s_int8_4b.bmodel  |      19.42      |      32.97      |      17.66      |      7.41       |
|    SE9-8    |  yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel  |      4.34       |      4.78       |     165.23      |      7.05       |
|    SE9-8    |  yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel  |      4.34       |      4.77       |      37.60      |      7.07       |
|    SE9-8    |  yolov8_bmcv.py   | yolov8s_int8_1b.bmodel  |      4.29       |      4.76       |      14.32      |      7.10       |
|    SE9-8    |  yolov8_bmcv.py   | yolov8s_int8_4b.bmodel  |      4.14       |      4.47       |      13.17      |      6.21       |
|    SE9-8    |  yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel  |      5.85       |      1.82       |     161.46      |      12.21      |
|    SE9-8    |  yolov8_bmcv.soc  | yolov8s_fp16_1b.bmodel  |      5.83       |      1.82       |      34.05      |      12.19      |
|    SE9-8    |  yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel  |      5.77       |      1.82       |      10.71      |      12.20      |
|    SE9-8    |  yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel  |      5.65       |      1.74       |      10.46      |      12.14      |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s_opt_fp32_1b.bmodel|      5.85       |      1.82       |     161.75      |      3.71       |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s_opt_fp16_1b.bmodel|      5.79       |      1.82       |      34.15      |      4.01       |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s_opt_int8_1b.bmodel|      5.70       |      1.82       |      7.77       |      3.68       |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s_opt_int8_4b.bmodel|      5.55       |      1.73       |      7.54       |      3.73       |
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
    parser.add_argument('--bmodel', type=str, default='yolov8_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov8_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov8_fp32_1b.bmodel_python_test.log')
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
        
