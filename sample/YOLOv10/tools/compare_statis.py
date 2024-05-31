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
# 待测试
baseline = """
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | --------  | ---------    | ---------     | ---------      |
| SE7-32   | yolov10_opencv.py | yolov10s_fp32_1b.bmodel | 14.77       | 22.81           | 26.16          | 0.61             |
| SE7-32   | yolov10_opencv.py | yolov10s_fp16_1b.bmodel | 13.74       | 22.48           | 10.22          | 0.61             |
| SE7-32   | yolov10_opencv.py | yolov10s_int8_1b.bmodel | 13.74       | 22.47           | 7.81           | 0.61             |
| SE7-32   | yolov10_opencv.py | yolov10s_int8_4b.bmodel | 13.82       | 24.79           | 6.58           | 0.42             |
| SE7-32   | yolov10_bmcv.py   | yolov10s_fp32_1b.bmodel | 3.00        | 2.25            | 22.57          | 0.56             |
| SE7-32   | yolov10_bmcv.py   | yolov10s_fp16_1b.bmodel | 3.00        | 2.25            | 6.64           | 0.56             |
| SE7-32   | yolov10_bmcv.py   | yolov10s_int8_1b.bmodel | 2.98        | 2.24            | 4.25           | 0.55             |
| SE7-32   | yolov10_bmcv.py   | yolov10s_int8_4b.bmodel | 2.84        | 2.09            | 3.39           | 0.38             |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_fp32_1b.bmodel | 4.25        | 0.74            | 22.02          | 0.025            |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_fp16_1b.bmodel | 4.30        | 0.74            | 6.08           | 0.025            |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_int8_1b.bmodel | 4.30        | 0.74            | 3.69           | 0.025            |
| SE7-32   | yolov10_bmcv.soc  | yolov10s_int8_4b.bmodel | 4.18        | 0.71            | 3.25           | 0.012            |
| SE9-16   | yolov10_opencv.py | yolov10s_fp32_1b.bmodel | 4.48        | 29.99           | 136.02         | 0.84             |
| SE9-16   | yolov10_opencv.py | yolov10s_fp16_1b.bmodel | 4.48        | 30.07           | 30.07          | 0.84             |
| SE9-16   | yolov10_opencv.py | yolov10s_int8_1b.bmodel | 4.45        | 29.65           | 15.19          | 0.83             |
| SE9-16   | yolov10_opencv.py | yolov10s_int8_4b.bmodel | 4.55        | 33.85           | 13.18          | 0.58             |
| SE9-16   | yolov10_bmcv.py   | yolov10s_fp32_1b.bmodel | 4.39        | 4.93            | 131.64         | 0.79             |
| SE9-16   | yolov10_bmcv.py   | yolov10s_fp16_1b.bmodel | 4.38        | 4.90            | 36.22          | 0.81             |
| SE9-16   | yolov10_bmcv.py   | yolov10s_int8_1b.bmodel | 4.31        | 4.87            | 10.77          | 0.81             |
| SE9-16   | yolov10_bmcv.py   | yolov10s_int8_4b.bmodel | 4.21        | 4.59            | 9.30           | 0.54             |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_fp32_1b.bmodel | 5.76        | 1.83            | 35.26          | 0.04             |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_fp16_1b.bmodel | 5.73        | 1.83            | 9.84           | 0.04             |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_int8_1b.bmodel | 5.73        | 1.83            | 9.84           | 0.04             |
| SE9-16   | yolov10_bmcv.soc  | yolov10s_int8_4b.bmodel | 5.57        | 1.75            | 9.06           | 0.02             |
| SE9-8    | yolov10_opencv.py | yolov10s_fp32_1b.bmodel | 23.86       | 29.78           | 135.99         | 0.86             |
| SE9-8    | yolov10_opencv.py | yolov10s_fp16_1b.bmodel | 23.13       | 29.58           | 40.68          | 0.86             |
| SE9-8    | yolov10_opencv.py | yolov10s_int8_1b.bmodel | 19.17       | 30.22           | 15.20          | 0.85             |
| SE9-8    | yolov10_opencv.py | yolov10s_int8_4b.bmodel | 19.26       | 32.95           | 13.15          | 0.60             |
| SE9-8    | yolov10_bmcv.py   | yolov10s_fp32_1b.bmodel | 4.12        | 4.58            | 131.57         | 0.80             |
| SE9-8    | yolov10_bmcv.py   | yolov10s_fp16_1b.bmodel | 4.11        | 4.58            | 36.15          | 0.81             |
| SE9-8    | yolov10_bmcv.py   | yolov10s_int8_1b.bmodel | 4.08        | 4.56            | 10.70          | 0.80             |
| SE9-8    | yolov10_bmcv.py   | yolov10s_int8_4b.bmodel | 3.90        | 4.26            | 9.24           | 0.55             |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_fp32_1b.bmodel | 5.59        | 1.74            | 130.46         | 0.039            |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_fp16_1b.bmodel | 5.62        | 1.73            | 35.22          | 0.04             |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_int8_1b.bmodel | 5.60        | 1.74            | 9.79           | 0.04             |
| SE9-8    | yolov10_bmcv.soc  | yolov10s_int8_4b.bmodel | 5.40        | 1.65            | 9.00           | 0.02             |
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
    parser.add_argument('--bmodel', type=str, default='yolov10_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov10_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov10_fp32_1b.bmodel_python_test.log')
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
        
