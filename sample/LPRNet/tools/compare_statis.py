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
|    测试平台  |     测试程序      |      测试模型          |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | --------------------- | --------   | ---------     | ---------    | ---------  |
| SE5-16      | lprnet_opencv.py | lprnet_fp32_1b.bmodel | 0.5        | 0.14          | 2.33         | 0.13       |
| SE5-16      | lprnet_opencv.py | lprnet_int8_1b.bmodel | 0.48       | 0.15          | 1.35         | 0.14       |
| SE5-16      | lprnet_opencv.py | lprnet_int8_4b.bmodel | 0.32       | 0.08          | 0.45         | 0.06       |
| SE5-16      | lprnet_bmcv.py   | lprnet_fp32_1b.bmodel | 0.6        | 0.33          | 2.01         | 0.15       |
| SE5-16      | lprnet_bmcv.py   | lprnet_int8_1b.bmodel | 0.66       | 0.35          | 1.07         | 0.15       |
| SE5-16      | lprnet_bmcv.py   | lprnet_int8_4b.bmodel | 0.45       | 0.25          | 0.33         | 0.06       |
| SE5-16      | lprnet_opencv.soc| lprnet_fp32_1b.bmodel | 0.537      | 0.211         | 1.642        | 0.072      |
| SE5-16      | lprnet_opencv.soc| lprnet_int8_1b.bmodel | 0.696      | 0.277         | 0.656        | 0.077      |
| SE5-16      | lprnet_opencv.soc| lprnet_int8_4b.bmodel | 0.46       | 0.695         | 0.232        | 0.048      |
| SE5-16      | lprnet_bmcv.soc  | lprnet_fp32_1b.bmodel | 1.629      | 0.283         | 1.664        | 0.08       |
| SE5-16      | lprnet_bmcv.soc  | lprnet_int8_1b.bmodel | 1.627      | 0.285         | 0.661        | 0.075      |
| SE5-16      | lprnet_bmcv.soc  | lprnet_int8_4b.bmodel | 1.184      | 0.647         | 0.233        | 0.047      |
| SE7-32      | lprnet_opencv.py  |       lprnet_fp32_1b.bmodel       |      0.39       |      0.11       |      1.50       |      0.11       |
| SE7-32      | lprnet_opencv.py  |       lprnet_fp16_1b.bmodel       |      0.37       |      0.10       |      1.16       |      0.10       |
| SE7-32      | lprnet_opencv.py  |       lprnet_int8_1b.bmodel       |      0.37       |      0.11       |      1.09       |      0.10       |
| SE7-32      | lprnet_opencv.py  |       lprnet_int8_4b.bmodel       |      0.28       |      0.08       |      0.49       |      0.06       |
| SE7-32      |  lprnet_bmcv.py   |       lprnet_fp32_1b.bmodel       |      0.74       |      0.31       |      1.32       |      0.13       |
| SE7-32      |  lprnet_bmcv.py   |       lprnet_fp16_1b.bmodel       |      0.71       |      0.31       |      0.99       |      0.13       |
| SE7-32      |  lprnet_bmcv.py   |       lprnet_int8_1b.bmodel       |      0.73       |      0.31       |      0.93       |      0.13       |
| SE7-32      |  lprnet_bmcv.py   |       lprnet_int8_4b.bmodel       |      0.53       |      0.26       |      0.38       |      0.06       |
| SE7-32      | lprnet_opencv.soc |       lprnet_fp32_1b.bmodel       |      0.34       |      0.15       |      0.83       |      0.05       |
| SE7-32      | lprnet_opencv.soc |       lprnet_fp16_1b.bmodel       |      0.35       |      0.15       |      0.53       |      0.05       |
| SE7-32      | lprnet_opencv.soc |       lprnet_int8_1b.bmodel       |      0.34       |      0.15       |      0.45       |      0.05       |
| SE7-32      | lprnet_opencv.soc |       lprnet_int8_4b.bmodel       |      0.35       |      0.14       |      0.25       |      0.04       |
| SE7-32      |  lprnet_bmcv.soc  |       lprnet_fp32_1b.bmodel       |      0.63       |      0.10       |      0.83       |      0.05       |
| SE7-32      |  lprnet_bmcv.soc  |       lprnet_fp16_1b.bmodel       |      0.62       |      0.10       |      0.53       |      0.05       |
| SE7-32      |  lprnet_bmcv.soc  |       lprnet_int8_1b.bmodel       |      0.62       |      0.10       |      0.45       |      0.05       |
| SE7-32      |  lprnet_bmcv.soc  |       lprnet_int8_4b.bmodel       |      0.61       |      0.08       |      0.25       |      0.04       |
| SE9-16      | lprnet_opencv.py | lprnet_fp32_1b.bmodel | 0.54       |  0.15         | 3.06         | 0.15       |
| SE9-16      | lprnet_opencv.py | lprnet_fp16_1b.bmodel | 0.54       |  0.16         | 1.71         | 0.15       |
| SE9-16      | lprnet_opencv.py | lprnet_int8_1b.bmodel | 0.54       |  0.16         | 1.38         | 0.15       |
| SE9-16      | lprnet_opencv.py | lprnet_int8_4b.bmodel | 0.39       |  0.10         | 0.65         | 0.08       |
| SE9-16      | lprnet_bmcv.py   | lprnet_fp32_1b.bmodel | 2.26       |  0.88         | 3.01         | 0.21       |
| SE9-16      | lprnet_bmcv.py   | lprnet_fp16_1b.bmodel | 2.28       |  0.88         | 1.58         | 0.21       |
| SE9-16      | lprnet_bmcv.py   | lprnet_int8_1b.bmodel | 2.26       |  0.88         | 1.29         | 0.21       |
| SE9-16      | lprnet_bmcv.py   | lprnet_int8_4b.bmodel | 1.84       |  0.71         | 0.52         | 0.09       |
| SE9-16      | lprnet_opencv.soc| lprnet_fp32_1b.bmodel | 1.53       |  1.11         | 2.31         | 0.10       |
| SE9-16      | lprnet_opencv.soc| lprnet_fp16_1b.bmodel | 1.54       |  1.12         | 0.88         | 0.10       |
| SE9-16      | lprnet_opencv.soc| lprnet_int8_1b.bmodel | 1.51       |  1.12         | 0.57         | 0.10       |
| SE9-16      | lprnet_opencv.soc| lprnet_int8_4b.bmodel | 1.33       |  1.04         | 0.36         | 0.69       |
| SE9-16      | lprnet_bmcv.soc  | lprnet_fp32_1b.bmodel | 2.17       |  0.69         | 2.34         | 0.09       |
| SE9-16      | lprnet_bmcv.soc  | lprnet_fp16_1b.bmodel | 2.17       |  0.68         | 0.89         | 0.09       |
| SE9-16      | lprnet_bmcv.soc  | lprnet_int8_1b.bmodel | 2.12       |  0.68         | 0.59         | 0.09       |
| SE9-16      | lprnet_bmcv.soc  | lprnet_int8_4b.bmodel | 1.84       |  0.59         | 0.36         | 0.07       |
|    SE9-8    | lprnet_opencv.py  |       lprnet_fp32_1b.bmodel       |      0.54       |      0.16       |      3.39       |      0.15       |
|    SE9-8    | lprnet_opencv.py  |       lprnet_fp16_1b.bmodel       |      0.54       |      0.15       |      1.91       |      0.15       |
|    SE9-8    | lprnet_opencv.py  |       lprnet_int8_1b.bmodel       |      0.53       |      0.15       |      1.49       |      0.15       |
|    SE9-8    | lprnet_opencv.py  |       lprnet_int8_4b.bmodel       |      0.38       |      0.11       |      0.76       |      0.08       |
|    SE9-8    |  lprnet_bmcv.py   |       lprnet_fp32_1b.bmodel       |      1.60       |      0.73       |      3.25       |      0.20       |
|    SE9-8    |  lprnet_bmcv.py   |       lprnet_fp16_1b.bmodel       |      1.57       |      0.71       |      1.73       |      0.20       |
|    SE9-8    |  lprnet_bmcv.py   |       lprnet_int8_1b.bmodel       |      1.65       |      0.73       |      1.35       |      0.20       |
|    SE9-8    |  lprnet_bmcv.py   |       lprnet_int8_4b.bmodel       |      1.17       |      0.56       |      0.62       |      0.09       |
|    SE9-8    | lprnet_opencv.soc |       lprnet_fp32_1b.bmodel       |      0.82       |      0.37       |      2.47       |      0.09       |
|    SE9-8    | lprnet_opencv.soc |       lprnet_fp16_1b.bmodel       |      0.79       |      0.37       |      0.98       |      0.09       |
|    SE9-8    | lprnet_opencv.soc |       lprnet_int8_1b.bmodel       |      0.80       |      0.37       |      0.56       |      0.08       |
|    SE9-8    | lprnet_opencv.soc |       lprnet_int8_4b.bmodel       |      0.71       |      0.31       |      0.42       |      0.07       |
|    SE9-8    |  lprnet_bmcv.soc  |       lprnet_fp32_1b.bmodel       |      1.26       |      0.44       |      2.47       |      0.08       |
|    SE9-8    |  lprnet_bmcv.soc  |       lprnet_fp16_1b.bmodel       |      1.25       |      0.45       |      0.98       |      0.08       |
|    SE9-8    |  lprnet_bmcv.soc  |       lprnet_int8_1b.bmodel       |      1.20       |      0.45       |      0.56       |      0.08       |
|    SE9-8    |  lprnet_bmcv.soc  |       lprnet_int8_4b.bmodel       |      1.10       |      0.36       |      0.42       |      0.06       |
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
    parser.add_argument('--bmodel', type=str, default='lprnet_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='lprnet_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_lprnet_fp32_1b.bmodel_python_test.log')
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
            if multiprocessing.cpu_count() == 6:
                platform = "SE9-8"
        elif args.target == "CV186X":
            platform = "SE9-8"
    else:
        platform = args.target + " SoC" if args.platform == "soc" else args.target + " PCIe"
    min_width = 17
    
    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^13}|{:^19}|{:^35}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
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
        
    benchmark_str = "|{:^13}|{:^19}|{:^35}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        