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
|   SE5-16    | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |      9.96       |      21.93      |      32.12      |      5.04       |
|   SE5-16    | yolov11_opencv.py | yolov11s_int8_1b.bmodel |      6.74       |      21.51      |      23.76      |      4.99       |
|   SE5-16    | yolov11_opencv.py | yolov11s_int8_4b.bmodel |      6.89       |      23.83      |      16.13      |      4.94       |
|   SE5-16    |  yolov11_bmcv.py  | yolov11s_fp32_1b.bmodel |      3.63       |      2.76       |      29.34      |      4.95       |
|   SE5-16    |  yolov11_bmcv.py  | yolov11s_int8_1b.bmodel |      3.62       |      2.75       |      20.94      |      4.91       |
|   SE5-16    |  yolov11_bmcv.py  | yolov11s_int8_4b.bmodel |      3.47       |      2.59       |      13.10      |      4.35       |
|   SE5-16    | yolov11_bmcv.soc  | yolov11s_fp32_1b.bmodel |      4.91       |      1.56       |      26.93      |      8.52       |
|   SE5-16    | yolov11_bmcv.soc  | yolov11s_int8_1b.bmodel |      4.90       |      1.55       |      18.59      |      8.53       |
|   SE5-16    | yolov11_bmcv.soc  | yolov11s_int8_4b.bmodel |      4.75       |      1.49       |      11.18      |      8.49       |
|   SE5-16    | yolov11_bmcv.soc  |yolov11s_opt_fp32_1b.bmodel|      4.89       |      1.55       |      27.16      |      2.60       |
|   SE5-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_1b.bmodel|      4.90       |      1.56       |      18.82      |      2.60       |
|   SE5-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_4b.bmodel|      4.74       |      1.49       |      11.40      |      2.67       |
|   SE7-32    | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |      6.80       |      22.42      |      30.43      |      5.40       |
|   SE7-32    | yolov11_opencv.py | yolov11s_fp16_1b.bmodel |      6.82       |      22.88      |      11.69      |      5.39       |
|   SE7-32    | yolov11_opencv.py | yolov11s_int8_1b.bmodel |      6.81       |      22.52      |      9.19       |      5.33       |
|   SE7-32    | yolov11_opencv.py | yolov11s_int8_4b.bmodel |      6.82       |      25.31      |      8.73       |      5.39       |
|   SE7-32    |  yolov11_bmcv.py  | yolov11s_fp32_1b.bmodel |      3.18       |      2.34       |      27.19      |      5.46       |
|   SE7-32    |  yolov11_bmcv.py  | yolov11s_fp16_1b.bmodel |      3.15       |      2.34       |      8.31       |      5.52       |
|   SE7-32    |  yolov11_bmcv.py  | yolov11s_int8_1b.bmodel |      3.15       |      2.33       |      5.86       |      5.41       |
|   SE7-32    |  yolov11_bmcv.py  | yolov11s_int8_4b.bmodel |      2.99       |      2.15       |      5.18       |      4.81       |
|   SE7-32    | yolov11_bmcv.soc  | yolov11s_fp32_1b.bmodel |      4.36       |      0.75       |      24.48      |      8.93       |
|   SE7-32    | yolov11_bmcv.soc  | yolov11s_fp16_1b.bmodel |      4.37       |      0.75       |      5.68       |      8.96       |
|   SE7-32    | yolov11_bmcv.soc  | yolov11s_int8_1b.bmodel |      4.37       |      0.75       |      3.21       |      8.96       |
|   SE7-32    | yolov11_bmcv.soc  | yolov11s_int8_4b.bmodel |      4.20       |      0.72       |      3.01       |      8.94       |
|   SE7-32    | yolov11_bmcv.soc  |yolov11s_opt_fp32_1b.bmodel|      4.35       |      0.75       |      24.63      |      2.62       |
|   SE7-32    | yolov11_bmcv.soc  |yolov11s_opt_fp16_1b.bmodel|      4.35       |      0.75       |      5.84       |      2.62       |
|   SE7-32    | yolov11_bmcv.soc  |yolov11s_opt_int8_1b.bmodel|      4.39       |      0.75       |      3.40       |      2.65       |
|   SE7-32    | yolov11_bmcv.soc  |yolov11s_opt_int8_4b.bmodel|      4.18       |      0.72       |      3.17       |      2.69       |
|   SE9-16    | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |      16.62      |      30.03      |     138.75      |      6.94       |
|   SE9-16    | yolov11_opencv.py | yolov11s_fp16_1b.bmodel |      9.48       |      29.50      |      40.82      |      6.93       |
|   SE9-16    | yolov11_opencv.py | yolov11s_int8_1b.bmodel |      9.45       |      29.94      |      15.80      |      6.81       |
|   SE9-16    | yolov11_opencv.py | yolov11s_int8_4b.bmodel |      9.47       |      33.37      |      14.88      |      7.36       |
|   SE9-16    |  yolov11_bmcv.py  | yolov11s_fp32_1b.bmodel |      4.02       |      4.44       |     134.92      |      7.14       |
|   SE9-16    |  yolov11_bmcv.py  | yolov11s_fp16_1b.bmodel |      3.99       |      4.40       |      37.02      |      7.01       |
|   SE9-16    |  yolov11_bmcv.py  | yolov11s_int8_1b.bmodel |      3.98       |      4.39       |      11.62      |      6.90       |
|   SE9-16    |  yolov11_bmcv.py  | yolov11s_int8_4b.bmodel |      3.81       |      4.11       |      10.44      |      6.08       |
|   SE9-16    | yolov11_bmcv.soc  | yolov11s_fp32_1b.bmodel |      5.50       |      1.73       |     131.40      |      12.15      |
|   SE9-16    | yolov11_bmcv.soc  | yolov11s_fp16_1b.bmodel |      5.54       |      1.72       |      33.48      |      12.14      |
|   SE9-16    | yolov11_bmcv.soc  | yolov11s_int8_1b.bmodel |      5.53       |      1.72       |      8.10       |      12.16      |
|   SE9-16    | yolov11_bmcv.soc  | yolov11s_int8_4b.bmodel |      5.37       |      1.64       |      7.74       |      12.09      |
|   SE9-16    | yolov11_opencv.py |yolov11s_fp32_1b_2core.bmodel|      9.46       |      29.96      |      77.64      |      6.93       |
|   SE9-16    | yolov11_opencv.py |yolov11s_fp16_1b_2core.bmodel|      9.43       |      29.84      |      26.66      |      6.93       |
|   SE9-16    | yolov11_opencv.py |yolov11s_int8_1b_2core.bmodel|      9.39       |      29.42      |      13.73      |      6.81       |
|   SE9-16    | yolov11_opencv.py |yolov11s_int8_4b_2core.bmodel|      9.52       |      33.06      |      12.35      |      6.98       |
|   SE9-16    |  yolov11_bmcv.py  |yolov11s_fp32_1b_2core.bmodel|      4.02       |      4.41       |      73.46      |      7.02       |
|   SE9-16    |  yolov11_bmcv.py  |yolov11s_fp16_1b_2core.bmodel|      3.98       |      4.42       |      22.77      |      7.00       |
|   SE9-16    |  yolov11_bmcv.py  |yolov11s_int8_1b_2core.bmodel|      3.99       |      4.40       |      9.86       |      6.91       |
|   SE9-16    |  yolov11_bmcv.py  |yolov11s_int8_4b_2core.bmodel|      3.81       |      4.09       |      7.92       |      6.05       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_fp32_1b_2core.bmodel|      5.55       |      1.72       |      70.05      |      12.15      |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_fp16_1b_2core.bmodel|      5.56       |      1.72       |      19.32      |      12.15      |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_int8_1b_2core.bmodel|      5.54       |      1.72       |      6.44       |      12.16      |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_int8_4b_2core.bmodel|      5.34       |      1.64       |      5.21       |      12.10      |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_fp32_1b.bmodel|      5.53       |      1.73       |     131.69      |      3.68       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_fp16_1b.bmodel|      5.53       |      1.73       |      33.58      |      3.66       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_1b.bmodel|      5.52       |      1.73       |      8.21       |      3.78       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_4b.bmodel|      5.35       |      1.64       |      7.96       |      3.69       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_fp32_1b_2core.bmodel|      5.58       |      1.72       |      70.34      |      4.04       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_fp16_1b_2core.bmodel|      5.53       |      1.72       |      19.42      |      3.66       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_1b_2core.bmodel|      5.57       |      1.73       |      6.55       |      3.66       |
|   SE9-16    | yolov11_bmcv.soc  |yolov11s_opt_int8_4b_2core.bmodel|      5.38       |      1.64       |      5.42       |      3.69       |
|    SE9-8    | yolov11_opencv.py | yolov11s_fp32_1b.bmodel |      9.48       |      30.13      |     138.92      |      7.01       |
|    SE9-8    | yolov11_opencv.py | yolov11s_fp16_1b.bmodel |      9.44       |      30.14      |      40.84      |      7.02       |
|    SE9-8    | yolov11_opencv.py | yolov11s_int8_1b.bmodel |      9.44       |      29.67      |      15.64      |      6.88       |
|    SE9-8    | yolov11_opencv.py | yolov11s_int8_4b.bmodel |      9.50       |      33.73      |      15.01      |      7.50       |
|    SE9-8    |  yolov11_bmcv.py  | yolov11s_fp32_1b.bmodel |      4.08       |      4.44       |     135.05      |      7.09       |
|    SE9-8    |  yolov11_bmcv.py  | yolov11s_fp16_1b.bmodel |      4.04       |      4.41       |      36.88      |      7.07       |
|    SE9-8    |  yolov11_bmcv.py  | yolov11s_int8_1b.bmodel |      4.04       |      4.40       |      11.43      |      6.96       |
|    SE9-8    |  yolov11_bmcv.py  | yolov11s_int8_4b.bmodel |      3.85       |      4.11       |      10.41      |      6.32       |
|    SE9-8    | yolov11_bmcv.soc  | yolov11s_fp32_1b.bmodel |      5.54       |      1.73       |     131.39      |      12.22      |
|    SE9-8    | yolov11_bmcv.soc  | yolov11s_fp16_1b.bmodel |      5.61       |      1.73       |      33.44      |      12.22      |
|    SE9-8    | yolov11_bmcv.soc  | yolov11s_int8_1b.bmodel |      5.59       |      1.73       |      8.04       |      12.23      |
|    SE9-8    | yolov11_bmcv.soc  | yolov11s_int8_4b.bmodel |      5.39       |      1.64       |      7.70       |      12.16      |
|    SE9-8    | yolov11_bmcv.soc  |yolov11s_opt_fp32_1b.bmodel|      5.57       |      1.73       |     131.68      |      3.69       |
|    SE9-8    | yolov11_bmcv.soc  |yolov11s_opt_fp16_1b.bmodel|      5.62       |      1.73       |      33.53      |      3.66       |
|    SE9-8    | yolov11_bmcv.soc  |yolov11s_opt_int8_1b.bmodel|      5.63       |      1.72       |      8.14       |      3.68       |
|    SE9-8    | yolov11_bmcv.soc  |yolov11s_opt_int8_4b.bmodel|      5.43       |      1.64       |      7.93       |      3.71       |
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
    parser.add_argument('--bmodel', type=str, default='yolov11_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov11_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov11_fp32_1b.bmodel_python_test.log')
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
        
