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
|    测试平台  |     测试程序      |             测试模型       |decode_time |preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | -------------------------| ---------  | ------------- | ------------ | --------- |
| SE5-16 | p2pnet_opencv.py | p2pnet_bm1684_fp32_1b.bmodel  | 31.35      | 38.94         | 95.00        | 4.36      |
| SE5-16 | p2pnet_opencv.py | p2pnet_bm1684_int8_1b.bmodel  | 31.43      | 38.78         | 50.79        | 4.35      |
| SE5-16 | p2pnet_opencv.py | p2pnet_bm1684_int8_4b.bmodel  | 31.44      | 42.87         | 15.95        | 4.18      |
| SE5-16 | p2pnet_bmcv.py   | p2pnet_bm1684_fp32_1b.bmodel  | 3.40       | 3.22          | 92.91        | 4.43      |
| SE5-16 | p2pnet_bmcv.py   | p2pnet_bm1684_int8_1b.bmodel  | 3.40       | 3.21          | 48.62        | 4.42      |
| SE5-16 | p2pnet_bmcv.py   | p2pnet_bm1684_int8_4b.bmodel  | 3.08       | 3.61          | 14.09        | 4.26      |
| SE5-16 | p2pnet_bmcv.soc  | p2pnet_bm1684_fp32_1b.bmodel  | 4.414      | 0.759         | 91.985       | 1.358     |
| SE5-16 | p2pnet_bmcv.soc  | p2pnet_bm1684_int8_1b.bmodel  | 4.467      | 0.768         | 47.755       | 1.355     |
| SE5-16 | p2pnet_bmcv.soc  | p2pnet_bm1684_int8_4b.bmodel  | 4.23       | 2.789         | 54.913       | 5.356     |
| SE7-32 | p2pnet_opencv.py | p2pnet_bm1684x_fp32_1b.bmodel | 27.72      | 41.47         | 169.78       | 4.36      |
| SE7-32 | p2pnet_opencv.py | p2pnet_bm1684x_fp16_1b.bmodel | 27.74      | 40.82         | 17.85        | 4.36      |
| SE7-32 | p2pnet_opencv.py | p2pnet_bm1684x_int8_1b.bmodel | 27.72      | 41.65         | 10.47        | 4.36      |
| SE7-32 | p2pnet_opencv.py | p2pnet_bm1684x_int8_4b.bmodel | 27.80      | 44.00         | 9.36         | 4.18      |
| SE7-32 | p2pnet_bmcv.py   | p2pnet_bm1684x_fp32_1b.bmodel | 2.87       | 2.71          | 167.31       | 4.42      |
| SE7-32 | p2pnet_bmcv.py   | p2pnet_bm1684x_fp16_1b.bmodel | 2.88       | 2.72          | 15.37        | 4.43      |
| SE7-32 | p2pnet_bmcv.py   | p2pnet_bm1684x_int8_1b.bmodel | 2.87       | 2.69          | 7.98         | 4.43      |
| SE7-32 | p2pnet_bmcv.py   | p2pnet_bm1684x_int8_4b.bmodel | 2.60       | 3.04          | 7.28         | 4.27      |
| SE7-32 | p2pnet_bmcv.soc  | p2pnet_bm1684x_fp32_1b.bmodel | 3.926      | 0.827         | 166.5        | 1.358     |
| SE7-32 | p2pnet_bmcv.soc  | p2pnet_bm1684x_fp16_1b.bmodel | 3.979      | 0.827         | 14.568       | 1.356     |
| SE7-32 | p2pnet_bmcv.soc  | p2pnet_bm1684x_int8_1b.bmodel | 4.01       | 0.828         | 7.217        | 1.355     |
| SE7-32 | p2pnet_bmcv.soc  | p2pnet_bm1684x_int8_4b.bmodel | 3.859      | 0.794         | 27.76        | 5.381     |
| SE9-16 | p2pnet_opencv.py | p2pnet_bm1688_fp32_1b.bmodel  | 38.52      | 52.18         | 941.72       | 6.11      |
| SE9-16 | p2pnet_opencv.py | p2pnet_bm1688_fp16_1b.bmodel  | 38.50      | 52.08         | 113.67       | 6.15      |
| SE9-16 | p2pnet_opencv.py | p2pnet_bm1688_int8_1b.bmodel  | 38.57      | 51.91         | 32.77        | 6.10      |
| SE9-16 | p2pnet_opencv.py | p2pnet_bm1688_int8_4b.bmodel  | 38.67      | 56.83         | 30.89        | 5.83      |
| SE9-16 | p2pnet_bmcv.py   | p2pnet_bm1688_fp32_1b.bmodel  | 5.32       | 5.91          | 938.84       | 6.27      |
| SE9-16 | p2pnet_bmcv.py   | p2pnet_bm1688_fp16_1b.bmodel  | 5.31       | 5.92          | 110.76       | 6.24      |
| SE9-16 | p2pnet_bmcv.py   | p2pnet_bm1688_int8_1b.bmodel  | 5.31       | 5.91          | 29.79        | 6.23      |
| SE9-16 | p2pnet_bmcv.py   | p2pnet_bm1688_int8_4b.bmodel  | 5.05       | 6.26          | 28.25        | 5.95      |
| SE9-16 | p2pnet_bmcv.soc  | p2pnet_bm1688_fp32_1b.bmodel  | 6.307      | 1.968         | 937.301      | 1.973     |
| SE9-16 | p2pnet_bmcv.soc  | p2pnet_bm1688_fp16_1b.bmodel  | 6.357      | 1.966         | 109.443      | 1.936     |
| SE9-16 | p2pnet_bmcv.soc  | p2pnet_bm1688_int8_1b.bmodel  | 6.456      | 1.965         | 28.535       | 1.934     |
| SE9-16 | p2pnet_bmcv.soc  | p2pnet_bm1688_int8_4b.bmodel  | 6.268      | 1.887         | 110.907      | 1.893     |
| SE9-8  | p2pnet_opencv.py | p2pnet_cv186x_fp32_1b.bmodel  | 39.86      | 53.21         | 942.04       | 6.12      |
| SE9-8  | p2pnet_opencv.py | p2pnet_cv186x_fp16_1b.bmodel  | 38.53      | 52.53         | 113.74       | 6.12      |
| SE9-8  | p2pnet_opencv.py | p2pnet_cv186x_int8_1b.bmodel  | 38.52      | 52.45         | 32.81        | 6.11      |
| SE9-8  | p2pnet_opencv.py | p2pnet_cv186x_int8_4b.bmodel  | 38.66      | 57.69         | 30.72        | 5.85      |
| SE9-8  | p2pnet_bmcv.py   | p2pnet_cv186x_fp32_1b.bmodel  | 9.01       | 5.80          | 939.25       | 6.23      |
| SE9-8  | p2pnet_bmcv.py   | p2pnet_cv186x_fp16_1b.bmodel  | 5.41       | 5.82          | 110.8        | 6.23      |
| SE9-8  | p2pnet_bmcv.py   | p2pnet_cv186x_int8_1b.bmodel  | 5.22       | 5.80          | 29.82        | 6.24      |
| SE9-8  | p2pnet_bmcv.py   | p2pnet_cv186x_int8_4b.bmodel  | 4.88       | 6.19          | 28.18        | 6.01      |
| SE9-8  | p2pnet_bmcv.soc  | p2pnet_cv186x_fp32_1b.bmodel  | 6.237      | 1.977         | 937.621      | 1.972     |
| SE9-8  | p2pnet_bmcv.soc  | p2pnet_cv186x_fp16_1b.bmodel  | 6.358      | 1.976         | 109.408      | 1.935     |
| SE9-8  | p2pnet_bmcv.soc  | p2pnet_cv186x_int8_1b.bmodel  | 6.334      | 1.972         | 28.505       | 1.917     |
| SE9-8  | p2pnet_bmcv.soc  | p2pnet_cv186x_int8_4b.bmodel  | 6.188      | 1.88          | 110.584      | 1.88      |

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
    parser.add_argument('--bmodel', type=str, default='resnet_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='resnet_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_resnet_fp32_1b.bmodel_python_test.log')
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
        
    benchmark_str = "|{:^13}|{:^19}|{:^35}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
