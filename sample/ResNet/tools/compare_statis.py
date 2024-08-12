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
|    测试平台  |     测试程序        |        测试模型       |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | --------------------- | -------- | --------- | --------- | --------- |
|   SE5-16    | resnet_opencv.py  |      resnet50_fp32_1b.bmodel      |      10.95      |      7.70       |      8.86       |      0.30       |
|   SE5-16    | resnet_opencv.py  |      resnet50_int8_1b.bmodel      |      10.12      |      7.66       |      6.39       |      0.30       |
|   SE5-16    | resnet_opencv.py  |      resnet50_int8_4b.bmodel      |      10.14      |      7.74       |      3.19       |      0.11       |
|   SE5-16    |  resnet_bmcv.py   |      resnet50_fp32_1b.bmodel      |      1.71       |      0.96       |      6.84       |      0.25       |
|   SE5-16    |  resnet_bmcv.py   |      resnet50_int8_1b.bmodel      |      1.71       |      0.97       |      4.42       |      0.26       |
|   SE5-16    |  resnet_bmcv.py   |      resnet50_int8_4b.bmodel      |      1.49       |      0.85       |      1.37       |      0.10       |
|   SE5-16    | resnet_opencv.soc |      resnet50_fp32_1b.bmodel      |      1.37       |      5.83       |      6.33       |      0.09       |
|   SE5-16    | resnet_opencv.soc |      resnet50_int8_1b.bmodel      |      1.37       |      5.86       |      3.92       |      0.09       |
|   SE5-16    | resnet_opencv.soc |      resnet50_int8_4b.bmodel      |      1.16       |      5.91       |      1.23       |      0.07       |
|   SE5-16    |  resnet_bmcv.soc  |      resnet50_fp32_1b.bmodel      |      2.54       |      2.51       |      6.31       |      0.11       |
|   SE5-16    |  resnet_bmcv.soc  |      resnet50_int8_1b.bmodel      |      2.48       |      2.50       |      3.89       |      0.11       |
|   SE5-16    |  resnet_bmcv.soc  |      resnet50_int8_4b.bmodel      |      2.44       |      2.44       |      1.22       |      0.10       |
|   SE7-32    | resnet_opencv.py  |      resnet50_fp32_1b.bmodel      |      10.12      |      7.63       |      11.83      |      0.30       |
|   SE7-32    | resnet_opencv.py  |      resnet50_fp16_1b.bmodel      |      10.10      |      7.63       |      4.31       |      0.31       |
|   SE7-32    | resnet_opencv.py  |      resnet50_int8_1b.bmodel      |      10.09      |      7.60       |      3.77       |      0.30       |
|   SE7-32    | resnet_opencv.py  |      resnet50_int8_4b.bmodel      |      9.97       |      7.64       |      3.07       |      0.11       |
|   SE7-32    |  resnet_bmcv.py   |      resnet50_fp32_1b.bmodel      |      1.52       |      0.72       |      9.68       |      0.26       |
|   SE7-32    |  resnet_bmcv.py   |      resnet50_fp16_1b.bmodel      |      1.52       |      0.72       |      2.16       |      0.26       |
|   SE7-32    |  resnet_bmcv.py   |      resnet50_int8_1b.bmodel      |      1.52       |      0.73       |      1.61       |      0.26       |
|   SE7-32    |  resnet_bmcv.py   |      resnet50_int8_4b.bmodel      |      1.32       |      0.62       |      0.96       |      0.10       |
|   SE7-32    | resnet_opencv.soc |      resnet50_fp32_1b.bmodel      |      1.15       |      5.67       |      9.12       |      0.09       |
|   SE7-32    | resnet_opencv.soc |      resnet50_fp16_1b.bmodel      |      1.17       |      5.69       |      1.64       |      0.09       |
|   SE7-32    | resnet_opencv.soc |      resnet50_int8_1b.bmodel      |      1.16       |      5.68       |      1.09       |      0.09       |
|   SE7-32    | resnet_opencv.soc |      resnet50_int8_4b.bmodel      |      0.99       |      5.75       |      0.81       |      0.07       |
|   SE7-32    |  resnet_bmcv.soc  |      resnet50_fp32_1b.bmodel      |      2.18       |      0.45       |      9.12       |      0.11       |
|   SE7-32    |  resnet_bmcv.soc  |      resnet50_fp16_1b.bmodel      |      2.16       |      0.45       |      1.61       |      0.11       |
|   SE7-32    |  resnet_bmcv.soc  |      resnet50_int8_1b.bmodel      |      2.19       |      0.45       |      1.08       |      0.11       |
|   SE7-32    |  resnet_bmcv.soc  |      resnet50_int8_4b.bmodel      |      2.11       |      0.41       |      0.81       |      0.10       |
|   SE9-16    | resnet_opencv.py  |      resnet50_fp32_1b.bmodel      |      13.95      |      10.68      |      49.07      |      0.42       |
|   SE9-16    | resnet_opencv.py  |      resnet50_fp16_1b.bmodel      |      13.02      |      10.65      |      10.93      |      0.43       |
|   SE9-16    | resnet_opencv.py  |      resnet50_int8_1b.bmodel      |      12.94      |      10.64      |      6.13       |      0.42       |
|   SE9-16    | resnet_opencv.py  |      resnet50_int8_4b.bmodel      |      12.80      |      10.68      |      4.86       |      0.15       |
|   SE9-16    |  resnet_bmcv.py   |      resnet50_fp32_1b.bmodel      |      3.17       |      1.71       |      46.33      |      0.38       |
|   SE9-16    |  resnet_bmcv.py   |      resnet50_fp16_1b.bmodel      |      3.07       |      1.71       |      8.17       |      0.38       |
|   SE9-16    |  resnet_bmcv.py   |      resnet50_int8_1b.bmodel      |      3.07       |      1.71       |      3.38       |      0.37       |
|   SE9-16    |  resnet_bmcv.py   |      resnet50_int8_4b.bmodel      |      2.88       |      1.50       |      2.27       |      0.14       |
|   SE9-16    | resnet_opencv.soc |      resnet50_fp32_1b.bmodel      |      2.46       |      7.65       |      45.40      |      0.14       |
|   SE9-16    | resnet_opencv.soc |      resnet50_fp16_1b.bmodel      |      2.42       |      7.60       |      7.36       |      0.13       |
|   SE9-16    | resnet_opencv.soc |      resnet50_int8_1b.bmodel      |      2.43       |      7.65       |      2.57       |      0.13       |
|   SE9-16    | resnet_opencv.soc |      resnet50_int8_4b.bmodel      |      2.11       |      7.74       |      2.06       |      0.10       |
|   SE9-16    |  resnet_bmcv.soc  |      resnet50_fp32_1b.bmodel      |      4.11       |      1.31       |      45.40      |      0.19       |
|   SE9-16    |  resnet_bmcv.soc  |      resnet50_fp16_1b.bmodel      |      3.96       |      1.29       |      7.37       |      0.17       |
|   SE9-16    |  resnet_bmcv.soc  |      resnet50_int8_1b.bmodel      |      3.91       |      1.30       |      2.56       |      0.16       |
|   SE9-16    |  resnet_bmcv.soc  |      resnet50_int8_4b.bmodel      |      3.83       |      1.17       |      2.06       |      0.14       |
|   SE9-16    | resnet_opencv.py  |   resnet50_fp32_1b_2core.bmodel   |      12.97      |      10.66      |      36.99      |      0.42       |
|   SE9-16    | resnet_opencv.py  |   resnet50_fp16_1b_2core.bmodel   |      12.98      |      10.73      |      10.19      |      0.42       |
|   SE9-16    | resnet_opencv.py  |   resnet50_int8_1b_2core.bmodel   |      12.90      |      10.64      |      6.02       |      0.42       |
|   SE9-16    | resnet_opencv.py  |   resnet50_int8_4b_2core.bmodel   |      12.85      |      10.63      |      4.25       |      0.15       |
|   SE9-16    |  resnet_bmcv.py   |   resnet50_fp32_1b_2core.bmodel   |      3.14       |      1.72       |      34.25      |      0.38       |
|   SE9-16    |  resnet_bmcv.py   |   resnet50_fp16_1b_2core.bmodel   |      3.10       |      1.71       |      7.48       |      0.37       |
|   SE9-16    |  resnet_bmcv.py   |   resnet50_int8_1b_2core.bmodel   |      3.18       |      1.73       |      3.29       |      0.38       |
|   SE9-16    |  resnet_bmcv.py   |   resnet50_int8_4b_2core.bmodel   |      2.79       |      1.50       |      1.69       |      0.14       |
|   SE9-16    | resnet_opencv.soc |   resnet50_fp32_1b_2core.bmodel   |      2.46       |      7.60       |      33.40      |      0.14       |
|   SE9-16    | resnet_opencv.soc |   resnet50_fp16_1b_2core.bmodel   |      2.47       |      7.66       |      6.66       |      0.14       |
|   SE9-16    | resnet_opencv.soc |   resnet50_int8_1b_2core.bmodel   |      2.46       |      7.67       |      2.46       |      0.14       |
|   SE9-16    | resnet_opencv.soc |   resnet50_int8_4b_2core.bmodel   |      2.09       |      7.73       |      1.48       |      0.10       |
|   SE9-16    |  resnet_bmcv.soc  |   resnet50_fp32_1b_2core.bmodel   |      4.02       |      1.29       |      33.39      |      0.17       |
|   SE9-16    |  resnet_bmcv.soc  |   resnet50_fp16_1b_2core.bmodel   |      3.97       |      1.31       |      6.64       |      0.17       |
|   SE9-16    |  resnet_bmcv.soc  |   resnet50_int8_1b_2core.bmodel   |      3.99       |      1.29       |      2.46       |      0.17       |
|   SE9-16    |  resnet_bmcv.soc  |   resnet50_int8_4b_2core.bmodel   |      3.80       |      1.20       |      1.48       |      0.14       |
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
        
