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
|   SE7-32    | yolov8_opencv.py  |yolov8s-obb_fp32_1b.bmodel|     129.87      |      65.81      |      86.56      |      40.35      |
|   SE7-32    | yolov8_opencv.py  |yolov8s-obb_fp16_1b.bmodel|     142.89      |      65.00      |      24.94      |      40.66      |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s-obb_fp32_1b.bmodel|      50.94      |      10.12      |      76.71      |      8.66       |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s-obb_fp16_1b.bmodel|      36.93      |      10.12      |      15.10      |      8.65       |
|   SE9-16    | yolov8_opencv.py  |yolov8s-obb_fp32_1b.bmodel|     164.73      |      87.15      |     439.43      |      44.46      |
|   SE9-16    | yolov8_opencv.py  |yolov8s-obb_fp16_1b.bmodel|     163.24      |      86.07      |     100.31      |      43.88      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-obb_fp32_1b.bmodel|      40.23      |      28.32      |     427.11      |      12.21      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-obb_fp16_1b.bmodel|      30.13      |      28.32      |      88.33      |      12.05      |
|   SE9-16    | yolov8_opencv.py  |yolov8s-obb_fp32_1b_2core.bmodel|     159.57      |      86.44      |     236.79      |      45.33      |
|   SE9-16    | yolov8_opencv.py  |yolov8s-obb_fp16_1b_2core.bmodel|     156.49      |      84.28      |      61.93      |      43.15      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-obb_fp32_1b_2core.bmodel|      52.33      |      28.32      |     224.74      |      12.07      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-obb_fp16_1b_2core.bmodel|      30.08      |      28.32      |      49.68      |      12.17      |
|    SE9-8    | yolov8_opencv.py  |yolov8s-obb_fp32_1b.bmodel|     169.42      |      86.45      |     448.55      |      48.90      |
|    SE9-8    | yolov8_opencv.py  |yolov8s-obb_fp16_1b.bmodel|     169.59      |      86.64      |     103.97      |      49.06      |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s-obb_fp32_1b.bmodel|      46.39      |      29.67      |     436.68      |      12.16      |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s-obb_fp16_1b.bmodel|      57.61      |      29.67      |      92.13      |      12.02      |
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
    parser.add_argument('--bmodel', type=str, default='yolov8s-obb_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov8_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/opencv_yolov8s-obb_fp32_1b.bmodel_python_test.log')
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
        threhold = 0.2
        if key == "decode":
            threhold = 0.5
        if key == "postprocess":
            threhold = 0.4
        if abs(statis - extracted_data[key]) / statis > threhold:
            print("{:} time, diff ratio > {:}".format(key, str(threhold)))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^25}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
