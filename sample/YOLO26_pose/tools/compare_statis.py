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
| 测试平台 | 测试程序             | 测试模型                | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | -------------------- | ----------------------- | ----------- | --------------- | -------------- | ---------------- |
|   SE7-32   |yolo26_pose_opencv.py |     yolo26s-pose_fp32_1b.bmodel     |     7.61      |     27.64     |     33.55     |     1.67      |
|   SE7-32   | yolo26_pose_bmcv.py  |     yolo26s-pose_fp32_1b.bmodel     |     2.83      |     2.02      |     30.18     |     1.62      |
|   SE7-32   | yolo26_pose_bmcv.soc |     yolo26s-pose_fp32_1b.bmodel     |     2.57      |     1.37      |     29.85     |     0.07      |
|   SE7-32   |yolo26_pose_opencv.py |     yolo26s-pose_fp16_1b.bmodel     |     7.20      |     27.36     |     10.61     |     1.65      |
|   SE7-32   | yolo26_pose_bmcv.py  |     yolo26s-pose_fp16_1b.bmodel     |     2.84      |     2.02      |     7.21      |     1.62      |
|   SE7-32   | yolo26_pose_bmcv.soc |     yolo26s-pose_fp16_1b.bmodel     |     2.57      |     1.37      |     6.87      |     0.07      |
|   SE7-32   |yolo26_pose_opencv.py |     yolo26s-pose_int8_1b.bmodel     |     7.18      |     26.86     |     7.70      |     1.63      |
|   SE7-32   | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_1b.bmodel     |     2.82      |     2.02      |     4.31      |     1.60      |
|   SE7-32   | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_1b.bmodel     |     2.56      |     1.36      |     3.97      |     0.07      |
|   SE7-32   |yolo26_pose_opencv.py |     yolo26s-pose_int8_4b.bmodel     |     6.95      |     25.53     |     6.95      |     1.49      |
|   SE7-32   | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_4b.bmodel     |     2.67      |     1.92      |     3.81      |     1.48      |
|   SE7-32   | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_4b.bmodel     |     2.49      |     1.32      |     3.72      |     0.06      |
|   SE9-16   |yolo26_pose_opencv.py |     yolo26s-pose_fp32_1b.bmodel     |     9.34      |     36.71     |    154.06     |     1.15      |
|   SE9-16   | yolo26_pose_bmcv.py  |     yolo26s-pose_fp32_1b.bmodel     |     3.72      |     3.96      |    149.89     |     1.13      |
|   SE9-16   | yolo26_pose_bmcv.soc |     yolo26s-pose_fp32_1b.bmodel     |     3.30      |     2.61      |    149.37     |     0.12      |
|   SE9-16   |yolo26_pose_opencv.py |     yolo26s-pose_fp16_1b.bmodel     |     9.30      |     35.39     |     42.81     |     1.14      |
|   SE9-16   | yolo26_pose_bmcv.py  |     yolo26s-pose_fp16_1b.bmodel     |     3.73      |     3.96      |     38.58     |     1.13      |
|   SE9-16   | yolo26_pose_bmcv.soc |     yolo26s-pose_fp16_1b.bmodel     |     3.33      |     2.59      |     38.12     |     0.12      |
|   SE9-16   |yolo26_pose_opencv.py |     yolo26s-pose_int8_1b.bmodel     |     9.29      |     35.83     |     14.46     |     1.13      |
|   SE9-16   | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_1b.bmodel     |     3.73      |     3.97      |     10.20     |     1.11      |
|   SE9-16   | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_1b.bmodel     |     3.30      |     2.59      |     9.74      |     0.11      |
|   SE9-16   |yolo26_pose_opencv.py |     yolo26s-pose_int8_4b.bmodel     |     9.28      |     34.09     |     13.05     |     0.98      |
|   SE9-16   | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_4b.bmodel     |     3.47      |     3.76      |     9.09      |     0.97      |
|   SE9-16   | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_4b.bmodel     |     3.12      |     2.14      |     8.92      |     0.09      |
|   SE9-16   |yolo26_pose_opencv.py |  yolo26s-pose_int8_1b_2core.bmodel  |     9.32      |     37.48     |     13.07     |     1.13      |
|   SE9-16   | yolo26_pose_bmcv.py  |  yolo26s-pose_int8_1b_2core.bmodel  |     3.72      |     3.97      |     8.82      |     1.10      |
|   SE9-16   | yolo26_pose_bmcv.soc |  yolo26s-pose_int8_1b_2core.bmodel  |     3.30      |     2.60      |     8.36      |     0.11      |
|   SE9-16   |yolo26_pose_opencv.py |  yolo26s-pose_int8_4b_2core.bmodel  |     9.29      |     33.43     |     9.83      |     0.98      |
|   SE9-16   | yolo26_pose_bmcv.py  |  yolo26s-pose_int8_4b_2core.bmodel  |     3.47      |     3.76      |     5.86      |     0.97      |
|   SE9-16   | yolo26_pose_bmcv.soc |  yolo26s-pose_int8_4b_2core.bmodel  |     3.11      |     2.14      |     5.71      |     0.09      |
|   SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_fp32_1b.bmodel     |     5.98      |     36.50     |    154.08     |     2.46      |
|   SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp32_1b.bmodel     |     3.80      |     3.97      |    149.86     |     2.42      |
|   SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp32_1b.bmodel     |     3.45      |     2.61      |    149.37     |     0.12      |
|   SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_fp16_1b.bmodel     |     4.50      |     36.57     |     42.82     |     2.44      |
|   SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp16_1b.bmodel     |     3.82      |     3.97      |     38.55     |     2.43      |
|   SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp16_1b.bmodel     |     3.44      |     2.60      |     38.12     |     0.11      |
|   SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_int8_1b.bmodel     |     4.47      |     36.60     |     14.47     |     2.42      |
|   SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_1b.bmodel     |     3.96      |     3.97      |     10.17     |     2.38      |
|   SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_1b.bmodel     |     3.38      |     2.60      |     9.74      |     0.11      |
|   SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_int8_4b.bmodel     |     4.26      |     34.53     |     13.06     |     2.25      |
|   SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_4b.bmodel     |     3.50      |     3.75      |     9.07      |     2.20      |
|   SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_4b.bmodel     |     3.12      |     2.15      |     8.90      |     0.09      |
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
    parser.add_argument('--bmodel', type=str, default='yolo26s-pose_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolo26_pose_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolo26s-pose_fp32_1b.bmodel_python_test.log')
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
        threhold = 0.2
        if key == "decode":
            threhold = 0.5
        if key == "postprocess":
            threhold = 0.4
        if statis < extracted_data[key] and abs(statis - extracted_data[key]) / statis > threhold:
            print("{:} time, diff ratio > {:}".format(key, str(threhold)))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^35}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
