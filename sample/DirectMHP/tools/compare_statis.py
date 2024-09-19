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
|    测试平台  |     测试程序      |             测试模型                |decode_time    |preprocess_time  |inference_time   |postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | --------      | ---------       | -------------    | --------------- |
|   SE7-32    |directmhp_opencv.py|     directmhp_fp32_1b.bmodel      |      35.46      |     103.07      |     102.92      |      5.95       |
|   SE7-32    |directmhp_opencv.py|     directmhp_fp16_1b.bmodel      |      40.34      |     101.64      |      41.95      |      5.96       |
|   SE7-32    | directmhp_bmcv.py |     directmhp_fp32_1b.bmodel      |      8.02       |      8.58       |      88.22      |      5.97       |
|   SE7-32    | directmhp_bmcv.py |     directmhp_fp16_1b.bmodel      |      6.56       |      8.68       |      27.41      |      5.97       |
|   SE7-32    |directmhp_bmcv.soc |     directmhp_fp32_1b.bmodel      |      9.95       |      4.06       |      84.32      |      2.74       |
|   SE7-32    |directmhp_bmcv.soc |     directmhp_fp16_1b.bmodel      |      9.90       |      4.06       |      23.43      |      2.74       |
|   SE9-16    |directmhp_opencv.py|     directmhp_fp32_1b.bmodel      |      54.05      |     134.03      |     430.60      |      7.30       |
|   SE9-16    |directmhp_opencv.py|     directmhp_fp16_1b.bmodel      |      51.51      |     137.22      |     131.09      |      7.31       |
|   SE9-16    | directmhp_bmcv.py |     directmhp_fp32_1b.bmodel      |      15.49      |      17.67      |     413.37      |      7.35       |
|   SE9-16    | directmhp_bmcv.py |     directmhp_fp16_1b.bmodel      |      11.31      |      17.65      |     112.64      |      7.35       |
|   SE9-16    |directmhp_bmcv.soc |     directmhp_fp32_1b.bmodel      |      15.30      |      7.15       |     407.72      |      3.32       |
|   SE9-16    |directmhp_bmcv.soc |     directmhp_fp16_1b.bmodel      |      15.17      |      7.15       |     107.29      |      3.28       |
|   SE9-16    |directmhp_opencv.py|  directmhp_fp32_1b_2core.bmodel   |      46.36      |     136.42      |     238.36      |      7.30       |
|   SE9-16    |directmhp_opencv.py|  directmhp_fp16_1b_2core.bmodel   |      49.97      |     134.85      |      86.90      |      7.30       |
|   SE9-16    | directmhp_bmcv.py |  directmhp_fp32_1b_2core.bmodel   |      12.88      |      17.63      |     221.12      |      7.37       |
|   SE9-16    | directmhp_bmcv.py |  directmhp_fp16_1b_2core.bmodel   |      11.33      |      17.63      |      68.06      |      7.36       |
|   SE9-16    |directmhp_bmcv.soc |  directmhp_fp32_1b_2core.bmodel   |      15.45      |      7.14       |     215.53      |      3.31       |
|   SE9-16    |directmhp_bmcv.soc |  directmhp_fp16_1b_2core.bmodel   |      15.13      |      7.15       |      62.87      |      3.27       |
|   SE9-8     |directmhp_opencv.py|     directmhp_fp32_1b.bmodel      |      50.38      |     139.18      |     442.47      |      7.37       |
|   SE9-8     |directmhp_opencv.py|     directmhp_fp16_1b.bmodel      |      58.76      |     140.77      |     137.92      |      7.41       |
|   SE9-8     | directmhp_bmcv.py |     directmhp_fp32_1b.bmodel      |      26.04      |      17.28      |     425.40      |      7.47       |
|   SE9-8     | directmhp_bmcv.py |     directmhp_fp16_1b.bmodel      |      24.99      |      17.27      |     120.67      |      7.49       |
|   SE9-8     |directmhp_bmcv.soc |     directmhp_fp32_1b.bmodel      |      17.18      |      7.16       |     420.06      |      3.45       |
|   SE9-8     |directmhp_bmcv.soc |     directmhp_fp16_1b.bmodel      |      15.77      |      7.16       |     115.50      |      3.42       |
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
    parser.add_argument('--bmodel', type=str, default='directmhp_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='directmhp_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_directmhp_fp32_1b.bmodel_python_test.log')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    args = argsparser()
    benchmark_path = current_dir + "/benchmark.txt"
        
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

    if args.platform == "soc":
        if args.target == "BM1684X":
            platform = "SE7-32"
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
        if key == "decode" and args.program == "directmhp_opencv.py":
            threhold = 0.4
        if key == "postprocess":
            threhold = 0.4
        if abs(statis - extracted_data[key]) / statis > threhold:
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
        
