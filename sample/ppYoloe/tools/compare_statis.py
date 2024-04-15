#===----------------------------------------------------------------------===#
#
# Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
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
| ----------- | ---------------- | ----------------------------------- | --------      | ---------       | ---------        | --------- |
|   SE5-16    | ppyoloe_opencv.py |      ppyoloe_fp32_1b.bmodel       |      15.02      |      43.29      |      45.16      |      12.46      |
|   SE5-16    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      3.76       |      3.63       |      34.07      |      12.64      |
|   SE5-16    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      4.84       |      1.08       |      30.71      |      8.57       |
|   SE5-16    | ppyoloe_sail.soc  |      ppyoloe_fp32_1b.bmodel       |      3.21       |      4.14       |      31.11      |      7.99       |
|   SE7-32    | ppyoloe_opencv.py |      ppyoloe_fp32_1b.bmodel       |      16.16      |      40.55      |      44.29      |      12.95      |
|   SE7-32    | ppyoloe_opencv.py |      ppyoloe_fp16_1b.bmodel       |      15.13      |      40.54      |      23.72      |      12.88      |
|   SE7-32    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      3.30       |      2.81       |      30.95      |      13.42      |
|   SE7-32    |  ppyoloe_bmcv.py  |      ppyoloe_fp16_1b.bmodel       |      3.26       |      2.79       |      10.40      |      13.44      |
|   SE7-32    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      4.34       |      0.99       |      27.35      |      8.70       |
|   SE7-32    | ppyoloe_bmcv.soc  |      ppyoloe_fp16_1b.bmodel       |      4.35       |      0.99       |      6.79       |      8.72       |
|   SE7-32    | ppyoloe_sail.soc  |      ppyoloe_fp32_1b.bmodel       |      2.71       |      3.33       |      27.76      |      8.10       |
|   SE7-32    | ppyoloe_sail.soc  |      ppyoloe_fp16_1b.bmodel       |      2.71       |      3.33       |      7.22       |      8.11       |
|   SE9-16    | ppyoloe_opencv.py |      ppyoloe_fp32_1b.bmodel       |      23.48      |      55.35      |     139.76      |      17.91      |
|   SE9-16    | ppyoloe_opencv.py |      ppyoloe_fp16_1b.bmodel       |      19.24      |      55.54      |      51.71      |      17.77      |
|   SE9-16    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      4.59       |      5.44       |     124.17      |      17.67      |
|   SE9-16    |  ppyoloe_bmcv.py  |      ppyoloe_fp16_1b.bmodel       |      4.61       |      5.41       |      36.18      |      17.95      |
|   SE9-16    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      5.75       |      2.24       |     119.31      |      12.15      |
|   SE9-16    | ppyoloe_bmcv.soc  |      ppyoloe_fp16_1b.bmodel       |      5.80       |      2.23       |      31.36      |      12.14      |
|   SE9-16    | ppyoloe_sail.soc  |      ppyoloe_fp32_1b.bmodel       |      3.80       |      6.12       |     120.29      |      11.33      |
|   SE9-16    | ppyoloe_sail.soc  |      ppyoloe_fp16_1b.bmodel       |      3.75       |      6.11       |      32.33      |      11.32      |
|   SE9-16    | ppyoloe_opencv.py |   ppyoloe_fp32_1b_2core.bmodel    |      19.29      |      55.28      |      96.96      |      17.91      |
|   SE9-16    | ppyoloe_opencv.py |   ppyoloe_fp16_1b_2core.bmodel    |      19.16      |      55.66      |      41.51      |      17.78      |
|   SE9-16    |  ppyoloe_bmcv.py  |   ppyoloe_fp32_1b_2core.bmodel    |      4.54       |      5.41       |      81.45      |      17.71      |
|   SE9-16    |  ppyoloe_bmcv.py  |   ppyoloe_fp16_1b_2core.bmodel    |      4.54       |      5.39       |      25.95      |      17.74      |
|   SE9-16    | ppyoloe_bmcv.soc  |   ppyoloe_fp32_1b_2core.bmodel    |      5.83       |      2.23       |      76.57      |      12.14      |
|   SE9-16    | ppyoloe_bmcv.soc  |   ppyoloe_fp16_1b_2core.bmodel    |      5.80       |      2.23       |      21.16      |      12.15      |
|   SE9-16    | ppyoloe_sail.soc  |   ppyoloe_fp32_1b_2core.bmodel    |      3.81       |      6.11       |      77.55      |      11.32      |
|   SE9-16    | ppyoloe_sail.soc  |   ppyoloe_fp16_1b_2core.bmodel    |      3.77       |      6.10       |      22.11      |      11.33      |
|    SE9-8    | ppyoloe_opencv.py |      ppyoloe_fp32_1b.bmodel       |      24.17      |      55.57      |     142.53      |      17.66      |
|    SE9-8    | ppyoloe_opencv.py |      ppyoloe_fp16_1b.bmodel       |      20.70      |      55.99      |      55.20      |      17.54      |
|    SE9-8    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      7.66       |      5.63       |     127.22      |      17.71      |
|    SE9-8    |  ppyoloe_bmcv.py  |      ppyoloe_fp16_1b.bmodel       |      4.45       |      5.59       |      39.63      |      17.73      |
|    SE9-8    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      5.61       |      2.59       |     122.30      |      12.15      |
|    SE9-8    | ppyoloe_bmcv.soc  |      ppyoloe_fp16_1b.bmodel       |      5.64       |      2.59       |      34.85      |      12.14      |
|    SE9-8    | ppyoloe_sail.soc  |      ppyoloe_fp32_1b.bmodel       |      3.67       |      6.34       |     123.30      |      11.33      |
|    SE9-8    | ppyoloe_sail.soc  |      ppyoloe_fp16_1b.bmodel       |      3.64       |      6.32       |      35.81      |      11.33      |
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
    parser.add_argument('--bmodel', type=str, default='yolov5s_v6.1_3output_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov5_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov5s_v6.1_3output_fp32_1b.bmodel_python_test.log')
    parser.add_argument('--use_cpu_opt', action="store_true", default=False, help='accelerate cpu postprocess')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    args = argsparser()
    if args.use_cpu_opt:
        benchmark_path = current_dir + "/benchmark_cpu_opt.txt"
        baseline = baseline_cpu_opt
    else:
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
        if abs(statis - extracted_data[key]) / statis > 0.2:
            print("{:} time, diff ratio > 0.2".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^35}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
