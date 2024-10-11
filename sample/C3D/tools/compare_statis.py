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
|    测试平台  |     测试程序  |      测试模型     |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------- | ---------------- | -------- | ---------   | ---------------| --------- |
|   SE5-16    |   c3d_opencv.py   |   c3d_fp32_1b.bmodel    |      66.43      |      30.22      |      68.69      |      0.09       |
|   SE5-16    |   c3d_opencv.py   |   c3d_fp32_4b.bmodel    |      67.00      |      37.55      |      56.48      |      0.03       |
|   SE5-16    |   c3d_opencv.py   |   c3d_int8_1b.bmodel    |      66.39      |      30.39      |      34.65      |      0.09       |
|   SE5-16    |   c3d_opencv.py   |   c3d_int8_4b.bmodel    |      67.18      |      37.69      |      13.71      |      0.03       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_fp32_1b.bmodel    |      71.78      |      26.17      |      62.32      |      0.01       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_fp32_4b.bmodel    |      71.94      |      25.91      |      50.09      |      0.00       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_int8_1b.bmodel    |      71.73      |      26.07      |      28.22      |      0.01       |
|   SE5-16    |  c3d_opencv.soc   |   c3d_int8_4b.bmodel    |      71.67      |      25.80      |      7.39       |      0.00       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_fp32_1b.bmodel    |      75.55      |      6.74       |      62.29      |      0.01       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_fp32_4b.bmodel    |      74.63      |      6.62       |      50.08      |      0.00       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_int8_1b.bmodel    |      74.72      |      6.72       |      28.21      |      0.01       |
|   SE5-16    |   c3d_bmcv.soc    |   c3d_int8_4b.bmodel    |      74.97      |      6.57       |      7.38       |      0.00       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp32_1b.bmodel    |      65.80      |      30.95      |      86.39      |      0.09       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp32_4b.bmodel    |      67.21      |      38.68      |      80.74      |      0.03       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp16_1b.bmodel    |      66.06      |      31.05      |      16.78      |      0.09       |
|   SE7-32    |   c3d_opencv.py   |   c3d_fp16_4b.bmodel    |      67.21      |      38.67      |      14.18      |      0.03       |
|   SE7-32    |   c3d_opencv.py   |   c3d_int8_1b.bmodel    |      65.83      |      30.88      |      12.88      |      0.09       |
|   SE7-32    |   c3d_opencv.py   |   c3d_int8_4b.bmodel    |      67.07      |      38.60      |      11.53      |      0.03       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp32_1b.bmodel    |      71.89      |      26.43      |      79.06      |      0.01       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp32_4b.bmodel    |      72.32      |      26.08      |      73.65      |      0.00       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp16_1b.bmodel    |      71.86      |      26.39      |      9.48       |      0.01       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_fp16_4b.bmodel    |      72.34      |      26.15      |      7.11       |      0.00       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_int8_1b.bmodel    |      72.16      |      26.40      |      5.57       |      0.01       |
|   SE7-32    |  c3d_opencv.soc   |   c3d_int8_4b.bmodel    |      72.36      |      26.14      |      4.40       |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp32_1b.bmodel    |      74.45      |      3.64       |      79.03      |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp32_4b.bmodel    |      75.02      |      3.48       |      73.63      |      0.00       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp16_1b.bmodel    |      74.71      |      3.60       |      9.46       |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_fp16_4b.bmodel    |      74.66      |      3.49       |      7.10       |      0.00       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_int8_1b.bmodel    |      74.84      |      3.62       |      5.52       |      0.01       |
|   SE7-32    |   c3d_bmcv.soc    |   c3d_int8_4b.bmodel    |      75.16      |      3.49       |      4.41       |      0.00       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp32_1b.bmodel    |      96.75      |      42.28      |     414.48      |      0.15       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp32_4b.bmodel    |      98.52      |      52.92      |     397.24      |      0.05       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp16_1b.bmodel    |      92.39      |      42.43      |      78.57      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |   c3d_fp16_4b.bmodel    |      95.02      |      52.87      |      72.27      |      0.05       |
|   SE9-16    |   c3d_opencv.py   |   c3d_int8_1b.bmodel    |      93.03      |      42.33      |      34.95      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |   c3d_int8_4b.bmodel    |      94.30      |      53.22      |      32.14      |      0.05       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp32_1b.bmodel    |     122.06      |      30.70      |     405.25      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp32_4b.bmodel    |     122.44      |      30.67      |     388.04      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp16_1b.bmodel    |     122.15      |      30.77      |      69.35      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_fp16_4b.bmodel    |     121.63      |      30.65      |      62.87      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_int8_1b.bmodel    |     122.96      |      30.66      |      25.71      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |   c3d_int8_4b.bmodel    |     122.21      |      30.44      |      22.66      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp32_1b.bmodel    |     128.66      |      7.74       |     405.21      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp32_4b.bmodel    |     127.97      |      7.67       |     388.03      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp16_1b.bmodel    |     128.49      |      7.75       |      69.32      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_fp16_4b.bmodel    |     128.36      |      7.58       |      62.86      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_int8_1b.bmodel    |     128.36      |      7.73       |      25.67      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |   c3d_int8_4b.bmodel    |     129.14      |      7.62       |      22.64      |      0.01       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp32_1b_2core.bmodel |      93.07      |      42.35      |     413.64      |      0.14       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp32_4b_2core.bmodel |      94.06      |      52.93      |     397.59      |      0.05       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp16_1b_2core.bmodel |      92.29      |      42.29      |      64.51      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |c3d_fp16_4b_2core.bmodel |      93.87      |      53.11      |      58.81      |      0.05       |
|   SE9-16    |   c3d_opencv.py   |c3d_int8_1b_2core.bmodel |      92.49      |      42.50      |      31.77      |      0.13       |
|   SE9-16    |   c3d_opencv.py   |c3d_int8_4b_2core.bmodel |      93.98      |      52.95      |      28.65      |      0.05       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp32_1b_2core.bmodel |     122.02      |      30.92      |     404.39      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp32_4b_2core.bmodel |     122.63      |      30.58      |     388.57      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp16_1b_2core.bmodel |     122.53      |      30.66      |      55.30      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |c3d_fp16_4b_2core.bmodel |     122.32      |      30.46      |      48.80      |      0.01       |
|   SE9-16    |  c3d_opencv.soc   |c3d_int8_1b_2core.bmodel |     122.52      |      30.69      |      22.54      |      0.02       |
|   SE9-16    |  c3d_opencv.soc   |c3d_int8_4b_2core.bmodel |     122.31      |      30.63      |      19.48      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp32_1b_2core.bmodel |     127.54      |      7.87       |     404.36      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp32_4b_2core.bmodel |     128.10      |      7.66       |     388.56      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp16_1b_2core.bmodel |     131.29      |      7.90       |      55.29      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_fp16_4b_2core.bmodel |     130.37      |      7.57       |      48.79      |      0.01       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_int8_1b_2core.bmodel |     128.57      |      7.83       |      22.49      |      0.02       |
|   SE9-16    |   c3d_bmcv.soc    |c3d_int8_4b_2core.bmodel |     127.93      |      7.50       |      19.46      |      0.01       |
|    SE9-8    |   c3d_opencv.py   |   c3d_fp32_1b.bmodel    |      91.67      |      41.95      |     427.43      |      0.13       |
|    SE9-8    |   c3d_opencv.py   |   c3d_fp32_4b.bmodel    |      93.11      |      50.04      |     403.67      |      0.05       |
|    SE9-8    |   c3d_opencv.py   |   c3d_fp16_1b.bmodel    |      91.76      |      42.03      |      85.14      |      0.13       |
|    SE9-8    |   c3d_opencv.py   |   c3d_fp16_4b.bmodel    |      88.07      |      49.75      |      75.09      |      0.05       |
|    SE9-8    |   c3d_opencv.py   |   c3d_int8_1b.bmodel    |      86.82      |      42.07      |      41.89      |      0.13       |
|    SE9-8    |   c3d_opencv.py   |   c3d_int8_4b.bmodel    |      87.39      |      49.91      |      37.04      |      0.04       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_fp32_1b.bmodel    |     120.19      |      33.83      |     418.01      |      0.02       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_fp32_4b.bmodel    |     119.16      |      33.51      |     394.04      |      0.01       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_fp16_1b.bmodel    |     119.90      |      33.60      |      75.82      |      0.02       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_fp16_4b.bmodel    |     118.36      |      33.43      |      66.01      |      0.01       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_int8_1b.bmodel    |     119.01      |      33.79      |      32.55      |      0.02       |
|    SE9-8    |  c3d_opencv.soc   |   c3d_int8_4b.bmodel    |     118.10      |      33.40      |      27.83      |      0.01       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_fp32_1b.bmodel    |     130.30      |      9.05       |     418.00      |      0.02       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_fp32_4b.bmodel    |     130.70      |      8.84       |     394.03      |      0.01       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_fp16_1b.bmodel    |     131.08      |      9.00       |      75.79      |      0.02       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_fp16_4b.bmodel    |     128.82      |      8.75       |      66.03      |      0.01       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_int8_1b.bmodel    |     130.24      |      8.95       |      32.53      |      0.02       |
|    SE9-8    |   c3d_bmcv.soc    |   c3d_int8_4b.bmodel    |     128.00      |      8.81       |      27.79      |      0.01       |
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
    'decode': re.compile(r'\[.*decode_time.*\]  loops:.*avg: ([\d.]+) ms'),
    'preprocess': re.compile(r'\[.*preprocess_time.*\]  loops:.*avg: ([\d.]+) ms'),
    'inference': re.compile(r'\[.*inference.*\]  loops:.*avg: ([\d.]+) ms'),
    'postprocess': re.compile(r'\[.*postprocess_time.*\]  loops:.*avg: ([\d.]+) ms'),
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
    parser.add_argument('--bmodel', type=str, default='c3d_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='c3d_opencv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/opencv_c3d_fp32_1b.bmodel_python_debug.log')
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
        
