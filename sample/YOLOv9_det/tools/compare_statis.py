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
|   SE5-16    | yolov9_opencv.py  | yolov9s_fp32_1b.bmodel  |      6.81       |      21.85      |      41.59      |      5.07       |
|   SE5-16    | yolov9_opencv.py  | yolov9s_int8_1b.bmodel  |      6.83       |      22.08      |      32.84      |      4.95       |
|   SE5-16    | yolov9_opencv.py  | yolov9s_int8_4b.bmodel  |      6.90       |      24.67      |      29.09      |      5.16       |
|   SE5-16    |  yolov9_bmcv.py   | yolov9s_fp32_1b.bmodel  |      3.59       |      2.75       |      38.77      |      4.97       |
|   SE5-16    |  yolov9_bmcv.py   | yolov9s_int8_1b.bmodel  |      3.74       |      2.89       |      28.09      |      4.93       |
|   SE5-16    |  yolov9_bmcv.py   | yolov9s_int8_4b.bmodel  |      3.45       |      2.56       |      20.58      |      4.35       |
|   SE5-16    |  yolov9_bmcv.soc  | yolov9s_fp32_1b.bmodel  |      4.87       |      1.55       |      36.39      |      8.53       |
|   SE5-16    |  yolov9_bmcv.soc  | yolov9s_int8_1b.bmodel  |      4.89       |      1.55       |      25.58      |      8.53       |
|   SE5-16    |  yolov9_bmcv.soc  | yolov9s_int8_4b.bmodel  |      4.76       |      1.49       |      18.62      |      8.49       |
|   SE7-32    | yolov9_opencv.py  | yolov9s_fp32_1b.bmodel  |      6.74       |      22.39      |      39.55      |      5.40       |
|   SE7-32    | yolov9_opencv.py  | yolov9s_fp16_1b.bmodel  |      6.80       |      22.85      |      13.27      |      5.38       |
|   SE7-32    | yolov9_opencv.py  | yolov9s_int8_1b.bmodel  |      6.80       |      22.79      |      10.79      |      5.33       |
|   SE7-32    | yolov9_opencv.py  | yolov9s_int8_4b.bmodel  |      6.83       |      24.81      |      10.18      |      5.41       |
|   SE7-32    |  yolov9_bmcv.py   | yolov9s_fp32_1b.bmodel  |      3.12       |      2.30       |      36.29      |      5.42       |
|   SE7-32    |  yolov9_bmcv.py   | yolov9s_fp16_1b.bmodel  |      3.10       |      2.30       |      10.03      |      5.46       |
|   SE7-32    |  yolov9_bmcv.py   | yolov9s_int8_1b.bmodel  |      3.10       |      2.28       |      7.58       |      5.42       |
|   SE7-32    |  yolov9_bmcv.py   | yolov9s_int8_4b.bmodel  |      2.94       |      2.11       |      6.76       |      4.88       |
|   SE7-32    |  yolov9_bmcv.soc  | yolov9s_fp32_1b.bmodel  |      4.31       |      0.74       |      33.67      |      8.64       |
|   SE7-32    |  yolov9_bmcv.soc  | yolov9s_fp16_1b.bmodel  |      4.34       |      0.74       |      7.40       |      8.61       |
|   SE7-32    |  yolov9_bmcv.soc  | yolov9s_int8_1b.bmodel  |      4.36       |      0.74       |      4.95       |      8.64       |
|   SE7-32    |  yolov9_bmcv.soc  | yolov9s_int8_4b.bmodel  |      4.19       |      0.71       |      4.57       |      8.59       |
|   SE9-16    | yolov9_opencv.py  | yolov9s_fp32_1b.bmodel  |      9.53       |      29.52      |     169.66      |      6.86       |
|   SE9-16    | yolov9_opencv.py  | yolov9s_fp16_1b.bmodel  |      9.36       |      29.43      |      48.36      |      6.93       |
|   SE9-16    | yolov9_opencv.py  | yolov9s_int8_1b.bmodel  |      9.38       |      29.60      |      25.45      |      6.87       |
|   SE9-16    | yolov9_opencv.py  | yolov9s_int8_4b.bmodel  |      9.44       |      32.74      |      24.92      |      7.43       |
|   SE9-16    |  yolov9_bmcv.py   | yolov9s_fp32_1b.bmodel  |      4.29       |      4.59       |     165.88      |      6.96       |
|   SE9-16    |  yolov9_bmcv.py   | yolov9s_fp16_1b.bmodel  |      4.28       |      4.56       |      44.27      |      6.95       |
|   SE9-16    |  yolov9_bmcv.py   | yolov9s_int8_1b.bmodel  |      4.28       |      4.58       |      21.50      |      6.90       |
|   SE9-16    |  yolov9_bmcv.py   | yolov9s_int8_4b.bmodel  |      4.17       |      4.27       |      20.38      |      6.11       |
|   SE9-16    |  yolov9_bmcv.soc  | yolov9s_fp32_1b.bmodel  |      5.88       |      1.73       |     162.30      |      12.06      |
|   SE9-16    |  yolov9_bmcv.soc  | yolov9s_fp16_1b.bmodel  |      5.86       |      1.73       |      40.89      |      12.05      |
|   SE9-16    |  yolov9_bmcv.soc  | yolov9s_int8_1b.bmodel  |      5.90       |      1.74       |      18.17      |      12.06      |
|   SE9-16    |  yolov9_bmcv.soc  | yolov9s_int8_4b.bmodel  |      5.74       |      1.66       |      17.70      |      12.01      |
|   SE9-16    | yolov9_opencv.py  |yolov9s_fp32_1b_2core.bmodel|      9.50       |      29.87      |      98.46      |      6.91       |
|   SE9-16    | yolov9_opencv.py  |yolov9s_fp16_1b_2core.bmodel|      9.40       |      29.44      |      32.21      |      6.91       |
|   SE9-16    | yolov9_opencv.py  |yolov9s_int8_1b_2core.bmodel|      9.37       |      29.89      |      19.96      |      6.85       |
|   SE9-16    | yolov9_opencv.py  |yolov9s_int8_4b_2core.bmodel|      9.47       |      32.67      |      17.02      |      6.90       |
|   SE9-16    |  yolov9_bmcv.py   |yolov9s_fp32_1b_2core.bmodel|      4.32       |      4.59       |      94.68      |      6.96       |
|   SE9-16    |  yolov9_bmcv.py   |yolov9s_fp16_1b_2core.bmodel|      4.28       |      4.58       |      28.31      |      6.93       |
|   SE9-16    |  yolov9_bmcv.py   |yolov9s_int8_1b_2core.bmodel|      4.28       |      4.57       |      15.99      |      6.90       |
|   SE9-16    |  yolov9_bmcv.py   |yolov9s_int8_4b_2core.bmodel|      4.16       |      4.26       |      12.93      |      6.12       |
|   SE9-16    |  yolov9_bmcv.soc  |yolov9s_fp32_1b_2core.bmodel|      5.87       |      1.73       |      91.16      |      12.06      |
|   SE9-16    |  yolov9_bmcv.soc  |yolov9s_fp16_1b_2core.bmodel|      5.93       |      1.74       |      24.88      |      12.05      |
|   SE9-16    |  yolov9_bmcv.soc  |yolov9s_int8_1b_2core.bmodel|      5.85       |      1.73       |      12.63      |      12.06      |
|   SE9-16    |  yolov9_bmcv.soc  |yolov9s_int8_4b_2core.bmodel|      5.72       |      1.66       |      10.11      |      12.01      |
|    SE9-8    | yolov9_opencv.py  | yolov9s_fp32_1b.bmodel  |      9.42       |      30.22      |     169.91      |      7.02       |
|    SE9-8    | yolov9_opencv.py  | yolov9s_fp16_1b.bmodel  |      9.45       |      30.16      |      48.42      |      6.97       |
|    SE9-8    | yolov9_opencv.py  | yolov9s_int8_1b.bmodel  |      9.38       |      29.75      |      25.53      |      6.94       |
|    SE9-8    | yolov9_opencv.py  | yolov9s_int8_4b.bmodel  |      9.48       |      32.99      |      24.74      |      7.06       |
|    SE9-8    |  yolov9_bmcv.py   | yolov9s_fp32_1b.bmodel  |      4.19       |      4.48       |     165.90      |      7.04       |
|    SE9-8    |  yolov9_bmcv.py   | yolov9s_fp16_1b.bmodel  |      4.13       |      4.48       |      44.29      |      7.03       |
|    SE9-8    |  yolov9_bmcv.py   | yolov9s_int8_1b.bmodel  |      4.13       |      4.47       |      21.54      |      6.96       |
|    SE9-8    |  yolov9_bmcv.py   | yolov9s_int8_4b.bmodel  |      3.96       |      4.16       |      20.54      |      6.40       |
|    SE9-8    |  yolov9_bmcv.soc  | yolov9s_fp32_1b.bmodel  |      5.74       |      1.72       |     162.34      |      12.15      |
|    SE9-8    |  yolov9_bmcv.soc  | yolov9s_fp16_1b.bmodel  |      5.79       |      1.72       |      40.94      |      12.14      |
|    SE9-8    |  yolov9_bmcv.soc  | yolov9s_int8_1b.bmodel  |      5.75       |      1.72       |      18.15      |      12.15      |
|    SE9-8    |  yolov9_bmcv.soc  | yolov9s_int8_4b.bmodel  |      5.62       |      1.64       |      17.66      |      12.10      |
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
    parser.add_argument('--bmodel', type=str, default='yolov9s_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov9_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov9s_fp32_1b.bmodel_python_test.log')
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
