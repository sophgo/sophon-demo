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
import os
import sys
import multiprocessing

baseline = """
| 测试平台 | 测试程序              | 测试模型                      | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | --------------------- | ----------------------------- | ----------- | --------------- | -------------- | ---------------- |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   |     33.25 |     11.11 |      74.17 |       1.35 |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   |      7.81 |      2.43 |      69.73 |       1.35 |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   |      1.50 |      2.00 |      68.36 |       1.88 |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   |      5.06 |     10.59 |      19.08 |       1.36 |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   |      1.70 |      2.43 |      14.65 |       1.36 |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   |      1.49 |      1.99 |      13.27 |       1.84 |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   |      5.06 |     11.48 |      13.21 |       1.35 |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   |      1.71 |      2.43 |       8.83 |       1.35 |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   |      1.49 |      1.99 |       7.43 |       1.83 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   |     14.50 |     13.05 |     403.74 |       1.77 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   |      9.61 |      4.39 |     398.34 |       1.69 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   |      9.76 |      3.33 |     396.60 |       2.76 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   |      6.35 |     14.31 |      82.65 |       2.61 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   |      2.94 |      4.37 |      77.22 |       1.68 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   |      2.62 |      3.33 |      75.48 |       2.73 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   |      6.34 |     13.03 |      26.34 |       1.70 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   |      2.95 |      4.38 |      20.94 |       1.68 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   |      2.62 |      3.33 |      19.21 |       2.75 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b_2core.bmodel |      6.34 |     13.89 |      21.95 |       1.75 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b_2core.bmodel |      2.95 |      4.38 |      16.44 |       1.67 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b_2core.bmodel |      2.61 |      3.33 |      14.74 |       2.55 |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   |      8.27 |     14.21 |     403.79 |       2.07 |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   |     37.66 |      4.38 |     398.30 |       1.65 |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   |     27.65 |      3.34 |     396.52 |       2.60 |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   |     50.30 |     14.45 |      82.83 |       2.19 |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   |     49.41 |      4.40 |      77.21 |       1.67 |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   |     32.38 |      3.34 |      75.48 |       2.66 |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   |     34.08 |     14.26 |      26.75 |       2.09 |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   |     28.07 |      4.38 |      20.92 |       1.67 |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   |     27.26 |      3.33 |      19.20 |       2.66 |
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
            results[key] = round(float(match.group(1)), 2)
    return results


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--target', type=str, default='BM1684X', help='target chip')
    parser.add_argument('--platform', type=str, default='soc', help='pcie or soc')
    parser.add_argument('--bmodel', type=str, default='yolo26s_depth_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolo26_depth_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolo26s_depth_fp32_1b.bmodel_python_test.log')
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
        platform = args.target + " soC" if args.platform == "soc" else args.target + " PCIe"
    min_width = 17

    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^13}|{:^25}|{:^40}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
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

    benchmark_str = "|{:^13}|{:^25}|{:^40}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
        platform, args.program, args.bmodel, **extracted_data, width=min_width)

    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)

    if compare_pass == False:
        sys.exit(1)