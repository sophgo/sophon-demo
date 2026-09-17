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
| 测试平台 | 测试程序              | 测试模型                 | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | --------------------- | ------------------------ | ----------- | --------------- | -------------- | ---------------- |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |    95.00 |   126.51 |   156.34 |     5.70 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |   109.06 |    10.18 |   141.99 |     5.71 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |   107.77 |     7.00 |   137.37 |     5.54 |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |    92.00 |   122.44 |    67.32 |     5.67 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |   106.37 |    10.85 |    52.98 |     5.72 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |   109.93 |     6.92 |    48.38 |     5.55 |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |    91.52 |   122.39 |    53.86 |     5.67 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |   106.53 |    10.18 |    39.66 |     5.71 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |   105.04 |     6.99 |    35.11 |     5.53 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |   118.43 |   157.77 |   615.58 |     7.10 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |   121.51 |    19.82 |   597.48 |     7.35 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |   118.61 |    11.17 |   592.14 |     6.89 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |   118.28 |   162.63 |   194.83 |     7.02 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |   121.00 |    19.82 |   176.69 |     7.32 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |   118.50 |    11.16 |   171.39 |     6.86 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |   118.23 |   162.36 |    89.80 |     7.02 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |   120.91 |    19.82 |    72.35 |     7.03 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |   118.53 |    11.16 |    66.35 |     6.85 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_int8_1b_2core.bmodel |   118.41 |   157.82 |    78.84 |     7.04 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_int8_1b_2core.bmodel |   120.95 |    19.85 |    61.03 |     7.03 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b_2core.bmodel |   118.56 |    11.15 |    55.10 |     6.88 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |   144.21 |   203.58 |   612.18 |     7.04 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |   148.90 |    19.87 |   594.51 |     7.08 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |   126.09 |    11.17 |   588.50 |     6.91 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |   145.38 |   187.29 |   194.86 |     7.04 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |   150.42 |    19.87 |   177.22 |     7.05 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |   129.40 |    11.16 |   171.29 |     6.91 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |   145.90 |   196.64 |    89.70 |     7.03 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |   150.42 |    19.88 |    72.10 |     7.08 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |   133.19 |    11.17 |    66.10 |     6.90 |
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
    parser.add_argument('--bmodel', type=str, default='yolo26s_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolo26_sem_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolo26s_fp32_1b.bmodel_python_test.log')
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