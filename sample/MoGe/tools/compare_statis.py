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

# 基线性能表（decode/preprocess/inference/postprocess 平均耗时，单位 ms）。
# 数值为 SE9-16 / SE9-8 / SE7 等平台首次实测后回填。
baseline = """
|    测试平台  |      测试程序       |                测试模型                 |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ------------------ | -------------------------------------- | -------- | ---------     | ---------     | --------- |
|   SE9-16    |  moge_bmcv.py      |  moge-2-vitl-normal_fp16_1b.bmodel     |  15.31  |  15.23  |  1975.00  |  124.09  |
|   SE9-16    |  moge_bmcv.soc     |  moge-2-vitl-normal_fp16_1b.bmodel     |  14.85  |  14.34  |  1964.94  |   54.00  |
|   SE9-8     |  moge_bmcv.py      |  moge-2-vitl-normal_fp16_1b.bmodel     |  10.40  |  14.76  |  1947.56  |  109.70  |
|   SE9-8     |  moge_bmcv.soc     |  moge-2-vitl-normal_fp16_1b.bmodel     |  10.71  |  14.58  |  1938.64  |   53.88  |
|   SE7       |  moge_bmcv.py      |  moge-2-vitl-normal_fp32_1b.bmodel     |   7.24  |   8.43  |  3466.87  |   87.32  |
|   SE7       |  moge_bmcv.soc     |  moge-2-vitl-normal_fp32_1b.bmodel     |   7.36  |   9.58  |  3461.66  |   42.20  |
|   SE7       |  moge_bmcv.py      |  moge-2-vitl-normal_fp16_1b.bmodel     |   7.28  |   8.54  |   566.71  |   91.48  |
|   SE7       |  moge_bmcv.soc     |  moge-2-vitl-normal_fp16_1b.bmodel     |   7.47  |   9.94  |   557.20  |   42.27  |
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

if baseline.strip():
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

# cpp: TimeStamp::show_summary 打印 “[<tag>] loops: N avg: X ms”（20 列右对齐）
patterns_cpp = {
    'decode':     re.compile(r'\[\s*moge decode\s*\]\s*loops:\s*\d+\s*avg:\s*([\d.]+) ms'),
    'preprocess': re.compile(r'\[\s*moge preprocess\s*\]\s*loops:\s*\d+\s*avg:\s*([\d.]+) ms'),
    'inference':  re.compile(r'\[\s*moge inference\s*\]\s*loops:\s*\d+\s*avg:\s*([\d.]+) ms'),
    'postprocess': re.compile(r'\[\s*moge postprocess\s*\]\s*loops:\s*\d+\s*avg:\s*([\d.]+) ms'),
}

# python: moge_bmcv.py 打印的 Predict Time Info
patterns_python = {
    'decode':     re.compile(r'decode_time\(ms\): ([\d.]+)'),
    'preprocess': re.compile(r'preprocess_time\(ms\): ([\d.]+)'),
    'inference':  re.compile(r'inference_time\(ms\): ([\d.]+)'),
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
    parser.add_argument('--target', type=str, default='BM1688', help='target board type')
    parser.add_argument('--platform', type=str, default='soc', help='soc or pcie')
    parser.add_argument('--bmodel', type=str, default='moge-2-vitl-normal_fp16_1b.bmodel')
    parser.add_argument('--program', type=str, default='moge_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/moge_bmcv_fp16_1b_python_test.log')
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
            platform = "SE7"
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
            benchmark_str = "|{:^13}|{:^19}|{:^38}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
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
        sys.exit(1)

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

    benchmark_str = "|{:^13}|{:^19}|{:^38}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
        platform, args.program, args.bmodel, **extracted_data, width=min_width)

    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)

    if compare_pass == False:
        sys.exit(1)