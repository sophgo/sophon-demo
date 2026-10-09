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
| 测试平台 | 测试程序              | 测试模型                      | delta1 | delta2 | delta3 | abs_rel | rmse  | silog  |
| -------- | --------------------- | ----------------------------- | ------ | ------ | ------ | ------- | ----- | ------ |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   |  0.7874 |  0.9365 |  0.9785 |  0.1528  |  0.6601 | 22.7200 |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   |  0.7874 |  0.9351 |  0.9773 |  0.1533  |  0.6638 | 22.9538 |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   |  0.7877 |  0.9365 |  0.9784 |  0.1527  |  0.6600 | 22.7261 |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   |  0.7874 |  0.9365 |  0.9785 |  0.1528  |  0.6601 | 22.7203 |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   |  0.7874 |  0.9351 |  0.9773 |  0.1533  |  0.6639 | 22.9542 |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   |  0.7877 |  0.9365 |  0.9784 |  0.1527  |  0.6601 | 22.7270 |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   |  0.7893 |  0.9362 |  0.9764 |  0.1528  |  0.6600 | 22.9592 |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   |  0.7898 |  0.9358 |  0.9762 |  0.1529  |  0.6603 | 23.0062 |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   |  0.7893 |  0.9360 |  0.9764 |  0.1529  |  0.6603 | 22.9691 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   |  0.7874 |  0.9365 |  0.9785 |  0.1528  |  0.6601 | 22.7200 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   |  0.7875 |  0.9351 |  0.9772 |  0.1533  |  0.6638 | 22.9547 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   |  0.7878 |  0.9365 |  0.9783 |  0.1527  |  0.6601 | 22.7295 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   |  0.7874 |  0.9365 |  0.9785 |  0.1528  |  0.6601 | 22.7202 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   |  0.7875 |  0.9351 |  0.9772 |  0.1533  |  0.6638 | 22.9544 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   |  0.7878 |  0.9365 |  0.9783 |  0.1527  |  0.6601 | 22.7302 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   |  0.7893 |  0.9362 |  0.9764 |  0.1528  |  0.6600 | 22.9591 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   |  0.7892 |  0.9359 |  0.9761 |  0.1532  |  0.6609 | 23.0373 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   |  0.7896 |  0.9360 |  0.9763 |  0.1527  |  0.6603 | 22.9676 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b_2core.bmodel |  0.7893 |  0.9362 |  0.9764 |  0.1528  |  0.6600 | 22.9591 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b_2core.bmodel |  0.7892 |  0.9359 |  0.9761 |  0.1532  |  0.6609 | 23.0373 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b_2core.bmodel |  0.7896 |  0.9360 |  0.9763 |  0.1527  |  0.6603 | 22.9676 |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   |  0.7874 |  0.9365 |  0.9785 |  0.1528  |  0.6601 | 22.7200 |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   |  0.7875 |  0.9351 |  0.9772 |  0.1533  |  0.6638 | 22.9547 |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   |  0.7878 |  0.9365 |  0.9783 |  0.1527  |  0.6601 | 22.7295 |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   |  0.7874 |  0.9365 |  0.9785 |  0.1528  |  0.6601 | 22.7202 |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   |  0.7875 |  0.9351 |  0.9772 |  0.1533  |  0.6638 | 22.9544 |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   |  0.7878 |  0.9365 |  0.9783 |  0.1527  |  0.6601 | 22.7302 |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   |  0.7893 |  0.9362 |  0.9764 |  0.1528  |  0.6600 | 22.9591 |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   |  0.7892 |  0.9359 |  0.9761 |  0.1532  |  0.6609 | 23.0373 |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   |  0.7896 |  0.9360 |  0.9763 |  0.1527  |  0.6603 | 22.9676 |
"""

metric_names = ["delta1", "delta2", "delta3", "abs_rel", "rmse", "silog"]
table_data = {"platform": [], "program": [], "bmodel": []}
for name in metric_names:
    table_data[name] = []

for line in baseline.strip().split("\n")[2:]:
    match = re.search(r'\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', line)
    if match:
        table_data["platform"].append(match.group(1))
        table_data["program"].append(match.group(2))
        table_data["bmodel"].append(match.group(3))
        for i, name in enumerate(metric_names):
            table_data[name].append(float(match.group(4 + i)))

patterns_eval = {
    'delta1': re.compile(r'delta1\s*:\s*([0-9.]+)'),
    'delta2': re.compile(r'delta2\s*:\s*([0-9.]+)'),
    'delta3': re.compile(r'delta3\s*:\s*([0-9.]+)'),
    'abs_rel': re.compile(r'abs_rel:\s*([0-9.]+)'),
    'rmse': re.compile(r'rmse\s*:\s*([0-9.]+)'),
    'silog': re.compile(r'silog\s*:\s*([0-9.]+)'),
}


def extract(text, patterns):
    results = {}
    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            results[key] = round(float(match.group(1)), 4)
    return results


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--target', type=str, default='BM1684X', help='target chip')
    parser.add_argument('--platform', type=str, default='soc', help='pcie or soc')
    parser.add_argument('--bmodel', type=str, default='yolo26s_depth_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolo26_depth_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolo26s_depth_fp32_1b.bmodel_python_eval_res.log')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    benchmark_path = current_dir + "/acc.txt"
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
    min_width = 10

    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            header = "|{:^13}|{:^25}|{:^40}|".format("platform", "program", "bmodel")
            header += "".join("|{name:^{width}}".format(name=name, width=min_width) for name in metric_names)
            f.write(header + "|\n")

    with open(args.input, "r") as f:
        data = f.read()
    extracted_data = extract(data, patterns_eval)
    match_index = -1
    for i in range(0, len(table_data["platform"])):
        if platform == table_data["platform"][i] and args.program == table_data["program"][i] and args.bmodel == table_data["bmodel"][i]:
            match_index = i
            break
    baseline_data = {}
    if match_index == -1:
        print("Unmatched case.")
    else:
        for name in metric_names:
            baseline_data[name] = table_data[name][match_index]
    for key, statis in baseline_data.items():
        if statis != 0 and abs(statis - extracted_data[key]) / abs(statis) > 0.01:
            print("{:}, diff ratio > 0.01".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False

    row = "|{:^13}|{:^25}|{:^40}|".format(platform, args.program, args.bmodel)
    row += "".join("|{v:^{width}.4f}".format(v=extracted_data[name], width=min_width) for name in metric_names)
    row += "|\n"

    with open(benchmark_path, "a") as f:
        f.write(row)

    if compare_pass == False:
        sys.exit(1)