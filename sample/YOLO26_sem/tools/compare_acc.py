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
| 测试平台 | 测试程序              | 测试模型                 | mIoU  | Pixel Acc |
| -------- | --------------------- | ------------------------ | ----- | --------- |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |  80.81 |    96.33 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |  80.81 |    96.33 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |  80.81 |    96.33 |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |  80.56 |    96.29 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |  80.56 |    96.29 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |  80.56 |    96.29 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |  80.81 |    96.33 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |  80.80 |    96.33 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |  80.80 |    96.33 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |  80.56 |    96.29 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |  80.56 |    96.30 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |  80.56 |    96.30 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_int8_1b_2core.bmodel |  80.56 |    96.29 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_int8_1b_2core.bmodel |  80.56 |    96.30 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b_2core.bmodel |  80.56 |    96.30 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |  80.81 |    96.33 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |  80.80 |    96.33 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |  80.80 |    96.33 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |  80.56 |    96.29 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |  80.56 |    96.30 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |  80.56 |    96.30 |
"""

table_data = {
    "platform": [],
    "program": [],
    "bmodel": [],
    "miou": [],
    "pacc": [],
}

for line in baseline.strip().split("\n")[2:]:
    match = re.search(r'\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', line)
    if match:
        table_data["platform"].append(match.group(1))
        table_data["program"].append(match.group(2))
        table_data["bmodel"].append(match.group(3))
        table_data["miou"].append(float(match.group(4)))
        table_data["pacc"].append(float(match.group(5)))

patterns_eval = {
    'miou': re.compile(r'mIoU\s*:\s*([0-9.]+)'),
    'pacc': re.compile(r'Pixel Acc\s*:\s*([0-9.]+)'),
}


def extract(text, patterns):
    results = {}
    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            results[key] = round(float(match.group(1)), 3)
    return results


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--target', type=str, default='BM1684X', help='target chip')
    parser.add_argument('--platform', type=str, default='soc', help='pcie or soc')
    parser.add_argument('--bmodel', type=str, default='yolo26s_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolo26_sem_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolo26s_fp32_1b.bmodel_python_eval.log')
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
            benchmark_str = "|{:^13}|{:^25}|{:^40}|{:^{width}}|{:^{width}}|\n".format(
                "platform", "program", "bmodel", "miou", "pacc", width=min_width)
            f.write(benchmark_str)

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
        baseline_data["miou"] = table_data["miou"][match_index]
        baseline_data["pacc"] = table_data["pacc"][match_index]
    for key, statis in baseline_data.items():
        if statis != 0 and abs(statis - extracted_data[key]) / statis > 0.01:
            print("{:}, diff ratio > 0.01".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False

    benchmark_str = "|{:^13}|{:^25}|{:^40}|{miou:^{width}.3f}|{pacc:^{width}.3f}|\n".format(
        platform, args.program, args.bmodel, **extracted_data, width=min_width)

    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)

    if compare_pass == False:
        sys.exit(1)