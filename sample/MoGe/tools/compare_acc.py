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

# 基线精度表：MoGe 官方 depth_metric 组的两个指标（见 tools/eval_moge.py 头部注释）。
#   rel    = AbsRel，越低越好
#   delta1 = delta 1.25，越高越好
# 数值为各平台首次实测后回填。首行以 fp32 bmodel（与 fp32 onnx/torch 参考一致）作为基准。
baseline = """
|   测试平台    |      测试程序      |               测试模型                |    rel   |  delta1 |
| ------------ | ------------------ | ------------------------------------- | -------- | ------- |
|   SE9-16     |  moge_bmcv.py      |  moge-2-vitl-normal_fp32_1b.bmodel    |  0.1034  |  0.9166 |
|   SE9-16     |  moge_bmcv.py      |  moge-2-vitl-normal_fp16_1b.bmodel    |  0.1041  |  0.9146 |
|   SE7        |  moge_bmcv.py      |  moge-2-vitl-normal_fp32_1b.bmodel    |  0.1048  |  0.9110 |
|   SE7        |  moge_bmcv.py      |  moge-2-vitl-normal_fp16_1b.bmodel    |  0.1056  |  0.9089 |
|   SE9-8      |  moge_bmcv.py      |  moge-2-vitl-normal_fp16_1b.bmodel    |  0.1041  |  0.9146 |
"""

table_data = {
    "platform": [],
    "program": [],
    "bmodel": [],
    "rel": [],
    "delta1": [],
}

if baseline.strip():
    for line in baseline.strip().split("\n")[2:]:
        match = re.search(r'\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', line)
        if match:
            table_data["platform"].append(match.group(1))
            table_data["program"].append(match.group(2))
            table_data["bmodel"].append(match.group(3))
            table_data["rel"].append(float(match.group(4)))
            table_data["delta1"].append(float(match.group(5)))

# tools/eval_moge.py 的汇总行: "[NYUv2] depth_metric rel=0.0743 delta1=0.9552"
patterns_eval = {
    'rel':    re.compile(r'depth_metric\s+rel=([\d.]+)'),
    'delta1': re.compile(r'depth_metric\s+rel=[\d.]+\s+delta1=([\d.]+)'),
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
    parser.add_argument('--target', type=str, default='BM1688', help='target board type')
    parser.add_argument('--platform', type=str, default='soc', help='soc or pcie')
    parser.add_argument('--bmodel', type=str, default='moge-2-vitl-normal_fp16_1b.bmodel')
    parser.add_argument('--program', type=str, default='moge_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/moge_bmcv_fp16_1b_eval.log')
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
    min_width = 9

    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^13}|{:^19}|{:^38}|{:^{width}}|{:^{width}}|\n".format(
                "platform", "program", "bmodel", "rel", "delta1", width=min_width)
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
        baseline_data["rel"] = table_data["rel"][match_index]
        baseline_data["delta1"] = table_data["delta1"][match_index]

    # rel 越低越好（相对退化 > 5% 判失败），delta1 越高越好（相对退化 > 2% 判失败）。
    for key, statis in baseline_data.items():
        now = extracted_data[key]
        if key == "rel":
            if now > statis * 1.05:
                print("rel, relative regression > 5%")
                compare_pass = False
        else:
            if now < statis * 0.98:
                print("delta1, relative regression > 2%")
                compare_pass = False
        if compare_pass is False:
            print("Baseline is:", statis)
            print("Now is: ", now)

    benchmark_str = "|{:^13}|{:^19}|{:^38}|{rel:^{width}.4f}|{delta1:^{width}.4f}|\n".format(
        platform, args.program, args.bmodel, **extracted_data, width=min_width)

    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)

    if compare_pass == False:
        sys.exit(1)