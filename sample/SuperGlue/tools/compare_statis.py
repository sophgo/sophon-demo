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
|    测试平台  |     测试程序     | superpoint模型            |   superglue模型                    | decode_time    |superpoint_time  |superglue_time   | 
| ----------- | ---------------- | ---------------          | ----------------                   | --------       | ---------       | ---------     |
|   SE7-32    |superglue_bmcv.soc |superpoint_fp32_1b.bmodel|superglue_fp32_1b_iter20_1024.bmodel|      12.38      |      97.33      |     301.75      |
|   SE9-16    |superglue_bmcv.soc |superpoint_fp32_1b.bmodel|superglue_fp32_1b_iter20_1024.bmodel|      5.70       |     263.51      |     686.38      |
|   SE9-16    |superglue_bmcv.soc |superpoint_fp16_1b.bmodel|superglue_fp16_1b_iter20_1024.bmodel|      5.62       |      89.39      |     198.20      |
|    SE9-8    |superglue_bmcv.soc |superpoint_fp32_1b.bmodel|superglue_fp32_1b_iter20_1024.bmodel|      5.45       |     269.53      |     686.53      |
|    SE9-8    |superglue_bmcv.soc |superpoint_fp16_1b.bmodel|superglue_fp16_1b_iter20_1024.bmodel|      5.78       |      84.62      |     200.16      |
"""

table_data = {
    "platform": [],
    "program": [],
    "bmodel_superpoint": [],
    "bmodel_superglue": [],
    "decode": [],
    "superpoint": [],
    "superglue": []
}

patterns_cpp = {
    'decode': re.compile(r'\[.*decode time.*\]  loops:.*avg: ([\d.]+) ms'),
    'superpoint': re.compile(r'\[.*superpoint time.*\]  loops:.*avg: ([\d.]+) ms'),
    'superglue': re.compile(r'\[.*superglue time.*\]  loops:.*avg: ([\d.]+) ms'),
}

patterns_python = {
    'decode': re.compile(r'decode_time\(ms\): ([\d.]+)'),
    'superpoint': re.compile(r'superpoint_time\(ms\): ([\d.]+)'),
    'superglue': re.compile(r'superglue_time\(ms\): ([\d.]+)'),
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
    parser.add_argument('--bmodel_superpoint', type=str, default='superpoint_fp32_1b.bmodel')
    parser.add_argument('--bmodel_superglue', type=str, default='superglue_fp32_1b_iter20_1024.bmodel')
    parser.add_argument('--program', type=str, default='superglue_bmcv.soc')
    parser.add_argument('--language', type=str, default='cpp')
    parser.add_argument('--input', type=str, default='')
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
            table_data["bmodel_superpoint"].append(match.group(3))
            table_data["bmodel_superglue"].append(match.group(4))
            table_data["decode"].append(float(match.group(5)))
            table_data["superpoint"].append(float(match.group(6)))
            table_data["superglue"].append(float(match.group(7)))

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
            benchmark_str = "|{:^13}|{:^19}|{:^25}|{:^35}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
           "platform", "program", "bmodel_superpoint", "bmodel_superglue", "decode_time", "superpoint_time", "superglue_time", width=min_width)
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
        if platform == table_data["platform"][i] and args.program == table_data["program"][i] and args.bmodel_superpoint == table_data["bmodel_superpoint"][i] \
            and args.bmodel_superglue == table_data["bmodel_superglue"][i]:
            match_index = i
            break
    baseline_data = {}
    if match_index == -1:
        print("Unmatched case.")
    else:
        baseline_data["decode"] = table_data["decode"][match_index]
        baseline_data["superpoint"] = table_data["superpoint"][match_index]
        baseline_data["superglue"] = table_data["superglue"][match_index]
    for key, statis in baseline_data.items():
        if abs(statis - extracted_data[key]) / statis > 0.4:
            print("{:} time, diff ratio > 0.4".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^25}|{:^35}|{decode:^{width}.2f}|{superpoint:^{width}.2f}|{superglue:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel_superpoint, args.bmodel_superglue, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
