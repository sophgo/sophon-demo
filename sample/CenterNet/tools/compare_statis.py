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

baseline = """
|    测试平台  |     测试程序           |             测试模型       |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ----------------      | ---------------------------| --------  | ---------     | ---------    | ---------    |
| BM1684 SoC  | centernet_opencv.py   | centernet_fp32_1b.bmodel   | 15.33     | 40.37         | 59.70        | 822.85       |
| BM1684 SoC  | centernet_opencv.py   | centernet_int8_1b.bmodel   | 15.46     | 40.83         | 44.46        | 812.17       |
| BM1684 SoC  | centernet_opencv.py   | centernet_int8_4b.bmodel   | 15.32     | 39.22         | 28.77        | 842.73       |
| BM1684 SoC  | centernet_bmcv.py     | centernet_fp32_1b.bmodel   | 3.03      | 2.61          | 50.53        | 820.98       |
| BM1684 SoC  | centernet_bmcv.py     | centernet_int8_1b.bmodel   | 3.04      | 2.25          | 26.18        | 827.04       |
| BM1684 SoC  | centernet_bmcv.py     | centernet_int8_4b.bmodel   | 2.89      | 2.11          | 11.79        | 851.17       |
| BM1684 SoC  | centernet_sail.soc    | centernet_fp32_1b.bmodel   | 3.48      | 1.87          | 46.94        | 1350.56      |
| BM1684 SoC  | centernet_sail.soc    | centernet_int8_1b.bmodel   | 3.53      | 1.29          | 23.11        | 1352.22      |
| BM1684 SoC  | centernet_sail.soc    | centernet_int8_4b.bmodel   | 3.24      | 1.01          | 8.46         | 1352.32      |
| BM1684 SoC  | centernet_bmcv.soc    | centernet_fp32_1b.bmodel   | 5.46      | 1.47          | 46.27        | 1179.17      |
| BM1684 SoC  | centernet_bmcv.soc    | centernet_int8_1b.bmodel   | 5.42      | 1.46          | 22.51        | 1180.33      |
| BM1684 SoC  | centernet_bmcv.soc    | centernet_int8_4b.bmodel   | 5.37      | 1.50          | 7.9          | 1189.37      |
| BM1684X SoC | centernet_opencv.py   | centernet_fp32_1b.bmodel   | 15.31     | 40.02         | 71.32        | 817.13       |
| BM1684X SoC | centernet_opencv.py   | centernet_int8_1b.bmodel   | 15.37     | 40.12         | 19.9         | 802.71       |
| BM1684X SoC | centernet_opencv.py   | centernet_int8_4b.bmodel   | 15.23     | 38.37         | 18.32        | 836.56       |
| BM1684X SoC | centernet_bmcv.py     | centernet_fp32_1b.bmodel   | 2.98      | 1.99          | 60.61        | 817.42       |
| BM1684X SoC | centernet_bmcv.py     | centernet_int8_1b.bmodel   | 2.99      | 1.99          | 8.97         | 801.77       |
| BM1684X SoC | centernet_bmcv.py     | centernet_int8_4b.bmodel   | 2.80      | 1.84          | 8.15         | 841.5        |
| BM1684X SoC | centernet_sail.soc    | centernet_fp32_1b.bmodel   | 2.9       | 1.63          | 57.1         | 1356.13      |
| BM1684X SoC | centernet_sail.soc    | centernet_int8_1b.bmodel   | 2.92      | 1.63          | 5.6          | 1357.22      |
| BM1684X SoC | centernet_sail.soc    | centernet_int8_4b.bmodel   | 2.64      | 1.57          | 5.08         | 1358.17      |
| BM1684X SoC | centernet_bmcv.soc    | centernet_fp32_1b.bmodel   | 4.77      | 0.75          | 56.53        | 1184.43      |
| BM1684X SoC | centernet_bmcv.soc    | centernet_int8_1b.bmodel   | 4.8       | 0.75          | 4.99         | 1185.35      |
| BM1684X SoC | centernet_bmcv.soc    | centernet_int8_4b.bmodel   | 4.47      | 0.68          | 4.52         | 1185.1       |
| BM1688 SoC  | centernet_opencv.py   | centernet_fp32_1b.bmodel   | 19.49     | 52.54         | 349.78       | 1116.06      |
| BM1688 SoC  | centernet_opencv.py   | centernet_fp16_1b.bmodel   | 19.39     | 52.20         | 64.80        | 1093.04      |
| BM1688 SoC  | centernet_opencv.py   | centernet_int8_1b.bmodel   | 19.47     | 52.17         | 38.51        | 1083.13      |
| BM1688 SoC  | centernet_opencv.py   | centernet_int8_4b.bmodel   | 19.39     | 50.10         | 36.29        | 1113.67      |
| BM1688 SoC  | centernet_bmcv.py     | centernet_fp32_1b.bmodel   | 4.58      | 4.52          | 337.37       | 1093.85      |
| BM1688 SoC  | centernet_bmcv.py     | centernet_fp16_1b.bmodel   | 4.53      | 4.51          | 52.45        | 1087.40      |
| BM1688 SoC  | centernet_bmcv.py     | centernet_int8_1b.bmodel   | 4.57      | 4.51          | 25.96        | 1063.44      |
| BM1688 SoC  | centernet_bmcv.py     | centernet_int8_4b.bmodel   | 4.29      | 4.20          | 24.56        | 1129.51      |
| BM1688 SoC  | centernet_bmcv.soc    | centernet_fp32_1b.bmodel   | 6.10      | 1.70          | 332.19       | 1620.93      |
| BM1688 SoC  | centernet_bmcv.soc    | centernet_fp16_1b.bmodel   | 6.11      | 1.71          | 47.42        | 1620.83      |
| BM1688 SoC  | centernet_bmcv.soc    | centernet_int8_1b.bmodel   | 6.13      | 1.70          | 20.83        | 1622.64      |
| BM1688 SoC  | centernet_bmcv.soc    | centernet_int8_4b.bmodel   | 5.85      | 1.55          | 20.03        | 1623.05      |
| BM1688 SoC  | centernet_sail.soc    | centernet_fp32_1b.bmodel   | 4.18      | 3.19          | 333.22       | 1859.35      |
| BM1688 SoC  | centernet_sail.soc    | centernet_fp16_1b.bmodel   | 4.21      | 3.19          | 48.40        | 1859.22      |
| BM1688 SoC  | centernet_sail.soc    | centernet_int8_1b.bmodel   | 4.21      | 3.18          | 21.79        | 1861.07      |
| BM1688 SoC  | centernet_sail.soc    | centernet_int8_4b.bmodel   | 3.85      | 3.04          | 21.08        | 1861.63      |
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
    parser.add_argument('--bmodel', type=str, default='centernet_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='centernet_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../python/log/soc_bmcv_centernet_fp32_1b.bmodel_debug.log')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    benchmark_path = current_dir + "/benchmark.txt"
    args = argsparser()
    platform = args.target + " SoC" if args.platform == "soc" else args.target + " PCIe"
    min_width = 17
    
    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^11}|{:^19}|{:^25}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
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
    benchmark_str = "|{:^11}|{:^19}|{:^25}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
