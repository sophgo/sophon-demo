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
|  测试平台     |    测试程序         |                 测试模型                        |decode_time|preprocess_time|inference_time |postprocess_time| 
| -----------  | ------------------- | ---------------------------------------------- | --------- | ------------- | ------------- | -------------- |
|    SE5-16    | segformer_opencv.py | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   110.23  |      23.15    |     389.67    |     182.70     |
|    SE5-16    | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   140.67  |      5.93     |     369.88    |     141.78     |
|    SE5-16    | segformer_bmcv.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   126.51  |      1.52     |     365.50    |     261.71     |
|    SE5-16    | segformer_sail.soc  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   106.44  |      7.14     |     365.86    |     259.91     |
|    SE7-32    |segformer_opencv.py  |segformer.b0.512x1024.city.160k_fp32_1b.bmodel  |   145.98  |      21.14    |     320.01    |     166.92     |
|    SE7-32    |segformer_opencv.py  |segformer.b0.512x1024.city.160k_fp16_1b.bmodel  |   153.89  |      21.21    |      74.10    |     167.25     |
|    SE7-32    | segformer_bmcv.py   |segformer.b0.512x1024.city.160k_fp32_1b.bmodel  |   154.53  |      5.19     |     303.69    |     123.74     |
|    SE7-32    | segformer_bmcv.py   |segformer.b0.512x1024.city.160k_fp16_1b.bmodel  |   146.50  |      5.15     |      57.59    |     123.58     |
|    SE7-32    |segformer_bmcv.soc   |segformer.b0.512x1024.city.160k_fp32_1b.bmodel  |   161.21  |      1.54     |     300.34    |     249.28     |
|    SE7-32    |segformer_bmcv.soc   |segformer.b0.512x1024.city.160k_fp16_1b.bmodel  |   148.56  |      1.54     |      54.27    |     249.03     |
|    SE7-32    |segformer_sail.soc   |segformer.b0.512x1024.city.160k_fp32_1b.bmodel  |   154.23  |      5.95     |     300.70    |     250.16     |
|    SE7-32    |segformer_sail.soc   |segformer.b0.512x1024.city.160k_fp16_1b.bmodel  |   152.21  |      5.96     |      54.60    |     249.54     |
|    SE9-16    |segformer_opencv.py  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   156.92  |      27.17    |     447.52    |   248.12       |
|    SE9-16    |segformer_opencv.py  | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   162.62  |      27.15    |     153.11    |   250.05       |
|    SE9-16    | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   183.98  |      12.02    |     419.88    |   194.48       |
|    SE9-16    | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   185.24  |      12.03    |     125.12    |   194.61       |
|    SE9-16    |segformer_bmcv.soc   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   207.84  |      4.28     |     413.60    |     344.03     |
|    SE9-16    |segformer_bmcv.soc   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   207.85  |      4.28     |     118.95    |     348.56     |
|    SE9-16    |segformer_sail.soc   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   179.58  |      13.25    |     414.41    |     338.35     |
|    SE9-16    |segformer_sail.soc   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   181.23  |      13.22    |     119.71    |     338.43     |
|    SE9-16    |segformer_opencv.py  |segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel|   160.19    |    28.94    |   322.68    |   253.40     |
|    SE9-16    |segformer_opencv.py  |segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel|   163.17    |    27.50    |   139.36    |   252.66     |
|    SE9-16    | segformer_bmcv.py   |segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel|   183.64    |    12.06    |   294.23    |   195.08     |
|    SE9-16    | segformer_bmcv.py   |segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel|   184.64    |    12.06    |   110.32    |   194.97     |
|    SE9-16    |segformer_bmcv.soc   |segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel|   206.37    |    4.28     |   288.10    |   347.99     |
|    SE9-16    |segformer_bmcv.soc   |segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel|   206.85    |    4.28     |   104.27    |   345.63     |
|    SE9-16    |segformer_sail.soc   |segformer.b0.512x1024.city.160k_fp32_1b_2core.bmodel|   184.14    |    13.25    |   288.70    |   339.33     |
|    SE9-16    |segformer_sail.soc   |segformer.b0.512x1024.city.160k_fp16_1b_2core.bmodel|   186.15    |    13.27    |   105.01    |   338.84     |
|    SE9-8     |segformer_opencv.py  | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   157.53  |      29.47    |     508.93    |   223.66       |
|    SE9-8     |segformer_opencv.py  | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   164.70  |      29.17    |     190.92    |   224.96       |
|    SE9-8     | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   185.69  |      11.82    |     481.06    |   192.43       |
|    SE9-8     | segformer_bmcv.py   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   185.56  |      11.89    |     162.81    |   197.55       |
|    SE9-8     |segformer_sail.soc   | segformer.b0.512x1024.city.160k_fp32_1b.bmodel |   183.28  |      12.80    |     413.78    |   341.59       |
|    SE9-8     |segformer_sail.soc   | segformer.b0.512x1024.city.160k_fp16_1b.bmodel |   183.28  |      12.80    |     413.78    |     341.59     |
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
    parser.add_argument('--bmodel', type=str, default='resnet_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='resnet_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_resnet_fp32_1b.bmodel_python_test.log')
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
            benchmark_str = "|{:^13}|{:^19}|{:^35}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
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
        
    benchmark_str = "|{:^13}|{:^19}|{:^35}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
