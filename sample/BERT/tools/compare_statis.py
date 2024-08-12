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
|    测试平台  |     测试程序      |             测试模型                |tot_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- |  --------- | ---------   | ---------   |
| SE5-16      | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 307.64   | 3.48       | 171.87      | 132.26      |
| SE5-16      | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 168.89   | 3.46       | 147.06      | 18.359      |
| SE5-16      | bert_sail.soc    | bert4torch_output_fp32_1b.bmodel    | 19.52    | 19.11      | 0.35        | 0.022       |
| SE5-16      | bert_sail.soc    | bert4torch_output_fp32_8b.bmodel    | 20.21    | 19.34      | 0.830       | 0.021       |
| SE7-32      | bert_sail.py     | bert4torch_output_fp32_1b.bmodel    | 225.16   | 3.50       | 92.25       | 129.39      |
| SE7-32      | bert_sail.py     | bert4torch_output_fp32_8b.bmodel    | 109.60   | 3.5        | 87.76       | 18.36       |
| SE7-32      | bert_sail.py     | bert4torch_output_fp16_1b.bmodel    | 141.59   | 3.5        | 9.50        | 128.57      |
| SE7-32      | bert_sail.py     | bert4torch_output_fp16_8b.bmodel    | 27.64    | 3.4        | 5.84        | 18.325      |
| SE7-32      | bert_sail.soc    | bert4torch_output_fp32_1b.bmodel    | 19.45    | 19.14      | 0.028       | 0.022       |
| SE7-32      | bert_sail.soc    | bert4torch_output_fp32_8b.bmodel    | 19.28    | 19.15      | 0.078       | 0.021       |
| SE7-32      | bert_sail.soc    | bert4torch_output_fp16_1b.bmodel    | 19.87    | 19.59      | 0.218       | 0.020       |
| SE7-32      | bert_sail.soc    | bert4torch_output_fp16_8b.bmodel    | 19.73    | 19.62      | 0.642       | 0.019       |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp32_1b.bmodel|     473.89      |      4.92       |     274.80      |     194.11      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp32_8b.bmodel|     297.54      |      4.90       |     264.84      |      27.79      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp16_1b.bmodel|     243.83      |      4.90       |      41.54      |     197.32      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp16_8b.bmodel|      67.42      |      4.94       |      34.47      |      28.00      |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp32_1b.bmodel|     298.32      |      27.11      |     271.15      |      0.04       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp32_8b.bmodel|     291.28      |      26.79      |     264.45      |      0.03       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp16_1b.bmodel|      67.56      |      26.86      |      40.66      |      0.04       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp16_8b.bmodel|      61.32      |      26.96      |      34.33      |      0.03       |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp32_1b_2core.bmodel|     381.68      |      4.90       |     180.75      |     195.95      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp32_8b_2core.bmodel|     185.70      |      4.98       |     152.89      |      27.83      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp16_1b_2core.bmodel|     234.26      |      4.92       |      31.72      |     197.54      |
|   SE9-16    |   bert_sail.py    |bert4torch_output_fp16_8b_2core.bmodel|      51.49      |      4.89       |      18.75      |      27.84      |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp32_1b_2core.bmodel|     206.84      |      26.88      |     179.91      |      0.04       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp32_8b_2core.bmodel|     179.37      |      26.73      |     152.61      |      0.03       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp16_1b_2core.bmodel|      57.63      |      26.82      |      30.77      |      0.03       |
|   SE9-16    |   bert_sail.soc   |bert4torch_output_fp16_8b_2core.bmodel|      45.67      |      27.01      |      18.64      |      0.03       |
|    SE9-8    |   bert_sail.soc   |bert4torch_output_fp32_1b.bmodel|     297.99      |      27.09      |     270.85      |      0.04       |
|    SE9-8    |   bert_sail.soc   |bert4torch_output_fp32_8b.bmodel|     290.67      |      26.48      |     264.16      |      0.03       |
|    SE9-8    |   bert_sail.soc   |bert4torch_output_fp16_1b.bmodel|      66.50      |      27.02      |      39.45      |      0.03       |
|    SE9-8    |   bert_sail.soc   |bert4torch_output_fp16_8b.bmodel|      60.02      |      26.85      |      33.13      |      0.03       |
|    SE9-8    |   bert_sail.py    |bert4torch_output_fp32_1b.bmodel|     472.14      |      4.88       |     272.00      |     195.19      |
|    SE9-8    |   bert_sail.py    |bert4torch_output_fp32_8b.bmodel|     297.02      |      4.89       |     264.52      |      27.60      |
|    SE9-8    |   bert_sail.py    |bert4torch_output_fp16_1b.bmodel|     242.85      |      4.89       |      40.59      |     197.30      |
|    SE9-8    |   bert_sail.py    |bert4torch_output_fp16_8b.bmodel|      65.86      |      4.92       |      33.27      |      27.66      |
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
    'decode': re.compile(r'\[.*tots*\]  loops:.*avg: ([\d.]+) ms'),
    'preprocess': re.compile(r'\[.*preprocess*\]  loops:.*avg: ([\d.]+) ms'),
    'inference': re.compile(r'\[.*inference*\]  loops:.*avg: ([\d.]+) ms'),
    'postprocess': re.compile(r'\[.*postprocess*\]  loops:.*avg: ([\d.]+) ms'),
}

patterns_python = {
    'decode': re.compile(r'avg_tot_time ([\d.]+)'),
    'preprocess': re.compile(r'avg_pre_time ([\d.]+)'),
    'inference': re.compile(r'avg_infer_time ([\d.]+)'),
    'postprocess': re.compile(r'avg_post_time ([\d.]+)'),
}

def extract_times(text, patterns,t):
    results = {}
    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            results[key] = round(float(match.group(1))*t,2)
    return results


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--target', type=str, default='BM1684X', help='path of label json')
    parser.add_argument('--platform', type=str, default='soc', help='path of result json')
    parser.add_argument('--bmodel', type=str, default='yolov8_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov8_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov8_fp32_1b.bmodel_python_test.log')
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
           "platform", "program", "bmodel", "tot_time", "preprocess_time", "inference_time", "postprocess_time", width=min_width)
            f.write(benchmark_str)
            
    with open(args.input, "r") as f:
        data = f.read()
    if args.language == "python":    
        extracted_data = extract_times(data, patterns_python,1000)
    elif args.language == "cpp":
        extracted_data = extract_times(data, patterns_cpp,1)
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
        elif abs(statis - extracted_data[key]) / statis > 0.4:
            print("{:} time, diff ratio > 0.4".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^25}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
