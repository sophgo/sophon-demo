#===----------------------------------------------------------------------===#
#
# Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
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
|    测试平台  |     测试程序      |             测试模型                |decode_time    |preprocess_time  |inference_time   |postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | --------      | ---------       | ---------        | --------- |
|   SE5-16    | ppyoloe_opencv.py |      ppyoloe_fp32_1b.bmodel       |      15.02      |      43.29      |      45.16      |      12.46      |
|   SE5-16    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      3.76       |      3.63       |      34.07      |      12.64      |
|   SE5-16    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      4.84       |      1.08       |      30.71      |      8.57       |
|   SE7-32    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      3.09       |      2.51       |      30.25      |      13.39      |
|   SE7-32    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      4.37       |      1.00       |      27.37      |      8.99       |
|   SE7-32    |  ppyoloe_bmcv.py  |      ppyoloe_fp16_1b.bmodel       |      2.89       |      2.50       |      9.67       |      13.50      |
|   SE7-32    | ppyoloe_bmcv.soc  |      ppyoloe_fp16_1b.bmodel       |      4.41       |      0.98       |      6.82       |      9.02       |
|   SE7-32    |  ppyoloe_bmcv.py  |      ppyoloe_int8_1b.bmodel       |      2.89       |      2.50       |      6.54       |      13.67      |
|   SE7-32    | ppyoloe_bmcv.soc  |      ppyoloe_int8_1b.bmodel       |      4.38       |      0.98       |      3.72       |      9.00       |
|   SE9-16    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      4.04       |      4.60       |     123.36      |      18.40      |
|   SE9-16    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      5.68       |      2.20       |     119.48      |      12.25      |
|   SE9-16    |  ppyoloe_bmcv.py  |      ppyoloe_fp16_1b.bmodel       |      4.04       |      4.61       |      35.33      |      18.19      |
|   SE9-16    | ppyoloe_bmcv.soc  |      ppyoloe_fp16_1b.bmodel       |      5.66       |      2.19       |      31.52      |      12.24      |
|   SE9-16    |  ppyoloe_bmcv.py  |      ppyoloe_int8_1b.bmodel       |      4.01       |      4.61       |      10.94      |      18.39      |
|   SE9-16    | ppyoloe_bmcv.soc  |      ppyoloe_int8_1b.bmodel       |      5.70       |      2.20       |      7.12       |      12.28      |
|   SE9-16    |  ppyoloe_bmcv.py  |   ppyoloe_int8_1b_2core.bmodel    |      4.01       |      4.60       |      9.30       |      18.40      |
|   SE9-16    | ppyoloe_bmcv.soc  |   ppyoloe_int8_1b_2core.bmodel    |      5.68       |      2.20       |      5.51       |      12.28      |
|    SE9-8    |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      9.12       |      9.64       |     130.23      |      20.94      |
|    SE9-8    | ppyoloe_bmcv.soc  |      ppyoloe_fp32_1b.bmodel       |      6.38       |      7.70       |     122.91      |      13.45      |
|    SE9-8    |  ppyoloe_bmcv.py  |      ppyoloe_fp16_1b.bmodel       |      4.34       |      7.24       |      40.24      |      20.68      |
|    SE9-8    | ppyoloe_bmcv.soc  |      ppyoloe_fp16_1b.bmodel       |      6.39       |      4.43       |      33.66      |      13.73      |
|    SE9-8    |  ppyoloe_bmcv.py  |      ppyoloe_int8_1b.bmodel       |      4.25       |      6.82       |      13.71      |      19.94      |
|    SE9-8    | ppyoloe_bmcv.soc  |      ppyoloe_int8_1b.bmodel       |      6.29       |      3.09       |      7.99       |      13.30      |
|   SRM1-20   | ppyoloe_opencv.py |      ppyoloe_fp32_1b.bmodel       |      13.23      |      66.43      |     109.16      |      8.64       |
|   SRM1-20   | ppyoloe_opencv.py |      ppyoloe_fp16_1b.bmodel       |      13.39      |      66.59      |      83.91      |      8.74       |
|   SRM1-20   |  ppyoloe_bmcv.py  |      ppyoloe_fp32_1b.bmodel       |      23.82      |      4.43       |      86.33      |      9.52       |
|   SRM1-20   |  ppyoloe_bmcv.py  |      ppyoloe_fp16_1b.bmodel       |      23.78      |      4.46       |      61.68      |      9.14       |
|   SRM1-20   | ppyoloe_bmcv.pcie |      ppyoloe_fp32_1b.bmodel       |      22.13      |      1.98       |      32.41      |      53.15      |
|   SRM1-20   | ppyoloe_bmcv.pcie |      ppyoloe_fp16_1b.bmodel       |      12.04      |      1.80       |      8.02       |      23.27      |
|   SRM1-20   | ppyoloe_sail.pcie |      ppyoloe_fp32_1b.bmodel       |      13.72      |      2.84       |      51.68      |      4.50       |
|   SRM1-20   | ppyoloe_sail.pcie |      ppyoloe_fp16_1b.bmodel       |      23.50      |      3.40       |      56.11      |      4.91       |
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
    parser.add_argument('--bmodel', type=str, default='yolov5s_v6.1_3output_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov5_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov5s_v6.1_3output_fp32_1b.bmodel_python_test.log')
    parser.add_argument('--use_cpu_opt', action="store_true", default=False, help='accelerate cpu postprocess')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    args = argsparser()
    if args.use_cpu_opt:
        benchmark_path = current_dir + "/benchmark_cpu_opt.txt"
        baseline = baseline_cpu_opt
    else:
        benchmark_path = current_dir + "/benchmark.txt"
        
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
        if abs(statis - extracted_data[key]) / statis > 0.4:
            print("{:} time, diff ratio > 0.4".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^35}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
