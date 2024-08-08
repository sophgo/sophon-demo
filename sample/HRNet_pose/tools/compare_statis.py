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
|    测试平台   | 测试程序       | 测试模型                                      | decode_time | hrnet_preprocess_time | hrnet_inference_time | hrnet_postprocess_time |
| ----------- |---------------|------------------------------------------------|-------------|-----------------------|----------------------|------------------------|
|   SE7-32    | hrnet_pose.py |  hrnet_w32_256x192_fp32.bmodel                 | 8.39        | 9.21                  | 39.85                | 2.80                  |
|   SE7-32    | hrnet_pose.py |  hrnet_w32_256x192_fp16.bmodel                 | 8.35        | 9.19                  | 10.13                | 2.77                  |               
|   SE7-32    | hrnet_pose.py |  hrnet_w32_256x192_int8.bmodel                 | 8.31        | 9.13                  | 8.81                 | 2.77                  |
|   SE7-32    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp32.bmodel          | 4.44        | 2.19                  | 16.73                | 1.40                  |
|   SE7-32    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp16.bmodel          | 4.49        | 2.21                  | 1.89                 | 1.40                  |
|   SE7-32    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_int8.bmodel          | 4.48        | 2.21                  | 1.26                 | 1.41                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_fp32.bmodel                 | 12.96       | 12.55                 | 161.07               | 3.92                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_fp16.bmodel                 | 12.80       | 12.51                 | 27.95                | 3.91                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_int8.bmodel                 | 13.02       | 12.53                 | 14.74                | 3.94                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp32.bmodel           | 8.71        | 3.30                  | 76.28                | 1.55                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp16.bmodel           | 8.56        | 3.29                  | 9.77                 | 1.54                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_int8.bmodel           | 10.14       | 3.28                  | 3.26                 | 1.52                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_fp32_2core.bmodel           | 13.00       | 12.64                 | 126.32               | 3.93                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_fp16_2core.bmodel           | 12.76       | 12.51                 | 27.43                | 3.95                  |
|   SE9-16    | hrnet_pose.py |  hrnet_w32_256x192_int8_2core.bmodel           | 12.80       | 12.47                 | 14.43                | 3.91                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp32_2core.bmodel     | 9.50        | 3.31                  | 58.85                | 1.54                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp16_2core.bmodel     | 8.29        | 3.30                  | 9.51                 | 1.54                  |
|   SE9-16    | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_int8_2core.bmodel     | 8.44        | 3.29                  | 3.14                 | 1.53                  |
|   SE9-8     | hrnet_pose.py |  hrnet_w32_256x192_fp32.bmodel                 | 12.90       | 12.41                 | 160.91               | 3.81                  |
|   SE9-8     | hrnet_pose.py |  hrnet_w32_256x192_fp16.bmodel                 | 12.69       | 12.46                 | 27.88                | 3.77                  |
|   SE9-8     | hrnet_pose.py |  hrnet_w32_256x192_int8.bmodel                 | 12.86       | 12.41                 | 14.75                | 3.78                  |
|   SE9-8     | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp32.bmodel           | 8.83        | 3.33                  | 76.26                | 1.73                  |
|   SE9-8     | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_fp16.bmodel           | 9.05        | 3.32                  | 9.75                 | 1.71                  |
|   SE9-8     | hrnet_pose_bmcv.soc |  hrnet_w32_256x192_int8.bmodel           | 9.15        | 3.31                  | 3.24                 | 1.71                  |
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
    parser.add_argument('--bmodel', type=str, default='hrnet_w32_256x192_int8.bmodel')
    parser.add_argument('--program', type=str, default='hrnet_pose.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_hrnet_w32_256x192_int8.bmodel_python_test.log')
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
            table_data["bmodel"].append(match.group(3))
            table_data["decode"].append(float(match.group(4)))
            table_data["preprocess"].append(float(match.group(5)))
            table_data["inference"].append(float(match.group(6)))
            table_data["postprocess"].append(float(match.group(7)))

    if args.platform == "soc":
        if args.target == "BM1684X":
            platform = "SE7-32"
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
        if abs(statis - extracted_data[key]) / statis > 0.2:
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
        
