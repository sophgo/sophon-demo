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
|    测试平台  |     测试程序      |             测试模型               |preprocess_time  |inference_time   |postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | ----------------| ----------------| --------------- |
|   SE7-32    |deepsort_opencv.py |     extractor_fp32_1b.bmodel      |      2.19       |      3.11       |      94.35      |
|   SE7-32    |deepsort_opencv.py |     extractor_fp32_4b.bmodel      |      2.19       |      2.66       |     148.60      |
|   SE7-32    |deepsort_opencv.py |     extractor_fp16_1b.bmodel      |      2.21       |      1.48       |      75.69      |
|   SE7-32    |deepsort_opencv.py |     extractor_fp16_4b.bmodel      |      2.15       |      0.76       |      66.93      |
|   SE7-32    |deepsort_opencv.py |     extractor_int8_1b.bmodel      |      2.17       |      1.27       |      64.47      |
|   SE7-32    |deepsort_opencv.py |     extractor_int8_4b.bmodel      |      2.14       |      0.65       |      75.33      |
|   SE7-32    | deepsort_bmcv.soc |     extractor_fp32_1b.bmodel      |      0.14       |      2.11       |      4.93       |
|   SE7-32    | deepsort_bmcv.soc |     extractor_fp32_4b.bmodel      |      0.36       |      7.29       |      5.54       |
|   SE7-32    | deepsort_bmcv.soc |     extractor_fp16_1b.bmodel      |      0.13       |      0.51       |      5.25       |
|   SE7-32    | deepsort_bmcv.soc |     extractor_fp16_4b.bmodel      |      0.35       |      0.93       |      5.48       |
|   SE7-32    | deepsort_bmcv.soc |     extractor_int8_1b.bmodel      |      0.13       |      0.28       |      5.82       |
|   SE7-32    | deepsort_bmcv.soc |     extractor_int8_4b.bmodel      |      0.35       |      0.54       |      5.74       |
|   SE9-16    |deepsort_opencv.py |     extractor_fp32_1b.bmodel      |      3.05       |      13.57      |      79.64      |
|   SE9-16    |deepsort_opencv.py |     extractor_fp32_4b.bmodel      |      3.01       |      13.54      |      75.79      |
|   SE9-16    |deepsort_opencv.py |     extractor_fp16_1b.bmodel      |      3.03       |      3.36       |      74.38      |
|   SE9-16    |deepsort_opencv.py |     extractor_fp16_4b.bmodel      |      3.01       |      2.44       |      76.62      |
|   SE9-16    |deepsort_opencv.py |     extractor_int8_1b.bmodel      |      3.04       |      2.08       |      82.18      |
|   SE9-16    |deepsort_opencv.py |     extractor_int8_4b.bmodel      |      3.00       |      1.20       |      74.43      |
|   SE9-16    | deepsort_bmcv.soc |     extractor_fp32_1b.bmodel      |      0.47       |      12.26      |      6.70       |
|   SE9-16    | deepsort_bmcv.soc |     extractor_fp32_4b.bmodel      |      1.44       |      43.84      |      6.59       |
|   SE9-16    | deepsort_bmcv.soc |     extractor_fp16_1b.bmodel      |      0.43       |      2.08       |      6.57       |
|   SE9-16    | deepsort_bmcv.soc |     extractor_fp16_4b.bmodel      |      1.42       |      6.17       |      6.60       |
|   SE9-16    | deepsort_bmcv.soc |     extractor_int8_1b.bmodel      |      0.44       |      0.81       |      6.68       |
|   SE9-16    | deepsort_bmcv.soc |     extractor_int8_4b.bmodel      |      1.43       |      1.95       |      6.40       |
|    SE9-8    |deepsort_opencv.py |     extractor_fp32_1b.bmodel      |      3.04       |      12.31      |      66.21      |
|    SE9-8    |deepsort_opencv.py |     extractor_fp32_4b.bmodel      |      3.00       |      13.30      |      54.83      |
|    SE9-8    |deepsort_opencv.py |     extractor_fp16_1b.bmodel      |      3.03       |      3.74       |      49.22      |
|    SE9-8    |deepsort_opencv.py |     extractor_fp16_4b.bmodel      |      3.01       |      2.34       |      48.79      |
|    SE9-8    |deepsort_opencv.py |     extractor_int8_1b.bmodel      |      3.10       |      2.39       |      47.86      |
|    SE9-8    |deepsort_opencv.py |     extractor_int8_4b.bmodel      |      3.00       |      1.25       |      49.77      |
|    SE9-8    | deepsort_bmcv.soc |     extractor_fp32_1b.bmodel      |      0.24       |      10.96      |      7.28       |
|    SE9-8    | deepsort_bmcv.soc |     extractor_fp32_4b.bmodel      |      0.78       |      42.97      |      7.23       |
|    SE9-8    | deepsort_bmcv.soc |     extractor_fp16_1b.bmodel      |      0.23       |      2.41       |      7.14       |
|    SE9-8    | deepsort_bmcv.soc |     extractor_fp16_4b.bmodel      |      0.78       |      5.79       |      7.00       |
|    SE9-8    | deepsort_bmcv.soc |     extractor_int8_1b.bmodel      |      0.23       |      1.06       |      7.02       |
|    SE9-8    | deepsort_bmcv.soc |     extractor_int8_4b.bmodel      |      0.78       |      2.06       |      6.97       |
"""
table_data = {
    "platform": [],
    "program": [],
    "bmodel": [],
    "preprocess": [],
    "inference": [],
    "postprocess": []
}

patterns_cpp = {
    'preprocess': re.compile(r'\[.*extractor preprocess.*\]  loops:.*avg: ([\d.]+) ms'),
    'inference': re.compile(r'\[.*extractor inference.*\]  loops:.*avg: ([\d.]+) ms'),
    'postprocess': re.compile(r'\[.*deepsort postprocess.*\]  loops:.*avg: ([\d.]+) ms'),
}

patterns_python = {
    'preprocess': re.compile(r'Deepsort Tracker Time Info.*?preprocess_time\(ms\): ([\d.]+)', re.DOTALL),
    'inference': re.compile(r'Deepsort Tracker Time Info.*?inference_time\(ms\): ([\d.]+)', re.DOTALL),
    'postprocess': re.compile(r'Deepsort Tracker Time Info.*?postprocess_time\(ms\): ([\d.]+)', re.DOTALL),
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    args = argsparser()
    
    benchmark_path = current_dir + "/benchmark.txt"
        
    for line in baseline.strip().split("\n")[2:]:
        match = re.search(r'\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', line)
        if match:
            table_data["platform"].append(match.group(1))
            table_data["program"].append(match.group(2))
            table_data["bmodel"].append(match.group(3))
            table_data["preprocess"].append(float(match.group(4)))
            table_data["inference"].append(float(match.group(5)))
            table_data["postprocess"].append(float(match.group(6)))

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
            benchmark_str = "|{:^13}|{:^19}|{:^35}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
           "platform", "program", "bmodel",  "preprocess_time", "inference_time", "postprocess_time", width=min_width)
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
        baseline_data["preprocess"] = table_data["preprocess"][match_index]
        baseline_data["inference"] = table_data["inference"][match_index]
        baseline_data["postprocess"] = table_data["postprocess"][match_index]
    for key, statis in baseline_data.items():
        if abs(statis - extracted_data[key]) / statis > 0.4:
            print("{:} time, diff ratio > 0.4".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^35}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
