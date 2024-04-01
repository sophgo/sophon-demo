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
|    测试平台  |     测试程序      |             测试模型                |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
| BM1684 SoC  | yolov34_opencv.py | yolov3_fp32_1b.bmodel | 15.3     | 25.1          | 112.0         | 159.2      |
| BM1684 SoC  | yolov34_opencv.py | yolov3_int8_1b.bmodel | 15.1     | 24.7          | 63.5          | 157.6      |
| BM1684 SoC  | yolov34_opencv.py | yolov3_int8_4b.bmodel | 15.0     | 23.2          | 27.5          | 152.7      |
| BM1684 SoC  | yolov34_bmcv.py   | yolov3_fp32_1b.bmodel | 3.6      | 2.8           | 107.8         | 166.3      |
| BM1684 SoC  | yolov34_bmcv.py   | yolov3_int8_1b.bmodel | 3.6      | 2.8           | 59.0          | 162.7      |
| BM1684 SoC  | yolov34_bmcv.py   | yolov3_int8_4b.bmodel | 3.4      | 2.6           | 23.9          | 159.1      |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov3_fp32_1b.bmodel | 5.1      | 1.5           | 102.2         | 20.1       |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov3_int8_1b.bmodel | 5.1      | 1.6           | 53.6          | 20.3       |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov3_int8_4b.bmodel | 4.9      | 1.5           | 19.3          | 20.0       |
| BM1684 SoC  | yolov34_sail.soc  | yolov3_fp32_1b.bmodel | 4.5      | 2.9           | 103.0         | 18.5       |
| BM1684 SoC  | yolov34_sail.soc  | yolov3_int8_1b.bmodel | 3.3      | 2.9           | 54.5          | 18.7       |
| BM1684 SoC  | yolov34_sail.soc  | yolov3_int8_4b.bmodel | 3.1      | 2.7           | 20.1          | 18.6       |
| BM1684X SoC | yolov34_opencv.py | yolov3_fp32_1b.bmodel | 3.7      | 8.8           | 194.5         | 28.5       |
| BM1684X SoC | yolov34_opencv.py | yolov3_fp16_1b.bmodel | 3.7      | 8.2           | 39.3          | 23.5       |
| BM1684X SoC | yolov34_opencv.py | yolov3_int8_1b.bmodel | 3.6      | 7.9           | 21.9          | 19.6       |
| BM1684X SoC | yolov34_opencv.py | yolov3_int8_4b.bmodel | 3.2      | 8.5           | 5.4           | 21.6       |
| BM1684X SoC | yolov34_bmcv.py   | yolov3_fp32_1b.bmodel | 3.1      | 1.7           | 167.4         | 26.0       |
| BM1684X SoC | yolov34_bmcv.py   | yolov3_fp16_1b.bmodel | 3.4      | 1.9           | 34.2          | 26.5       |
| BM1684X SoC | yolov34_bmcv.py   | yolov3_int8_1b.bmodel | 3.9      | 2.5           | 16.6          | 22.2       |
| BM1684X SoC | yolov34_bmcv.py   | yolov3_int8_4b.bmodel | 3.9      | 2.4           | 4.2           | 23.3       |
| BM1684X SoC | yolov34_bmcv.soc  | yolov3_fp32_1b.bmodel | 4.4      | 1.2           | 169.0         | 20.0       |
| BM1684X SoC | yolov34_bmcv.soc  | yolov3_fp16_1b.bmodel | 4.4      | 1.2           | 26.3          | 19.9       |
| BM1684X SoC | yolov34_bmcv.soc  | yolov3_int8_1b.bmodel | 4.3      | 1.2           | 9.3           | 19.6       |
| BM1684X SoC | yolov34_bmcv.soc  | yolov3_int8_4b.bmodel | 3.9      | 1.0           | 9.7           | 15.1       |
| BM1684X SoC | yolov34_sail.soc  | yolov3_fp32_1b.bmodel | 4.0      | 1.6           | 153.1         | 14.8       |
| BM1684X SoC | yolov34_sail.soc  | yolov3_fp16_1b.bmodel | 3.9      | 1.5           | 32.5          | 14.8       |
| BM1684X SoC | yolov34_sail.soc  | yolov3_int8_1b.bmodel | 4.0      | 1.6           | 16.2          | 14.8       |
| BM1684X SoC | yolov34_sail.soc  | yolov3_int8_4b.bmodel | 3.9      | 1.4           | 15.4          | 9.8        |
| BM1688 SoC  | yolov34_opencv.py | yolov3_fp32_1b.bmodel | 19.3     | 35.6          | 786.4         | 218.9      |
| BM1688 SoC  | yolov34_opencv.py | yolov3_fp16_1b.bmodel | 19.2     | 29.6          | 147.1         | 212.6      |
| BM1688 SoC  | yolov34_opencv.py | yolov3_int8_1b.bmodel | 19.2     | 28.7          | 43.2          | 215.8      |
| BM1688 SoC  | yolov34_opencv.py | yolov3_int8_4b.bmodel | 19.3     | 31.8          | 40.6          | 219.6      |
| BM1688 SoC  | yolov34_bmcv.py   | yolov3_fp32_1b.bmodel | 4.6      | 5.0           | 780.3         | 227.6      |
| BM1688 SoC  | yolov34_bmcv.py   | yolov3_fp16_1b.bmodel | 4.6      | 5.0           | 143.3         | 227.2      |
| BM1688 SoC  | yolov34_bmcv.py   | yolov3_int8_1b.bmodel | 4.6      | 5.0           | 39.2          | 227.8      |
| BM1688 SoC  | yolov34_bmcv.py   | yolov3_int8_4b.bmodel | 4.4      | 4.7           | 35.6          | 239.9      |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov3_fp32_1b.bmodel | 9.7      | 1.9           | 772.7         | 28.0       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov3_fp16_1b.bmodel | 6.0      | 1.9           | 135.7         | 28.1       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov3_int8_1b.bmodel | 6.0      | 1.9           | 31.8          | 28.3       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov3_int8_4b.bmodel | 5.8      | 1.8           | 29.4          | 28.1       |
| BM1688 SoC  | yolov34_sail.soc  | yolov3_fp32_1b.bmodel | 7.6      | 4.9           | 774.0         | 25.7       |
| BM1688 SoC  | yolov34_sail.soc  | yolov3_fp16_1b.bmodel | 4.0      | 4.9           | 137.0         | 25.7       |
| BM1688 SoC  | yolov34_sail.soc  | yolov3_int8_1b.bmodel | 4.2      | 5.0           | 33.1          | 26.0       |
| BM1688 SoC  | yolov34_sail.soc  | yolov3_int8_4b.bmodel | 4.0      | 4.7           | 30.6          | 25.8       |
| BM1684 SoC  | yolov34_opencv.py | yolov4_fp32_1b.bmodel | 2.9      | 2.0           | 84.2          | 10.4       |
| BM1684 SoC  | yolov34_opencv.py | yolov4_int8_1b.bmodel | 2.8      | 1.8           | 34.8          | 7.8        |
| BM1684 SoC  | yolov34_opencv.py | yolov4_int8_4b.bmodel | 2.7      | 2.0           | 20.2          | 6.0        |
| BM1684 SoC  | yolov34_bmcv.py   | yolov4_fp32_1b.bmodel | 2.7      | 2.1           | 81.6          | 10.4       |
| BM1684 SoC  | yolov34_bmcv.py   | yolov4_int8_1b.bmodel | 2.6      | 1.9           | 32.5          | 10.4       |
| BM1684 SoC  | yolov34_bmcv.py   | yolov4_int8_4b.bmodel | 2.3      | 1.6           | 16.8          | 5.4        |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov4_fp32_1b.bmodel | 4.9      | 1.6           | 77.5          | 8.3        |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov4_int8_1b.bmodel | 4.7      | 1.5           | 28.6          | 7.9        |
| BM1684 SoC  | yolov34_bmcv.soc  | yolov4_int8_4b.bmodel | 3.6      | 1.4           | 13.8          | 6.2        |
| BM1684 SoC  | yolov34_sail.soc  | yolov4_fp32_1b.bmodel | 4.2      | 14.2          | 80.6          | 5.4        |
| BM1684 SoC  | yolov34_sail.soc  | yolov4_int8_1b.bmodel | 4.1      | 11.8          | 31.4          | 5.1        |
| BM1684 SoC  | yolov34_sail.soc  | yolov4_int8_4b.bmodel | 3.5      | 2.6           | 15.6          | 2.7        |
| BM1684X SoC | yolov34_opencv.py | yolov4_fp32_1b.bmodel | 3.0      | 2.1           | 73.7          | 11.0       |
| BM1684X SoC | yolov34_opencv.py | yolov4_fp16_1b.bmodel | 3.0      | 2.5           | 20.6          | 5.5        |
| BM1684X SoC | yolov34_opencv.py | yolov4_int8_1b.bmodel | 3.2      | 2.9           | 15.2          | 7.6        |
| BM1684X SoC | yolov34_opencv.py | yolov4_int8_4b.bmodel | 3.0      | 2.8           | 15.2          | 14.2       |
| BM1684X SoC | yolov34_bmcv.py   | yolov4_fp32_1b.bmodel | 2.3      | 1.3           | 72.0          | 10.7       |
| BM1684X SoC | yolov34_bmcv.py   | yolov4_fp16_1b.bmodel | 2.1      | 1.1           | 17.1          | 8.5        |
| BM1684X SoC | yolov34_bmcv.py   | yolov4_int8_1b.bmodel | 2.1      | 1.1           | 9.7           | 5.5        |
| BM1684X SoC | yolov34_bmcv.py   | yolov4_int8_4b.bmodel | 2.0      | 1.0           | 11.8          | 12.6       |
| BM1684X SoC | yolov34_bmcv.soc  | yolov4_fp32_1b.bmodel | 4.4      | 0.7           | 67.2          | 8.5        |
| BM1684X SoC | yolov34_bmcv.soc  | yolov4_fp16_1b.bmodel | 4.1      | 0.6           | 13.1          | 8.1        |
| BM1684X SoC | yolov34_bmcv.soc  | yolov4_int8_1b.bmodel | 3.6      | 0.6           | 5.9           | 7.3        |
| BM1684X SoC | yolov34_bmcv.soc  | yolov4_int8_4b.bmodel | 3.1      | 0.5           | 5.4           | 6.2        |
| BM1684X SoC | yolov34_sail.soc  | yolov4_fp32_1b.bmodel | 4.0      | 12.3          | 70.6          | 5.4        |
| BM1684X SoC | yolov34_sail.soc  | yolov4_fp16_1b.bmodel | 3.5      | 8.7           | 15.9          | 4.4        |
| BM1684X SoC | yolov34_sail.soc  | yolov4_int8_1b.bmodel | 3.2      | 6.2           | 8.3           | 2.5        |
| BM1684X SoC | yolov34_sail.soc  | yolov4_int8_4b.bmodel | 3.2      | 2.3           | 8.3           | 3.1        |
| BM1688 SoC  | yolov34_opencv.py | yolov4_fp32_1b.bmodel | 19.1     | 16.0          | 380.3         | 61.8       |
| BM1688 SoC  | yolov34_opencv.py | yolov4_fp16_1b.bmodel | 19.5     | 15.6          | 99.3          | 63.0       |
| BM1688 SoC  | yolov34_opencv.py | yolov4_int8_1b.bmodel | 19.2     | 15.7          | 23.2          | 61.7       |
| BM1688 SoC  | yolov34_opencv.py | yolov4_int8_4b.bmodel | 19.3     | 18.1          | 21.0          | 64.6       |
| BM1688 SoC  | yolov34_bmcv.py   | yolov4_fp32_1b.bmodel | 4.6      | 3.9           | 378.4         | 61.9       |
| BM1688 SoC  | yolov34_bmcv.py   | yolov4_fp16_1b.bmodel | 4.6      | 3.9           | 97.2          | 62.0       |
| BM1688 SoC  | yolov34_bmcv.py   | yolov4_int8_1b.bmodel | 4.5      | 4.0           | 21.3          | 61.8       |
| BM1688 SoC  | yolov34_bmcv.py   | yolov4_int8_4b.bmodel | 4.3      | 3.6           | 18.4          | 61.8       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov4_fp32_1b.bmodel | 5.9      | 1.4           | 374.1         | 11.2       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov4_fp16_1b.bmodel | 5.9      | 1.4           | 92.9          | 11.2       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov4_int8_1b.bmodel | 5.8      | 1.4           | 17.0          | 11.2       |
| BM1688 SoC  | yolov34_bmcv.soc  | yolov4_int8_4b.bmodel | 5.7      | 1.3           | 15.2          | 11.0       |
| BM1688 SoC  | yolov34_sail.soc  | yolov4_fp32_1b.bmodel | 4.0      | 3.5           | 374.9         | 10.0       |
| BM1688 SoC  | yolov34_sail.soc  | yolov4_fp16_1b.bmodel | 4.1      | 3.7           | 93.8          | 10.1       |
| BM1688 SoC  | yolov34_sail.soc  | yolov4_int8_1b.bmodel | 4.2      | 3.6           | 17.8          | 10.1       |
| BM1688 SoC  | yolov34_sail.soc  | yolov4_int8_4b.bmodel | 4.0      | 3.3           | 15.8          | 10.0       |
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
    parser.add_argument('--bmodel', type=str, default='yolov3_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov34_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov3_fp32_1b.bmodel_python_test.log')
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
        
