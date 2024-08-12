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
|    测试平台  |      测试程序       |               测试模型           |   decode_time   | preprocess_time | inference_time  |postprocess_time | 
| ----------- | - --------------- | ------------------------------- | --------------  | ------------    | -----------     | --------------  |
|    SE9-8    | yolov8_opencv.py  |   yolov8s-pose_fp32_1b.bmodel   |      9.42       |      29.57      |     182.99      |      3.49       |
|    SE9-8    | yolov8_opencv.py  |   yolov8s-pose_fp16_1b.bmodel   |      9.42       |      29.57      |      44.01      |      3.46       |
|    SE9-8    | yolov8_opencv.py  |   yolov8s-pose_int8_1b.bmodel   |      9.41       |      30.34      |      15.95      |      3.47       |
|    SE9-8    | yolov8_opencv.py  |   yolov8s-pose_int8_4b.bmodel   |      9.48       |      32.95      |      14.85      |      3.19       |
|    SE9-8    |  yolov8_bmcv.py   |   yolov8s-pose_fp32_1b.bmodel   |      4.14       |      4.59       |     179.02      |      3.49       |
|    SE9-8    |  yolov8_bmcv.py   |   yolov8s-pose_fp16_1b.bmodel   |      4.14       |      4.57       |      39.96      |      3.50       |
|    SE9-8    |  yolov8_bmcv.py   |   yolov8s-pose_int8_1b.bmodel   |      4.13       |      4.59       |      11.90      |      3.51       |
|    SE9-8    |  yolov8_bmcv.py   |   yolov8s-pose_int8_4b.bmodel   |      4.01       |      4.26       |      11.03      |      3.22       |
|    SE9-8    |  yolov8_bmcv.soc  |   yolov8s-pose_fp32_1b.bmodel   |      5.60       |      1.75       |     176.26      |      1.00       |
|    SE9-8    |  yolov8_bmcv.soc  |   yolov8s-pose_fp16_1b.bmodel   |      5.76       |      1.75       |      37.29      |      0.98       |
|    SE9-8    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_1b.bmodel   |      5.66       |      1.75       |      9.26       |      0.98       |
|    SE9-8    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_4b.bmodel   |      5.49       |      1.65       |      9.12       |      0.89       |
|   SE9-16    | yolov8_opencv.py  |   yolov8s-pose_fp32_1b.bmodel   |      9.42       |      29.95      |     178.69      |      3.48       |
|   SE9-16    | yolov8_opencv.py  |   yolov8s-pose_fp16_1b.bmodel   |      9.38       |      29.91      |      42.10      |      3.47       |
|   SE9-16    | yolov8_opencv.py  |   yolov8s-pose_int8_1b.bmodel   |      9.36       |      29.44      |      14.49      |      3.48       |
|   SE9-16    | yolov8_opencv.py  |   yolov8s-pose_int8_4b.bmodel   |      9.40       |      32.01      |      13.23      |      3.19       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s-pose_fp32_1b.bmodel   |      4.32       |      4.93       |     174.68      |      3.47       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s-pose_fp16_1b.bmodel   |      4.33       |      4.89       |      37.94      |      3.47       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s-pose_int8_1b.bmodel   |      4.28       |      4.87       |      10.41      |      3.49       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s-pose_int8_4b.bmodel   |      4.19       |      4.55       |      9.47       |      3.19       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s-pose_fp32_1b.bmodel   |      5.72       |      1.85       |     171.94      |      0.98       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s-pose_fp16_1b.bmodel   |      5.80       |      1.85       |      35.33      |      0.97       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_1b.bmodel   |      5.73       |      1.84       |      7.74       |      0.96       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_4b.bmodel   |      5.58       |      1.74       |      7.52       |      0.89       |
|   SE9-16    | yolov8_opencv.py  |yolov8s-pose_fp32_1b_2core.bmodel|      9.41       |      29.46      |      97.98      |      3.49       |
|   SE9-16    | yolov8_opencv.py  |yolov8s-pose_fp16_1b_2core.bmodel|      9.39       |      29.95      |      27.32      |      3.46       |
|   SE9-16    | yolov8_opencv.py  |yolov8s-pose_int8_1b_2core.bmodel|      9.37       |      29.81      |      13.65      |      3.49       |
|   SE9-16    | yolov8_opencv.py  |yolov8s-pose_int8_4b_2core.bmodel|      9.41       |      32.07      |      10.34      |      3.18       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s-pose_fp32_1b_2core.bmodel|      4.33       |      4.91       |      93.90      |      3.48       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s-pose_fp16_1b_2core.bmodel|      4.34       |      4.90       |      23.20      |      3.50       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s-pose_int8_1b_2core.bmodel|      4.31       |      4.88       |      9.49       |      3.50       |
|   SE9-16    |  yolov8_bmcv.py   |yolov8s-pose_int8_4b_2core.bmodel|      4.22       |      4.56       |      6.75       |      3.20       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-pose_fp32_1b_2core.bmodel|      5.81       |      1.83       |      91.25      |      0.97       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-pose_fp16_1b_2core.bmodel|      5.75       |      1.84       |      20.59      |      0.97       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-pose_int8_1b_2core.bmodel|      5.78       |      1.84       |      6.85       |      0.96       |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-pose_int8_4b_2core.bmodel|      5.57       |      1.74       |      4.79       |      0.89       |
|   SE5-16    | yolov8_opencv.py  |   yolov8s-pose_fp32_1b.bmodel   |      7.65       |      21.60      |      32.32      |      2.47       |
|   SE5-16    | yolov8_opencv.py  |   yolov8s-pose_int8_1b.bmodel   |      6.83       |      21.76      |      20.85      |      2.45       |
|   SE5-16    | yolov8_opencv.py  |   yolov8s-pose_int8_4b.bmodel   |      6.84       |      23.48      |      11.73      |      2.23       |
|   SE5-16    |  yolov8_bmcv.py   |   yolov8s-pose_fp32_1b.bmodel   |      3.60       |      2.83       |      29.43      |      2.47       |
|   SE5-16    |  yolov8_bmcv.py   |   yolov8s-pose_int8_1b.bmodel   |      3.60       |      2.82       |      17.87      |      2.41       |
|   SE5-16    |  yolov8_bmcv.py   |   yolov8s-pose_int8_4b.bmodel   |      3.45       |      2.64       |      9.09       |      2.21       |
|   SE5-16    |  yolov8_bmcv.soc  |   yolov8s-pose_fp32_1b.bmodel   |      4.85       |      1.56       |      27.60      |      0.69       |
|   SE5-16    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_1b.bmodel   |      4.97       |      1.57       |      16.07      |      0.69       |
|   SE5-16    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_4b.bmodel   |      4.74       |      1.49       |      7.72       |      0.63       |
|   SE7-32    | yolov8_opencv.py  |   yolov8s-pose_fp32_1b.bmodel   |      17.72      |      25.24      |      37.31      |      2.57       |
|   SE7-32    | yolov8_opencv.py  |   yolov8s-pose_fp16_1b.bmodel   |      14.80      |      24.68      |      11.87      |      2.57       |
|   SE7-32    | yolov8_opencv.py  |   yolov8s-pose_int8_1b.bmodel   |      14.21      |      24.92      |      9.28       |      2.54       |
|   SE7-32    | yolov8_opencv.py  |   yolov8s-pose_int8_4b.bmodel   |      14.39      |      27.58      |      8.22       |      2.35       |
|   SE7-32    |  yolov8_bmcv.py   |   yolov8s-pose_fp32_1b.bmodel   |      3.39       |      2.54       |      33.44      |      2.59       |
|   SE7-32    |  yolov8_bmcv.py   |   yolov8s-pose_fp16_1b.bmodel   |      3.28       |      2.46       |      7.97       |      2.55       |
|   SE7-32    |  yolov8_bmcv.py   |   yolov8s-pose_int8_1b.bmodel   |      3.44       |      2.62       |      5.59       |      2.60       |
|   SE7-32    |  yolov8_bmcv.py   |   yolov8s-pose_int8_4b.bmodel   |      3.40       |      2.61       |      4.74       |      2.43       |
|   SE7-32    |  yolov8_bmcv.soc  |   yolov8s-pose_fp32_1b.bmodel   |      5.73       |      1.02       |      31.14      |      0.89       |
|   SE7-32    |  yolov8_bmcv.soc  |   yolov8s-pose_fp16_1b.bmodel   |      5.10       |      0.80       |      5.62       |      0.83       |
|   SE7-32    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_1b.bmodel   |      5.04       |      0.80       |      3.09       |      0.80       |
|   SE7-32    |  yolov8_bmcv.soc  |   yolov8s-pose_int8_4b.bmodel   |      4.87       |      0.75       |      2.82       |      0.70       |
"""
# 定义一个字典来存储表格数据
table_data = {
    "platform": [],
    "program": [],
    "bmodel": [],
    "decode": [],
    "preprocess": [],
    "inference": [],
    "postprocess": []
}

# 处理基准测试字符串，提取并存储数据
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

# 定义正则表达式模式，用于从C++和Python日志中提取时间数据
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

# 根据给定的日志文本和模式提取时间数据
def extract_times(text, patterns):
    """
    从给定的文本中提取时间数据。

    参数:
    text (str): 包含时间数据的日志文本。
    patterns (dict): 匹配时间数据的正则表达式模式字典。

    返回:
    dict: 包含提取的时间数据的字典。
    """
    results = {}
    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            results[key] = round(float(match.group(1)),2)
    return results

# 定义命令行参数解析函数
def argsparser():
    """
    解析命令行参数。

    返回:
    argparse.Namespace: 包含解析后参数的对象。
    """
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--target', type=str, default='BM1684X', help='path of label json')
    parser.add_argument('--platform', type=str, default='soc', help='path of result json')
    parser.add_argument('--bmodel', type=str, default='yolov8_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov8_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov8_fp32_1b.bmodel_python_test.log')
    args = parser.parse_args()
    return args

# 主程序入口
if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    benchmark_path = current_dir + "/benchmark.txt"
    args = argsparser()
    # 根据平台和目标类型设置基准测试中的平台名称
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
    
    # 检查基准测试文件是否存在，如果不存在则创建
    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^13}|{:^19}|{:^25}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
               "platform", "program", "bmodel", "decode_time", "preprocess_time", "inference_time", "postprocess_time", width=min_width)
            f.write(benchmark_str)
            
    with open(args.input, "r") as f:
        data = f.read()
    # 根据语言类型选择合适的模式提取时间数据
    if args.language == "python":    
        extracted_data = extract_times(data, patterns_python)
    elif args.language == "cpp":
        extracted_data = extract_times(data, patterns_cpp)
    else:
        print("unsupport code language")
    match_index = -1
    # 在基准测试数据中查找匹配的条目
    for i in range(0, len(table_data["platform"])):
        if platform == table_data["platform"][i] and args.program == table_data["program"][i] and args.bmodel == table_data["bmodel"][i]:
            match_index = i
            break
    baseline_data = {}
    # 如果找到匹配条目，提取基准测试数据
    if match_index == -1:
        print("Unmatched case.")
    else:
        baseline_data["decode"] = table_data["decode"][match_index]
        baseline_data["preprocess"] = table_data["preprocess"][match_index]
        baseline_data["inference"] = table_data["inference"][match_index]
        baseline_data["postprocess"] = table_data["postprocess"][match_index]
    # 比较提取的数据和基准测试数据，检查差异
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
        
    # 根据当前数据生成新的基准测试条目
    benchmark_str = "|{:^13}|{:^19}|{:^33}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    # 如果存在不通过的比较，退出程序
    if compare_pass == False:
        sys.exit(1)
        
