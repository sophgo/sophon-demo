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
|    测试平台  |     测试程序  |      测试模型     |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------- | ---------------- | -------- | ---------   | ---------------| --------- |
| SE5-16      | openpose_opencv.py | pose_coco_fp32_1b.bmodel |  13.86  |  8.03  | 130.78  | 3068.47  |
| SE5-16      | openpose_opencv.py | pose_coco_int8_1b.bmodel |  13.95  |  8.20  | 74.49   | 3068.18  |
| SE5-16      | openpose_opencv.py | pose_coco_int8_4b.bmodel |  14.07  |  8.81  | 26.83   | 3052.46  |
| SE5-16      | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  5.27   |  1.24  | 125.56  | 302.45   |
| SE5-16      | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  5.24   |  1.25  | 62.99   | 301.48   |
| SE5-16      | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  5.18   |  1.29  | 15.77   | 306.28   |
| SE7-32      | openpose_opencv.py | pose_coco_fp32_1b.bmodel |  15.02  |  7.26  | 257.63  | 3111.41  |
| SE7-32      | openpose_opencv.py | pose_coco_fp16_1b.bmodel |  15.00  |  7.30  | 24.60   | 3111.20  |
| SE7-32      | openpose_opencv.py | pose_coco_int8_1b.bmodel |  15.02  |  7.33  | 14.96   | 3111.70  |
| SE7-32      | openpose_opencv.py | pose_coco_int8_4b.bmodel |  14.99  |  7.42  | 14.22   | 3111.17  |
| SE7-32      | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  4.81   |  0.45  | 252.15  | 295.07   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |  4.76   |  0.45  | 19.02   | 300.03   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  4.76   |  0.45  | 9.37    | 293.81   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  4.67   |  0.43  | 9.25    | 296.2    |
| SE9-16      | openpose_opencv.py | pose_coco_fp32_1b.bmodel |  22.44  |  9.76  | 1318.85 | 4145.92  |
| SE9-16      | openpose_opencv.py | pose_coco_fp16_1b.bmodel |  22.55  |  9.85  | 162.88  | 4142.10  |
| SE9-16      | openpose_opencv.py | pose_coco_int8_1b.bmodel |  20.74  |  9.71  | 47.33   | 4133.52  |
| SE9-16      | openpose_opencv.py | pose_coco_int8_4b.bmodel |  19.75  |  10.20 | 45.93   | 4139.08  |
| SE9-16      | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  5.95   |  1.30  | 1311.69 | 684.23   |
| SE9-16      | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |  7.97   |  1.31  | 155.83  | 696.08   |
| SE9-16      | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  5.97   |  1.30  | 40.30   | 678.41   |
| SE9-16      | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  5.77   |  1.20  | 39.81   | 682.26   |
| SE9-16      | openpose_opencv.py | pose_coco_fp32_1b_2core.bmodel |  19.35  |  9.73  | 1260.21 | 4140.66  |
| SE9-16      | openpose_opencv.py | pose_coco_fp16_1b_2core.bmodel |  19.31  |  9.74  | 132.36  | 4139.41  |
| SE9-16      | openpose_opencv.py | pose_coco_int8_1b_2core.bmodel |  19.28  |  9.73  | 45.06   | 4131.16  |
| SE9-16      | openpose_opencv.py | pose_coco_int8_4b_2core.bmodel |  19.27  |  9.74  | 26.62   | 4132.07  |
| SE9-16      | openpose_bmcv.soc  | pose_coco_fp32_1b_2core.bmodel |  6.00   |  1.30  | 1253.08 | 688.29   |
| SE9-16      | openpose_bmcv.soc  | pose_coco_fp16_1b_2core.bmodel |  5.94   |  1.31  | 125.33  | 686.40   |
| SE9-16      | openpose_bmcv.soc  | pose_coco_int8_1b_2core.bmodel |  5.96   |  1.31  | 38.05   | 675.93   |
| SE9-16      | openpose_bmcv.soc  | pose_coco_int8_4b_2core.bmodel |  5.77   |  1.20  | 20.80   | 678.30   |
"""

baseline_tpu_kernel_opt = """
|    测试平台  |     测试程序  |      测试模型     |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------- | ---------------- | -------- | ---------   | ---------------| --------- |
| SE7-32      | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  4.56   |  0.46  | 252.02   | 51.58   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |  4.62   |  0.47  | 19.02    | 50.24   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  4.60   |  0.47  | 9.37     | 50.43   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  4.47   |  0.42  | 9.28     | 50.38    |
"""

baseline_tpu_kernel_half_img_size_opt = """
|    测试平台  |     测试程序  |      测试模型     |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------- | ---------------- | -------- | ---------   | ---------------| --------- |
| SE7-32      | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  4.54   |  0.46  | 252.01  | 10.65   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |  4.56   |  0.46  | 19.03   | 10.66   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  4.60   |  0.46  | 9.37    | 10.43   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  4.50   |  0.42  | 9.27    | 10.69   |
"""

baseline_cpu_opt = """
|    测试平台  |     测试程序  |      测试模型     |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------- | ---------------- | -------- | ---------   | ---------------| --------- |
| SE5-16      | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  5.10   |  1.30  | 125.39  | 37.80   |
| SE5-16      | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  5.17   |  1.30  | 62.93   | 39.16   |
| SE5-16      | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  5.04   |  1.22  | 15.74   | 39.02   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  4.52   |  0.45  | 251.99  | 36.65   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |  4.57   |  0.46  | 19.01   | 37.18   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  4.52   |  0.46  | 9.37    | 36.39   |
| SE7-32      | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  4.40   |  0.41  | 9.27    | 35.87   |
| SE9-16      | openpose_bmcv.soc  | pose_coco_fp32_1b.bmodel |  6.08   |  1.31  | 1311.69 | 341.78  |
| SE9-16      | openpose_bmcv.soc  | pose_coco_fp16_1b.bmodel |  6.01   |  1.32  | 155.83  | 342.54  |
| SE9-16      | openpose_bmcv.soc  | pose_coco_int8_1b.bmodel |  6.13   |  1.29  | 40.30   | 340.54  |
| SE9-16      | openpose_bmcv.soc  | pose_coco_int8_4b.bmodel |  5.99   |  1.20  | 39.81   | 340.52  |
| SE9-16      | openpose_bmcv.soc  | pose_coco_fp32_1b_2core.bmodel |  7.07   |  1.31  | 1253.08  | 342.05   |
| SE9-16      | openpose_bmcv.soc  | pose_coco_fp16_1b_2core.bmodel |  6.10   |  1.31  | 125.33   | 342.49   |
| SE9-16      | openpose_bmcv.soc  | pose_coco_int8_1b_2core.bmodel |  6.03   |  1.31  | 38.06    | 340.73   |
| SE9-16      | openpose_bmcv.soc  | pose_coco_int8_4b_2core.bmodel |  5.83   |  1.20  | 20.80    | 340.56   |
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
    parser.add_argument('--bmodel', type=str, default='pose_coco_int8_1b.bmodel')
    parser.add_argument('--program', type=str, default='openpose_opencv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/opencv_pose_coco_int8_1b.bmodel_python_test.log')
    parser.add_argument('--performance_opt', type=str, default='no_opt', choices=['tpu_kernel_opt', 'tpu_kernel_half_img_size_opt', 'cpu_opt', 'no_opt'], 
                            help="opt type, selected from ['tpu_kernel_opt', 'tpu_kernel_half_img_size_opt', 'cpu_opt', 'no_opt']")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    args = argsparser()

    if args.performance_opt != 'no_opt':
        benchmark_path = current_dir + "/benchmark_" + args.performance_opt + ".txt"
        baseline = eval("baseline_" + args.performance_opt)
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
    else:
        platform = args.target + " SoC" if args.platform == "soc" else args.target + " PCIe"
    min_width = 17
    
    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^13}|{:^19}|{:^25}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
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
        