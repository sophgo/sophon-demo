import re
import argparse
import math
import os
import sys

baseline = """
| 测试平台     | 测试程序              | 测试模型                            | track_time |
| ------------ | --------------------- | ----------------------------------- | ---------- |
|   SE5-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.77       |
|   SE5-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      5.33       |
|   SE5-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      5.49       |
|   SE5-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.60       |
|   SE5-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      0.53       |
|   SE5-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      0.51       |
|   SE5-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.35       |
|   SE5-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_1b.bmodel|      0.31       |
|   SE5-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_4b.bmodel|      0.30       |
|   SE7-32    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.81       |
|   SE7-32    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      5.81       |
|   SE7-32    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      5.79       |
|   SE7-32    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      5.99       |
|   SE7-32    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.59       |
|   SE7-32    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      0.59       |
|   SE7-32    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      0.61       |
|   SE7-32    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      0.58       |
|   SE7-32    |    bytetrack_eigen.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.35       |
|   SE7-32    |    bytetrack_eigen.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      0.35       |
|   SE7-32    |    bytetrack_eigen.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      0.36       |
|   SE7-32    |    bytetrack_eigen.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      0.34       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      8.21       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      8.18       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      7.94       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      8.24       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      2.04       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      2.03       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      1.93       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      1.92       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp32_1b.bmodel|      0.49       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp16_1b.bmodel|      0.49       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_1b.bmodel|      0.47       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_4b.bmodel|      0.46       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      8.24       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      8.23       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      7.92       |
|   SE9-16    |   bytetrack_opencv.py   |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      8.18       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      2.04       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      2.03       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      1.93       |
|   SE9-16    |   bytetrack_opencv.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      1.92       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      0.49       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      0.49       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      0.47       |
|   SE9-16    |   bytetrack_eigen.soc   |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      0.46       |
"""
table_data = {
    "platform": [],
    "program": [],
    "bmodel": [],
    "track": [],
}

for line in baseline.strip().split("\n")[2:]:
    match = re.search(r'\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', line)
    if match:
        table_data["platform"].append(match.group(1))
        table_data["program"].append(match.group(2))
        table_data["bmodel"].append(match.group(3))
        table_data["track"].append(float(match.group(4)))

patterns_cpp = {
    'track': re.compile(r'\[.*bytetrack time.*\]  loops:.*avg: ([\d.]+) ms'),
}

patterns_python = {
    'track': re.compile(r'bytetrack_track_time\(ms\): ([\d.]+)'),
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
    parser.add_argument('--program', type=str, default='bytetrack_opencv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov5s_v6.1_3output_fp32_1b.bmodel_python_test.log')
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
    else:
        platform = args.target + " SoC" if args.platform == "soc" else args.target + " PCIe"
    min_width = 17
    
    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^13}|{:^25}|{:^35}|{:^{width}}|\n".format(
           "platform", "program", "bmodel", "track_time", width=min_width)
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
        baseline_data["track"] = table_data["track"][match_index]
    for key, statis in baseline_data.items():
        if abs(statis - extracted_data[key]) / statis > 0.2:
            print("{:} time, diff ratio > 0.2".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^25}|{:^35}|{track:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
