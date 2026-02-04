import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
| 测试平台 | 测试程序          | 测试模型                | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | ----------------- | ----------------------- | ----------- | --------------- | -------------- | ---------------- |
| SE7-32   | yolo26_opencv.py | yolo26s_fp32_1b.bmodel | 6.79       | 22.91           | 29.77          | 0.60             |
| SE7-32   | yolo26_opencv.py | yolo26s_fp16_1b.bmodel | 6.80       | 22.94           | 10.22          | 0.60             |
| SE7-32   | yolo26_opencv.py | yolo26s_int8_1b.bmodel | 6.80       | 22.93           | 7.97           | 0.61             |
| SE7-32   | yolo26_bmcv.py   | yolo26s_fp32_1b.bmodel | 2.91        | 1.99            | 26.12          | 0.58             |
| SE7-32   | yolo26_bmcv.py   | yolo26s_fp16_1b.bmodel | 2.90        | 1.98            | 6.88           | 0.57             |
| SE7-32   | yolo26_bmcv.py   | yolo26s_int8_1b.bmodel | 2.92        | 1.99            | 4.46           | 0.57             |
| SE7-32   | yolo26_bmcv.soc  | yolo26s_fp32_1b.bmodel | 4.44        | 0.76            | 25.85          | 0.12            |
| SE7-32   | yolo26_bmcv.soc  | yolo26s_fp16_1b.bmodel | 4.44        | 0.76            | 6.59           | 0.12            |
| SE7-32   | yolo26_bmcv.soc  | yolo26s_int8_1b.bmodel | 4.46        | 0.76            | 4.18           | 0.12            |
| SE9-16   | yolo26_opencv.py | yolo26s_fp32_1b.bmodel | 9.36        | 35.17           | 135.31         | 0.63             |
| SE9-16   | yolo26_opencv.py | yolo26s_fp16_1b.bmodel | 9.27        | 35.35           | 39.86          | 0.63             |
| SE9-16   | yolo26_opencv.py | yolo26s_int8_1b.bmodel | 9.17        | 34.49           | 17.83          | 0.63             |
| SE9-16   | yolo26_opencv.py | yolo26s_int8_1b_2core.bmodel | 9.20        | 34.67           | 15.03          | 0.63             |
| SE9-16   | yolo26_bmcv.py   | yolo26s_fp32_1b.bmodel | 3.87        | 3.99            | 130.84         | 0.61             |
| SE9-16   | yolo26_bmcv.py   | yolo26s_fp16_1b.bmodel | 3.86        | 3.98            | 35.54          | 0.61             |
| SE9-16   | yolo26_bmcv.py   | yolo26s_int8_1b.bmodel | 3.85        | 3.99            | 13.54          | 0.61             |
| SE9-16   | yolo26_bmcv.py   | yolo26s_int8_1b_2core.bmodel | 3.87        | 3.99            | 10.72          | 0.61             |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_fp32_1b.bmodel | 5.89        | 1.74            | 130.34         | 0.17             |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_fp16_1b.bmodel | 5.87        | 1.74            | 35.09          | 0.17             |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_int8_1b.bmodel | 5.90        | 1.76            | 13.12          | 0.18             |
| SE9-16   | yolo26_bmcv.soc  | yolo26s_int8_1b_2core.bmodel | 5.88        | 1.74            | 10.28          | 0.17             |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       |      9.54       |      34.69      |     134.96      |      0.63       |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       |      9.55       |      34.77      |      39.72      |      0.62       |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_int8_1b.bmodel       |      9.53       |      34.73      |      17.74      |      0.63       |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       |      3.90       |      3.99       |     130.73      |      0.58       |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       |      3.83       |      3.98       |      35.47      |      0.57       |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_int8_1b.bmodel       |      3.84       |      3.99       |      13.47      |      0.57       |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       |      5.64       |      1.74       |     130.23      |      0.09       |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       |      5.78       |      1.74       |      35.02      |      0.09       |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_int8_1b.bmodel       |      5.71       |      1.74       |      13.04      |      0.09       |
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
    parser.add_argument('--bmodel', type=str, default='yolo26s_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolo26_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolo26s_fp32_1b.bmodel_python_test.log')
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
        threhold = 0.2
        if key == "decode":
            threhold = 0.5
        if key == "postprocess":
            threhold = 0.4
        if statis < extracted_data[key] and abs(statis - extracted_data[key]) / statis > threhold:
            print("{:} time, diff ratio > {:}".format(key, str(threhold)))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^35}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
