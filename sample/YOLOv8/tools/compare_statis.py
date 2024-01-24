import re
import argparse
import math
import os
import sys

baseline = """
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | --------  | ---------    | ---------     | ---------      |
| BM1684 SoC  | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 15.90     | 23.54        | 31.30         | 5.00           |
| BM1684 SoC  | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 15.09     | 23.06        | 22.10         | 4.97           | 
| BM1684 SoC  | yolov8_opencv.py | yolov8s_int8_4b.bmodel | 15.18     | 25.36        | 25.39         | 5.09           |
| BM1684 SoC  | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 3.55      | 2.81         | 29.13         | 4.81           |
| BM1684 SoC  | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 3.55      | 2.79         | 18.98         | 4.90           |
| BM1684 SoC  | yolov8_bmcv.py   | yolov8s_int8_4b.bmodel | 3.43      | 2.61         | 9.78          | 4.44           |
| BM1684 SoC  | yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel | 4.928     | 1.560        | 27.00         | 17.59          |
| BM1684 SoC  | yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel | 4.930     | 1.560        | 15.10         | 17.58          |
| BM1684 SoC  | yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel | 4.809     | 1.491        | 8.089         | 17.34          |
| BM1684X SoC | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 15.03     | 22.98        | 34.80         | 5.45           |
| BM1684X SoC | yolov8_opencv.py | yolov8s_fp16_1b.bmodel | 15.03     | 22.46        | 12.14         | 5.45           |
| BM1684X SoC | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 14.99     | 22.40        | 9.18          | 5.37           |
| BM1684X SoC | yolov8_opencv.py | yolov8s_int8_4b.bmodel | 15.03     | 24.77        | 8.91          | 5.47           |
| BM1684X SoC | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 3.0       | 2.2          | 31.0          | 5.4            |
| BM1684X SoC | yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel | 3.0       | 2.2          | 8.5           | 5.4            |
| BM1684X SoC | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 3.0       | 2.2          | 5.5           | 5.4            |
| BM1684X SoC | yolov8_bmcv.py   | yolov8s_int8_4b.bmodel | 2.9       | 2.1          | 5.1           | 4.8            |
| BM1684X SoC | yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel | 4.324     | 0.772        | 28.97         | 17.96          |
| BM1684X SoC | yolov8_bmcv.soc  | yolov8s_fp16_1b.bmodel | 4.312     | 0.772        | 6.259         | 17.80          |
| BM1684X SoC | yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel | 4.276     | 0.772        | 3.350         | 17.95          |
| BM1684X SoC | yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel | 4.128     | 0.736        | 3.277         | 17.70          |
| BM1688 SoC  | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 23.40     | 30.15        | 170.44        | 6.99           |
| BM1688 SoC  | yolov8_opencv.py | yolov8s_fp16_1b.bmodel | 22.15     | 30.18        |  42.50        | 6.97           |
| BM1688 SoC  | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 19.33     | 30.21        |  16.50        | 6.97           |
| BM1688 SoC  | yolov8_opencv.py | yolov8s_int8_4b.bmodel | 19.35     | 32.95        |  15.82        | 7.33           |
| BM1688 SoC  |  yolov8_bmcv.py  | yolov8s_fp32_1b.bmodel | 4.27      | 4.88         | 165.86        | 6.97           |
| BM1688 SoC  |  yolov8_bmcv.py  | yolov8s_fp16_1b.bmodel | 4.21      | 4.86         |  38.09        | 7.03           |
| BM1688 SoC  |  yolov8_bmcv.py  | yolov8s_int8_1b.bmodel | 4.21      | 4.88         |  12.10        | 7.04           |
| BM1688 SoC  |  yolov8_bmcv.py  | yolov8s_int8_4b.bmodel | 4.02      | 4.56         |  11.06        | 6.21           |
| BM1688 SoC  |  yolov8_bmcv.soc | yolov8s_fp32_1b.bmodel | 5.68      | 1.91         | 162.71        | 25.07          |
| BM1688 SoC  |  yolov8_bmcv.soc | yolov8s_fp16_1b.bmodel | 5.61      | 1.91         |  34.98        | 25.10          |
| BM1688 SoC  |  yolov8_bmcv.soc | yolov8s_int8_1b.bmodel | 5.66      | 1.91         |  9.03         | 25.05          |
| BM1688 SoC  |  yolov8_bmcv.soc | yolov8s_int8_4b.bmodel | 5.48      | 1.81         |  8.68         | 24.95          |
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
        
