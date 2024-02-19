import re
import argparse
import math
import os
import sys

baseline = """
|    测试平台  |     测试程序      |             测试模型                | decode_time | preprocess_time | inference_time | postprocess_time |
| ----------- | ---------------- | ----------------------------------- | ----------- | --------------- | -------------- | ---------------- |
| BM1684 SoC  | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel  | 17.8        | 27.8            | 93.9           | 144.0            |
| BM1684 SoC  | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel  | 16.3        | 23.8            | 70.3           | 143.2            |
| BM1684 SoC  | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel  | 14.6        | 25.1            | 41.2           | 148.2            |
| BM1684 SoC  | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel  | 3.0         | 3.0             | 88.9           | 139.9            |
| BM1684 SoC  | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel  | 3.0         | 2.6             | 54.9           | 147.2            |
| BM1684 SoC  | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel  | 2.8         | 2.4             | 24.5           | 150.8            |
| BM1684 SoC  | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel  | 14.2        | 1.9             | 82.9           | 20.5             |
| BM1684 SoC  | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel  | 14.7        | 1.7             | 48.7           | 21.9             |
| BM1684 SoC  | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel  | 14.5        | 1.6             | 19.3           | 21.5             |
| BM1684X SoC | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel  | 15.0        | 26.6            | 142.5          | 135.9            |
| BM1684X SoC | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel  | 3.2         | 24.7            | 39.5           | 133.4            |
| BM1684X SoC | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel  | 15.0        | 22.3            | 22.3           | 132.1            |
| BM1684X SoC | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel  | 18.5        | 22.4            | 18.7           | 107.7            |
| BM1684X SoC | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel  | 2.6         | 2.3             | 114.9          | 133.7            |
| BM1684X SoC | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b.bmodel  | 2.6         | 2.2             | 35.9           | 132.4            |
| BM1684X SoC | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel  | 2.6         | 2.2             | 19.0           | 133.3            |
| BM1684X SoC | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel  | 2.4         | 2.1             | 17.9           | 149.5            |
| BM1684X SoC | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel  | 13.9        | 0.8             | 131.1          | 20.7             |
| BM1684X SoC | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b.bmodel  | 13.9        | 0.8             | 29.4           | 20.7             |
| BM1684X SoC | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel  | 13.9        | 0.8             | 12.6           | 20.8             |
| BM1684X SoC | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel  | 12.3        | 0.7             | 9.5            | 18.7             |
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
    parser.add_argument('--bmodel', type=str, default='yolov7_v0.1_3output_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov7_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov7_v0.1_3output_fp32_1b.bmodel_python_test.log')
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
        
