import re
import argparse
import math
import os
import sys

baseline = """
|    测试平台  |     测试程序      |             测试模型                |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
| SE5-16      | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 14.0     | 27.8          | 33.5          | 115       |
| SE5-16      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 13.9     | 23.5          | 33.5          | 111       |
| SE5-16      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 13.8     | 24.2          | 28.2          | 115       |
| SE5-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 3.0      | 3.0           | 28.5          | 111       |
| SE5-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 3.0      | 2.4           | 17.4          | 111       |
| SE5-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 2.8      | 2.3           | 11.5          | 115       |
| SE5-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 5.4      | 1.5           | 22.6          | 15.7      |
| SE5-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 5.4      | 1.5           | 11.5          | 15.7      |
| SE5-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 5.2      | 1.6           | 6.2           | 15.5      |
| SE5-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 3.3      | 3.1           | 23.3          | 13.9      |
| SE5-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 3.3      | 1.9           | 12.2          | 13.9      |
| SE5-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 3.1      | 1.8           | 6.9           | 13.8      |
| SE7-32      | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel |  15.26   | 27.31         | 33.20         |  108.40   |
| SE7-32      | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel |  13.95   | 26.98         | 18.88         |  108.68   |
| SE7-32      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel |  13.93   | 27.52         | 15.07         |  108.64   |
| SE7-32      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel |  13.81   | 25.16         | 13.23         |  107.75   |
| SE7-32      | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel |  3.08    | 2.34          | 29.12         |  103.93   |
| SE7-32      | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel |  3.07    | 2.32          | 14.78         |  103.74   |
| SE7-32      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel |  3.05    | 2.33          | 10.87         |  103.78   |
| SE7-32      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel |  2.92    | 2.16          | 9.78          |  108.15   |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel |  4.27    | 0.73          | 21.62         |   15.81   |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel |  4.29    | 0.73          | 7.37          |   15.89   |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel |  4.30    | 0.73          | 3.47          |   15.90   |
| SE7-32      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel |  4.14    | 0.71          | 3.33          |   15.73   |
| SE7-32      | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel |  2.69    | 2.58          | 23.40         |   14.15   |
| SE7-32      | yolov5_sail.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel |  2.69    | 2.59          | 9.11          |   14.17   |
| SE7-32      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel |  2.70    | 2.58          | 5.22          |   14.16   |
| SE7-32      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_4b.bmodel |  2.56    | 2.51          | 4.97          |   14.07   |
| SE9-16      | yolov5_opencv.py | yolov5s_v6.1_3output_fp32_1b.bmodel | 21.3     | 36.8          | 113.4         | 151.9     |
| SE9-16      | yolov5_opencv.py | yolov5s_v6.1_3output_fp16_1b.bmodel | 22.4     | 36.3          | 42.1          | 152.1     |
| SE9-16      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_1b.bmodel | 19.5     | 35.6          | 21.8          | 150.6     |
| SE9-16      | yolov5_opencv.py | yolov5s_v6.1_3output_int8_4b.bmodel | 20.1     | 33.2          | 19.5          | 151.6     |
| SE9-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp32_1b.bmodel | 4.8      | 5.2           | 106.5         | 143.97    |
| SE9-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_fp16_1b.bmodel | 4.5      | 5.2           | 36.2          | 143.9     |
| SE9-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_1b.bmodel | 4.5      | 5.2           | 15.6          | 144.2     |
| SE9-16      | yolov5_bmcv.py   | yolov5s_v6.1_3output_int8_4b.bmodel | 4.4      | 4.9           | 14.1          | 149.9     |
| SE9-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 5.8      | 2.0           | 98.1          | 22.4      |
| SE9-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 5.8      | 2.0           | 27.5          | 22.4      |
| SE9-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 5.8      | 2.0           | 7.3           | 22.4      |
| SE9-16      | yolov5_bmcv.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 5.7      | 1.9           | 7.0           | 22.4      |
| SE9-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_fp32_1b.bmodel | 4.3      | 5.3           | 99.6          | 19.9      |
| SE9-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_fp16_1b.bmodel | 4.3      | 5.3           | 28.9          | 19.9      |
| SE9-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_1b.bmodel | 4.3      | 5.3           | 8.7           | 19.9      |
| SE9-16      | yolov5_sail.soc  | yolov5s_v6.1_3output_int8_4b.bmodel | 4.3      | 5.3           | 8.2           | 19.9      |
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
        
