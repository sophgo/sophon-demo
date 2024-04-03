import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
|    测试平台  |     测试程序      |             测试模型                | decode_time | preprocess_time | inference_time | postprocess_time |
| ----------- | ---------------- | ----------------------------------- | ----------- | --------------- | -------------- | ---------------- |
| SE5-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 20.10       | 26.29           | 93.05          | 111.12           |
| SE5-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 13.99       | 26.35           | 71.77          | 109.78           |
| SE5-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 13.85       | 23.91           | 40.65          | 112.05           |
| SE5-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 3.57        | 2.82            | 89.29          | 106.19           |
| SE5-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 3.57        | 2.30            | 54.98          | 105.83           |
| SE5-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 3.45        | 2.12            | 24.84          | 109.49           |
| SE5-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 4.87        | 1.55            | 82.58          | 18.60            |
| SE5-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 4.86        | 1.54            | 48.19          | 18.57            |
| SE5-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 4.72        | 1.47            | 19.05          | 18.48            |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 18.38       | 27.02           | 111.60         | 109.77           |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel       | 14.11       | 27.70           | 35.96          | 109.32           |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 14.01       | 27.78           | 20.83          | 109.39           |
| SE7-32   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 13.96       | 25.33           | 18.73          | 112.40           |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 3.04        | 2.34            | 106.45         | 104.30           |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b.bmodel       | 3.02        | 2.35            | 30.94          | 103.90           |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 3.03        | 2.34            | 15.72          | 104.00           |
| SE7-32   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 2.90        | 2.17            | 14.37          | 108.36           |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 4.31        | 0.74            | 99.85          | 18.65            |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b.bmodel       | 4.29        | 0.74            | 24.34          | 18.65            |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 4.31        | 0.74            | 9.11           | 18.66            |
| SE7-32   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 4.16        | 0.71            | 8.70           | 18.54            |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 22.21       | 37.03           | 581.77         | 150.65           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel       | 19.42       | 36.64           | 128.73         | 150.99           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 19.46       | 36.37           | 44.55          | 150.91           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 19.26       | 33.41           | 41.89          | 150.92           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 4.44        | 5.04            | 577.06         | 143.57           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b.bmodel       | 4.37        | 5.06            | 123.28         | 143.58           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 4.36        | 5.00            | 38.96          | 143.61           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 4.27        | 4.75            | 37.47          | 150.91           |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 5.92        | 1.82            | 567.47         | 26.02            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b.bmodel       | 5.88        | 1.83            | 113.91         | 25.95            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 5.84        | 1.82            | 29.66          | 25.92            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 5.68        | 1.74            | 29.50          | 25.83            |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b_2core.bmodel | 19.49       | 36.79           | 318.75         | 151.24           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b_2core.bmodel | 19.28       | 36.04           | 87.75          | 151.02           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b_2core.bmodel | 19.46       | 36.73           | 31.93          | 151.16           |
| SE9-16   | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b_2core.bmodel | 19.28       | 33.22           | 26.07          | 150.90           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b_2core.bmodel | 4.43        | 5.02            | 313.23         | 143.94           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b_2core.bmodel | 4.37        | 5.03            | 82.11          | 143.78           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b_2core.bmodel | 4.37        | 5.04            | 26.54          | 143.44           |
| SE9-16   | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b_2core.bmodel | 4.40        | 4.70            | 21.64          | 150.61           |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b_2core.bmodel | 6.28        | 1.82            | 303.79         | 25.97            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b_2core.bmodel | 6.94        | 1.82            | 72.65          | 25.99            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b_2core.bmodel | 5.85        | 1.83            | 17.07          | 26.01            |
| SE9-16   | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b_2core.bmodel | 5.67        | 1.74            | 13.64          | 25.79            |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_fp32_1b.bmodel       | 33.68       | 35.82           | 592.63         | 153.60           |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_fp16_1b.bmodel       | 25.47       | 36.44           | 137.69         | 150.43           |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_int8_1b.bmodel       | 21.03       | 36.40           | 47.95          | 150.02           |
| SE9-8    | yolov7_opencv.py | yolov7_v0.1_3output_int8_4b.bmodel       | 20.88       | 32.59           | 45.06          | 149.44           |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_fp32_1b.bmodel       | 4.22        | 4.86            | 587.39         | 143.62           |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_fp16_1b.bmodel       | 4.21        | 4.90            | 132.43         | 143.64           |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_1b.bmodel       | 5.82        | 4.87            | 42.63          | 143.51           |
| SE9-8    | yolov7_bmcv.py   | yolov7_v0.1_3output_int8_4b.bmodel       | 4.09        | 4.57            | 41.92          | 149.68           |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp32_1b.bmodel       | 5.88        | 1.81            | 577.88         | 26.00            |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_fp16_1b.bmodel       | 6.74        | 1.81            | 123.12         | 25.96            |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_1b.bmodel       | 5.63        | 1.81            | 33.37          | 25.96            |
| SE9-8    | yolov7_bmcv.soc  | yolov7_v0.1_3output_int8_4b.bmodel       | 5.47        | 1.72            | 32.98          | 25.80            |
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
        
