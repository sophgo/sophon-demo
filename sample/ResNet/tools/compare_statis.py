import re
import argparse
import math
import os
import sys

baseline = """
|    测试平台  |     测试程序        |        测试模型       |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ------------------ | --------------------- | -------- | --------- | --------- | --------- |
| SE5-16      | resnet_opencv.py   | resnet50_fp32_1b.bmodel | 10.28    | 8.06      | 9.03      | 0.31      |
| SE5-16      | resnet_opencv.py   | resnet50_int8_1b.bmodel | 10.19    | 7.95      | 5.91      | 0.33      |
| SE5-16      | resnet_opencv.py   | resnet50_int8_4b.bmodel | 10.06    | 8.00      | 3.24      | 0.11      |
| SE5-16      | resnet_bmcv.py     | resnet50_fp32_1b.bmodel | 1.34     | 1.52      | 6.90      | 0.25      |
| SE5-16      | resnet_bmcv.py     | resnet50_int8_1b.bmodel | 1.35     | 1.52      | 4.05      | 0.24      |
| SE5-16      | resnet_bmcv.py     | resnet50_int8_4b.bmodel | 1.19     | 1.43      | 1.24      | 0.10      |
| SE5-16      | resnet_opencv.soc  | resnet50_fp32_1b.bmodel | 1.47     | 6.23      | 6.49      | 0.14      |
| SE5-16      | resnet_opencv.soc  | resnet50_int8_1b.bmodel | 1.47     | 6.27      | 3.64      | 0.15      |
| SE5-16      | resnet_opencv.soc  | resnet50_int8_4b.bmodel | 1.29     | 6.26      | 1.11      | 0.12      |
| SE5-16      | resnet_bmcv.soc    | resnet50_fp32_1b.bmodel | 3.90     | 2.45      | 6.49      | 0.11      |
| SE5-16      | resnet_bmcv.soc    | resnet50_int8_1b.bmodel | 2.91     | 2.45      | 3.63      | 0.13      |
| SE5-16      | resnet_bmcv.soc    | resnet50_int8_4b.bmodel | 2.85     | 2.41      | 1.11      | 0.11      |
| SE7-32      | resnet_opencv.py   | resnet50_fp32_1b.bmodel | 9.20     | 7.63      | 11.81     | 0.30      |
| SE7-32      | resnet_opencv.py   | resnet50_fp16_1b.bmodel | 9.18     | 7.62      | 4.35      | 0.30      |
| SE7-32      | resnet_opencv.py   | resnet50_int8_1b.bmodel | 9.18     | 7.60      | 3.78      | 0.30      |
| SE7-32      | resnet_opencv.py   | resnet50_int8_4b.bmodel | 9.13     | 7.64      | 3.08      | 0.11      |
| SE7-32      |  resnet_bmcv.py    | resnet50_fp32_1b.bmodel | 1.50     | 0.72      | 9.65      | 0.26      |
| SE7-32      |  resnet_bmcv.py    | resnet50_fp16_1b.bmodel | 1.51     | 0.72      | 2.14      | 0.26      |
| SE7-32      |  resnet_bmcv.py    | resnet50_int8_1b.bmodel | 1.51     | 0.73      | 1.62      | 0.26      |
| SE7-32      |  resnet_bmcv.py    | resnet50_int8_4b.bmodel | 1.28     | 0.62      | 0.95      | 0.10      |
| SE7-32      | resnet_opencv.soc  | resnet50_fp32_1b.bmodel | 1.17     | 5.68      | 9.12      | 0.09      |
| SE7-32      | resnet_opencv.soc  | resnet50_fp16_1b.bmodel | 1.16     | 5.66      | 1.61      | 0.09      |
| SE7-32      | resnet_opencv.soc  | resnet50_int8_1b.bmodel | 1.17     | 5.72      | 1.09      | 0.09      |
| SE7-32      | resnet_opencv.soc  | resnet50_int8_4b.bmodel | 0.99     | 5.72      | 0.81      | 0.07      |
| SE7-32      |  resnet_bmcv.soc   | resnet50_fp32_1b.bmodel | 2.18     | 0.45      | 9.12      | 0.11      |
| SE7-32      |  resnet_bmcv.soc   | resnet50_fp16_1b.bmodel | 2.18     | 0.45      | 1.61      | 0.11      |
| SE7-32      |  resnet_bmcv.soc   | resnet50_int8_1b.bmodel | 2.18     | 0.45      | 1.09      | 0.11      |
| SE7-32      |  resnet_bmcv.soc   | resnet50_int8_4b.bmodel | 2.14     | 0.41      | 0.81      | 0.10      |
| SE9-16      | resnet_opencv.py   | resnet50_fp32_1b.bmodel | 12.96    | 10.61     | 49.14     | 0.13      |
| SE9-16      | resnet_opencv.py   | resnet50_fp16_1b.bmodel | 12.82    | 10.64     | 11.11     | 0.42      |
| SE9-16      | resnet_opencv.py   | resnet50_int8_1b.bmodel | 12.83    | 10.71     | 6.26      | 0.42      |
| SE9-16      | resnet_opencv.py   | resnet50_int8_4b.bmodel | 12.80    | 10.67     | 4.87      | 0.15      |
| SE9-16      | resnet_bmcv.py     | resnet50_fp32_1b.bmodel | 3.23     | 1.83      | 46.43     | 0.38      |
| SE9-16      | resnet_bmcv.py     | resnet50_fp16_1b.bmodel | 3.22     | 1.83      | 8.37      | 0.38      |
| SE9-16      | resnet_bmcv.py     | resnet50_int8_1b.bmodel | 3.22     | 1.83      | 3.56      | 0.38      |
| SE9-16      | resnet_bmcv.py     | resnet50_int8_4b.bmodel | 2.90     | 1.62      | 2.31      | 0.14      |
| SE9-16      | resnet_opencv.soc  | resnet50_fp32_1b.bmodel | 2.45     | 61.32     | 45.59     | 0.12      |
| SE9-16      | resnet_opencv.soc  | resnet50_fp16_1b.bmodel | 2.47     | 61.38     | 7.58      | 0.12      |
| SE9-16      | resnet_opencv.soc  | resnet50_int8_1b.bmodel | 2.42     | 61.41     | 2.77      | 0.12      |
| SE9-16      | resnet_opencv.soc  | resnet50_int8_4b.bmodel | 2.05     | 61.42     | 2.08      | 0.09      |
| SE9-16      | resnet_bmcv.soc    | resnet50_fp32_1b.bmodel | 3.95     | 1.42      | 45.56     | 0.17      |
| SE9-16      | resnet_bmcv.soc    | resnet50_fp16_1b.bmodel | 3.88     | 1.42      | 7.54      | 0.17      |
| SE9-16      | resnet_bmcv.soc    | resnet50_int8_1b.bmodel | 3.92     | 1.42      | 2.72      | 0.17      |
| SE9-16      | resnet_bmcv.soc    | resnet50_int8_4b.bmodel | 3.68     | 5.27      | 2.08      | 0.12      |
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
    parser.add_argument('--bmodel', type=str, default='resnet_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='resnet_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_resnet_fp32_1b.bmodel_python_test.log')
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
        
