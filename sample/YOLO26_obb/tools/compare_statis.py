import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
| 测试平台 | 测试程序          | 测试模型                | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | ----------------- | ----------------------- | ----------- | --------------- | -------------- | ---------------- |
|   SE7-32    | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       |      95.03      |      65.83      |      77.48      |      0.71                          |
|   SE7-32    | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       |      93.17      |      66.96      |      25.18      |      0.71                          |
|   SE7-32    |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       |      21.85      |      10.95      |      69.45      |      0.69                          |
|   SE7-32    |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       |      21.91      |      10.98      |      16.97      |      0.69                          |
|   SE7-32    |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       |      21.30      |      10.20      |      69.11      |      0.15                          |
|   SE7-32    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       |      21.31      |      10.21      |      16.65      |      0.15                          |
|   SE9-16    | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       |     140.91      |      85.57      |     364.98      |      0.90       |
|   SE9-16    | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       |     139.66      |      84.53      |      99.98      |      0.90       |
|   SE9-16    | yolo26_opencv.py  |   yolo26s_fp16_1b_2core.bmodel    |     130.23      |      84.15      |      61.99      |      0.90       |
|   SE9-16    |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       |      36.88      |      30.82      |     354.51      |      0.88       |
|   SE9-16    |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       |      39.12      |      30.84      |      89.77      |      0.87       |
|   SE9-16    |  yolo26_bmcv.py   |   yolo26s_fp16_1b_2core.bmodel    |      31.32      |      30.83      |      51.68      |      0.87       |
|   SE9-16    |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       |      31.72      |      28.39      |     353.86      |      0.22       |
|   SE9-16    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       |      30.38      |      28.38      |      89.33      |      0.22       |
|   SE9-16    |  yolo26_bmcv.soc  |   yolo26s_fp16_1b_2core.bmodel    |      31.46      |      28.38      |      51.24      |      0.22       |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       |     154.26      |      89.15      |     362.32      |      0.90       |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       |     160.87      |      87.05      |      99.77      |      0.89       |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       |      40.90      |      30.87      |     352.19      |      0.87       |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       |      63.90      |      30.82      |      89.54      |      0.95       |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       |      32.19      |      28.40      |     351.68      |      0.22       |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       |      30.41      |      28.38      |      89.08      |      0.22       |
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
        
