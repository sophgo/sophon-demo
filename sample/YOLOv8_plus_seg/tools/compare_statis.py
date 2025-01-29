import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
|    测试平台  |     测试程序      |             测试模型                |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      7.70       |      2.40       |      47.94      |      79.62      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      3.39       |      2.39       |      13.71      |      79.90      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      3.38       |      2.38       |      10.34      |      77.68      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      3.01       |      2.20       |      9.22       |      73.47      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      2.63       |      1.35       |      42.25      |      33.24      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      2.63       |      1.35       |      8.12       |      34.47      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      2.62       |      1.35       |      4.61       |      31.42      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      2.49       |      1.30       |      4.47       |      30.96      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_fp32_1b.bmodel       |      4.58       |      2.40       |     143.60      |      79.54      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_fp16_1b.bmodel       |      3.34       |      2.39       |      27.87      |      86.25      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_int8_1b.bmodel       |      3.35       |      2.39       |      16.29      |      97.64      |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9c_int8_4b.bmodel       |      3.01       |      2.21       |      14.99      |      93.01      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_fp32_1b.bmodel       |      2.62       |      1.36       |     137.97      |      35.07      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_fp16_1b.bmodel       |      2.62       |      1.36       |      22.21      |      34.56      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_int8_1b.bmodel       |      2.63       |      1.35       |      10.65      |      38.48      |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9c_int8_4b.bmodel       |      2.50       |      1.30       |      10.30      |      37.94      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      7.92       |      4.66       |     240.53      |      91.33      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      4.65       |      4.60       |      53.84      |      88.29      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      4.61       |      4.55       |      18.69      |      84.55      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      4.09       |      4.18       |      17.18      |      85.46      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      3.35       |      2.59       |     233.53      |      43.24      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      3.42       |      2.59       |      46.89      |      43.30      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      3.40       |      2.59       |      11.60      |      40.85      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      3.19       |      2.47       |      11.53      |      40.78      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_fp32_1b_2core.bmodel    |      4.64       |      4.62       |     129.53      |      89.23      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_fp16_1b_2core.bmodel    |      4.59       |      4.59       |      34.87      |      90.64      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_int8_1b_2core.bmodel    |      4.57       |      4.56       |      16.31      |      86.65      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_int8_4b_2core.bmodel    |      4.09       |      4.19       |      13.38      |      87.41      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_fp32_1b_2core.bmodel    |      3.36       |      2.59       |     122.59      |      43.19      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_fp16_1b_2core.bmodel    |      3.39       |      2.59       |      27.67      |      43.35      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_int8_1b_2core.bmodel    |      3.38       |      2.60       |      9.13       |      40.80      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_int8_4b_2core.bmodel    |      3.16       |      2.48       |      7.41       |      40.71      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_fp32_1b.bmodel       |      4.72       |      4.66       |     789.65      |      92.48      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_fp16_1b.bmodel       |      4.67       |      4.67       |     156.08      |      92.12      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_int8_1b.bmodel       |      4.67       |      4.59       |      39.43      |     106.58      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9c_int8_4b.bmodel       |      4.11       |      4.16       |      37.73      |     103.82      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_fp32_1b.bmodel       |      3.38       |      2.59       |     782.38      |      45.98      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_fp16_1b.bmodel       |      3.39       |      2.60       |     149.05      |      45.31      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_int8_1b.bmodel       |      3.39       |      2.60       |      32.42      |      48.44      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9c_int8_4b.bmodel       |      3.18       |      2.47       |      31.99      |      48.66      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov9c_fp32_1b_2core.bmodel    |      4.65       |      4.62       |     407.49      |      92.53      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov9c_fp16_1b_2core.bmodel    |      4.65       |      4.60       |      88.61      |      97.73      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov9c_int8_1b_2core.bmodel    |      4.66       |      4.56       |      28.37      |     101.84      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov9c_int8_4b_2core.bmodel    |      4.11       |      4.18       |      24.77      |     104.64      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov9c_fp32_1b_2core.bmodel    |      3.42       |      2.59       |     400.30      |      45.56      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov9c_fp16_1b_2core.bmodel    |      3.39       |      2.60       |      81.50      |      45.34      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov9c_int8_1b_2core.bmodel    |      3.38       |      2.59       |      21.34      |      48.58      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov9c_int8_4b_2core.bmodel    |      3.13       |      2.46       |      18.95      |      48.48      |
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
    parser.add_argument('--bmodel', type=str, default='yolov8s_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov8_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov8s_fp32_1b.bmodel_python_test.log')
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
        
