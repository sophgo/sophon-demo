import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
|    测试平台  |     测试程序      |             测试模型                |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
|   SE5-16    | yolov9_opencv.py  |      yolov9c_fp32_1b.bmodel       |      6.82       |      21.83      |     116.21      |      72.16      |
|   SE5-16    | yolov9_opencv.py  |      yolov9c_int8_1b.bmodel       |      6.84       |      21.95      |      65.04      |      68.01      |
|   SE5-16    | yolov9_opencv.py  |      yolov9c_int8_4b.bmodel       |      6.86       |      23.20      |      32.77      |      64.29      |
|   SE5-16    |  yolov9_bmcv.py   |      yolov9c_fp32_1b.bmodel       |      3.84       |      2.80       |     113.79      |      77.57      |
|   SE5-16    |  yolov9_bmcv.py   |      yolov9c_int8_1b.bmodel       |      3.88       |      2.81       |      62.65      |      71.35      |
|   SE5-16    |  yolov9_bmcv.py   |      yolov9c_int8_4b.bmodel       |      3.53       |      2.59       |      29.32      |      68.15      |
|   SE5-16    |  yolov9_bmcv.soc  |      yolov9c_fp32_1b.bmodel       |      5.02       |      1.55       |     108.19      |      83.33      |
|   SE5-16    |  yolov9_bmcv.soc  |      yolov9c_int8_1b.bmodel       |      5.03       |      1.55       |      57.07      |      76.70      |
|   SE5-16    |  yolov9_bmcv.soc  |      yolov9c_int8_4b.bmodel       |      4.99       |      1.49       |      24.48      |      73.58      |
|   SE7-32    | yolov9_opencv.py  |      yolov9c_fp32_1b.bmodel       |      6.88       |      23.32      |     146.29      |      79.88      |
|   SE7-32    | yolov9_opencv.py  |      yolov9c_fp16_1b.bmodel       |      6.83       |      22.81      |      30.36      |      80.03      |
|   SE7-32    | yolov9_opencv.py  |      yolov9c_int8_1b.bmodel       |      6.79       |      22.75      |      18.81      |      73.48      |
|   SE7-32    | yolov9_opencv.py  |      yolov9c_int8_4b.bmodel       |      6.83       |      22.59      |      17.23      |      73.51      |
|   SE7-32    |  yolov9_bmcv.py   |      yolov9c_fp32_1b.bmodel       |      3.35       |      2.35       |     143.56      |      82.37      |
|   SE7-32    |  yolov9_bmcv.py   |      yolov9c_fp16_1b.bmodel       |      3.35       |      2.35       |      27.55      |      83.45      |
|   SE7-32    |  yolov9_bmcv.py   |      yolov9c_int8_1b.bmodel       |      3.35       |      2.36       |      15.95      |      76.93      |
|   SE7-32    |  yolov9_bmcv.py   |      yolov9c_int8_4b.bmodel       |      3.00       |      2.13       |      14.72      |      77.48      |
|   SE7-32    |  yolov9_bmcv.soc  |      yolov9c_fp32_1b.bmodel       |      4.45       |      0.74       |     137.37      |      89.15      |
|   SE7-32    |  yolov9_bmcv.soc  |      yolov9c_fp16_1b.bmodel       |      4.49       |      0.74       |      21.39      |      91.95      |
|   SE7-32    |  yolov9_bmcv.soc  |      yolov9c_int8_1b.bmodel       |      4.52       |      0.74       |      9.79       |      82.50      |
|   SE7-32    |  yolov9_bmcv.soc  |      yolov9c_int8_4b.bmodel       |      4.50       |      0.71       |      9.30       |      82.11      |
|   SE9-16    | yolov9_opencv.py  |      yolov9c_fp32_1b.bmodel       |      9.48       |      29.76      |     792.35      |     101.83      |
|   SE9-16    | yolov9_opencv.py  |      yolov9c_fp16_1b.bmodel       |      9.47       |      29.93      |     158.00      |     105.56      |
|   SE9-16    | yolov9_opencv.py  |      yolov9c_int8_1b.bmodel       |      9.51       |      30.48      |      42.30      |      92.94      |
|   SE9-16    | yolov9_opencv.py  |      yolov9c_int8_4b.bmodel       |      9.45       |      29.78      |      40.45      |      92.51      |
|   SE9-16    |  yolov9_bmcv.py   |      yolov9c_fp32_1b.bmodel       |      4.65       |      4.71       |     789.43      |     105.47      |
|   SE9-16    |  yolov9_bmcv.py   |      yolov9c_fp16_1b.bmodel       |      4.67       |      4.71       |     154.50      |     101.52      |
|   SE9-16    |  yolov9_bmcv.py   |      yolov9c_int8_1b.bmodel       |      4.77       |      4.69       |      38.99      |      96.56      |
|   SE9-16    |  yolov9_bmcv.py   |      yolov9c_int8_4b.bmodel       |      4.21       |      4.28       |      37.62      |      95.59      |
|   SE9-16    |  yolov9_bmcv.soc  |      yolov9c_fp32_1b.bmodel       |      5.87       |      1.74       |     781.26      |     119.35      |
|   SE9-16    |  yolov9_bmcv.soc  |      yolov9c_fp16_1b.bmodel       |      5.88       |      1.74       |     146.66      |     118.59      |
|   SE9-16    |  yolov9_bmcv.soc  |      yolov9c_int8_1b.bmodel       |      5.88       |      1.73       |      31.11      |     110.35      |
|   SE9-16    |  yolov9_bmcv.soc  |      yolov9c_int8_4b.bmodel       |      5.87       |      1.66       |      30.72      |     110.37      |
|   SE9-16    | yolov9_opencv.py  |   yolov9c_fp32_1b_2core.bmodel    |      9.49       |      30.45      |     415.48      |     100.86      |
|   SE9-16    | yolov9_opencv.py  |   yolov9c_fp16_1b_2core.bmodel    |      9.48       |      30.51      |      91.45      |     100.04      |
|   SE9-16    | yolov9_opencv.py  |   yolov9c_int8_1b_2core.bmodel    |      9.45       |      30.56      |      33.31      |      94.72      |
|   SE9-16    | yolov9_opencv.py  |   yolov9c_int8_4b_2core.bmodel    |      9.48       |      32.91      |      29.48      |      90.92      |
|   SE9-16    |  yolov9_bmcv.py   |   yolov9c_fp32_1b_2core.bmodel    |      4.69       |      4.72       |     412.10      |     106.78      |
|   SE9-16    |  yolov9_bmcv.py   |   yolov9c_fp16_1b_2core.bmodel    |      4.69       |      4.77       |      87.89      |     100.90      |
|   SE9-16    |  yolov9_bmcv.py   |   yolov9c_int8_1b_2core.bmodel    |      4.68       |      4.75       |      30.05      |      96.75      |
|   SE9-16    |  yolov9_bmcv.py   |   yolov9c_int8_4b_2core.bmodel    |      4.25       |      4.30       |      24.49      |      96.67      |
|   SE9-16    |  yolov9_bmcv.soc  |   yolov9c_fp32_1b_2core.bmodel    |      5.91       |      1.74       |     404.17      |     118.55      |
|   SE9-16    |  yolov9_bmcv.soc  |   yolov9c_fp16_1b_2core.bmodel    |      5.90       |      1.73       |      80.05      |     118.54      |
|   SE9-16    |  yolov9_bmcv.soc  |   yolov9c_int8_1b_2core.bmodel    |      5.92       |      1.74       |      22.14      |     110.50      |
|   SE9-16    |  yolov9_bmcv.soc  |   yolov9c_int8_4b_2core.bmodel    |      5.87       |      1.65       |      17.90      |     110.19      |
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
    parser.add_argument('--bmodel', type=str, default='yolov9c_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov9_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov9c_fp32_1b.bmodel_python_test.log')
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
        if abs(statis - extracted_data[key]) / statis > 0.4:
            print("{:} time, diff ratio > 0.4".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False

    benchmark_str = "|{:^13}|{:^19}|{:^35}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)

    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)

    if compare_pass == False:
        sys.exit(1)
