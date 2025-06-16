import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
|    测试平台  |     测试程序      |             测试模型                |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      3.16       |      2.39       |      31.99      |      5.88       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      2.58       |      1.37       |      29.53      |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      3.17       |      2.40       |      8.29       |      5.93       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      2.59       |      1.37       |      5.82       |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      3.20       |      2.41       |      5.63       |      5.86       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      2.59       |      1.37       |      3.13       |      3.24       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      2.98       |      2.20       |      4.97       |      5.31       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      2.47       |      1.31       |      3.06       |      3.22       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_fp32_1b.bmodel       |      3.17       |      2.38       |      36.20      |      5.88       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_fp32_1b.bmodel       |      2.59       |      1.37       |      33.73      |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_fp16_1b.bmodel       |      3.15       |      2.40       |      10.09      |      5.87       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_fp16_1b.bmodel       |      2.59       |      1.37       |      7.65       |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_int8_1b.bmodel       |      3.16       |      2.39       |      7.47       |      5.90       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_int8_1b.bmodel       |      2.60       |      1.37       |      5.04       |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov9s_int8_4b.bmodel       |      2.99       |      2.20       |      6.78       |      5.31       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov9s_int8_4b.bmodel       |      2.48       |      1.32       |      4.86       |      3.21       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_fp32_1b.bmodel      |      3.15       |      2.40       |      27.34      |      5.87       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_fp32_1b.bmodel      |      2.59       |      1.37       |      24.86      |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_fp16_1b.bmodel      |      3.16       |      2.39       |      8.58       |      5.88       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_fp16_1b.bmodel      |      2.60       |      1.37       |      6.12       |      3.24       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_int8_1b.bmodel      |      3.16       |      2.39       |      5.94       |      5.79       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_int8_1b.bmodel      |      2.60       |      1.37       |      3.50       |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov11s_int8_4b.bmodel      |      2.98       |      2.21       |      5.21       |      5.23       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov11s_int8_4b.bmodel      |      2.46       |      1.32       |      3.27       |      3.21       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_fp32_1b.bmodel      |      3.17       |      2.40       |      57.73      |      5.81       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_fp32_1b.bmodel      |      2.59       |      1.37       |      55.27      |      3.24       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_fp16_1b.bmodel      |      3.16       |      2.40       |      29.84      |      5.79       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_fp16_1b.bmodel      |      2.59       |      1.37       |      27.38      |      3.24       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_int8_1b.bmodel      |      3.16       |      2.39       |      25.26      |      5.73       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_int8_1b.bmodel      |      2.59       |      1.37       |      22.83      |      3.23       |
|   SE7-32    |  yolov8_bmcv.py   |      yolov12s_int8_4b.bmodel      |      2.99       |      2.21       |      24.17      |      5.21       |
|   SE7-32    |  yolov8_bmcv.soc  |      yolov12s_int8_4b.bmodel      |      2.47       |      1.32       |      22.19      |      3.22       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      4.18       |      4.53       |     165.23      |      7.29       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      4.20       |      4.47       |      38.08      |      7.30       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      4.21       |      4.49       |      11.08      |      7.46       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      3.99       |      4.17       |      10.24      |      6.51       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      3.28       |      2.60       |     162.00      |      4.06       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      3.31       |      2.58       |      34.93      |      4.03       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      3.28       |      2.58       |      7.99       |      4.03       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      3.09       |      2.46       |      7.90       |      3.98       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_int8_4b_2core.bmodel    |      3.98       |      4.19       |      7.75       |      6.50       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_int8_4b_2core.bmodel    |      3.10       |      2.46       |      5.17       |      3.99       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_fp32_1b.bmodel       |      4.19       |      4.51       |     165.90      |      7.28       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_fp16_1b.bmodel       |      4.19       |      4.47       |      44.67      |      7.30       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_int8_1b.bmodel       |      4.18       |      4.47       |      21.49      |      7.29       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov9s_int8_4b.bmodel       |      3.98       |      4.18       |      20.57      |      6.51       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_fp32_1b.bmodel       |      3.25       |      2.59       |     162.66      |      4.05       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_fp16_1b.bmodel       |      3.29       |      2.58       |      41.53      |      4.04       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_int8_1b.bmodel       |      3.29       |      2.58       |      18.34      |      4.04       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov9s_int8_4b.bmodel       |      3.11       |      2.46       |      18.14      |      3.99       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov9s_int8_4b_2core.bmodel    |      3.96       |      4.17       |      12.82      |      6.51       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov9s_int8_4b_2core.bmodel    |      3.09       |      2.46       |      10.47      |      3.99       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_fp32_1b.bmodel      |      4.21       |      4.49       |     135.11      |      7.25       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_fp16_1b.bmodel      |      4.17       |      4.50       |      37.24      |      7.27       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_int8_1b.bmodel      |      4.20       |      4.48       |      11.83      |      7.18       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov11s_int8_4b.bmodel      |      4.00       |      4.19       |      10.75      |      6.40       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_fp32_1b.bmodel      |      3.27       |      2.60       |     131.90      |      4.06       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_fp16_1b.bmodel      |      3.29       |      2.58       |      34.18      |      4.04       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_int8_1b.bmodel      |      3.29       |      2.59       |      8.68       |      4.02       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov11s_int8_4b.bmodel      |      3.10       |      2.46       |      8.33       |      3.98       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov11s_int8_4b_2core.bmodel   |      3.99       |      4.18       |      8.23       |      6.40       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov11s_int8_4b_2core.bmodel   |      3.11       |      2.46       |      5.68       |      3.99       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_fp32_1b.bmodel      |      8.54       |      4.52       |     213.61      |      7.60       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_fp32_1b.bmodel      |      3.21       |      2.61       |     210.32      |      4.08       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_fp16_1b.bmodel      |      5.30       |      4.51       |      72.61      |      7.58       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_fp16_1b.bmodel      |      3.25       |      2.61       |      69.31      |      4.05       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_int8_1b.bmodel      |      5.83       |      4.50       |      63.46      |      7.48       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_int8_1b.bmodel      |      3.27       |      2.61       |      60.38      |      4.05       |
|   SE9-16    |  yolov8_bmcv.py   |      yolov12s_int8_4b.bmodel      |      5.35       |      4.16       |      62.94      |      6.62       |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov12s_int8_4b.bmodel      |      3.06       |      2.49       |      60.57      |      4.00       |
|   SE9-16    |  yolov8_bmcv.py   |   yolov12s_int8_4b_2core.bmodel   |      5.55       |      4.16       |      40.36      |      6.63       |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov12s_int8_4b_2core.bmodel   |      3.08       |      2.49       |      37.96      |      4.01       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      4.22       |      4.51       |     168.83      |      7.57       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      4.25       |      4.50       |      40.15      |      7.66       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      4.21       |      4.48       |      12.62      |      7.59       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      3.95       |      4.18       |      11.83      |      6.89       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      3.25       |      2.61       |     165.69      |      4.05       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      3.27       |      2.60       |      37.13      |      4.02       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      3.27       |      2.60       |      9.54       |      4.04       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      3.07       |      2.48       |      9.55       |      3.99       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_fp32_1b.bmodel       |      4.16       |      4.51       |     169.02      |      7.59       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_fp16_1b.bmodel       |      4.17       |      4.50       |      47.11      |      7.56       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_int8_1b.bmodel       |      4.17       |      4.51       |      23.57      |      7.58       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov9s_int8_4b.bmodel       |      3.95       |      4.17       |      22.70      |      6.80       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_fp32_1b.bmodel       |      3.25       |      2.61       |     165.91      |      4.05       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_fp16_1b.bmodel       |      3.27       |      2.60       |      44.02      |      4.03       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_int8_1b.bmodel       |      3.27       |      2.60       |      20.50      |      4.03       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov9s_int8_4b.bmodel       |      3.07       |      2.48       |      20.37      |      4.01       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_fp32_1b.bmodel      |      4.21       |      4.52       |     138.64      |      7.56       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_fp16_1b.bmodel      |      4.16       |      4.49       |      39.61      |      7.55       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_int8_1b.bmodel      |      4.18       |      4.49       |      14.02      |      7.43       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov11s_int8_4b.bmodel      |      3.96       |      4.18       |      12.79      |      6.65       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_fp32_1b.bmodel      |      3.26       |      2.61       |     135.49      |      4.05       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_fp16_1b.bmodel      |      3.27       |      2.60       |      36.59      |      4.04       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_int8_1b.bmodel      |      3.27       |      2.60       |      10.93      |      4.03       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov11s_int8_4b.bmodel      |      3.08       |      2.48       |      10.47      |      4.00       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_fp32_1b.bmodel      |      8.81       |      4.52       |     213.03      |      7.63       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_fp32_1b.bmodel      |      3.26       |      2.60       |     209.85      |      4.08       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_fp16_1b.bmodel      |      4.21       |      4.49       |      72.26      |      7.59       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_fp16_1b.bmodel      |      3.24       |      2.60       |      69.05      |      4.05       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_int8_1b.bmodel      |      4.77       |      4.50       |      63.41      |      7.48       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_int8_1b.bmodel      |      3.24       |      2.60       |      60.16      |      4.05       |
|    SE9-8    |  yolov8_bmcv.py   |      yolov12s_int8_4b.bmodel      |      4.48       |      4.15       |      62.67      |      6.63       |
|    SE9-8    |  yolov8_bmcv.soc  |      yolov12s_int8_4b.bmodel      |      3.05       |      2.49       |      60.36      |      4.02       |
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
        
