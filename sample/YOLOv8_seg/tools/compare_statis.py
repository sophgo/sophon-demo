import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
|    测试平台  |     测试程序      |             测试模型                |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
| SE5-16      | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 3.40 | 26.79 | 46.53 | 165.20 |
| SE5-16      | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 3.39  | 27.03 | 34.33 | 162.30 | 
| SE5-16      | yolov8_opencv.py | yolov8s_int8_4b.bmodel | 4.05  | 25.76 | 24.19 | 139.20 |
| SE5-16      | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 2.84  | 2.99  | 43.67 | 181.50 |
| SE5-16      | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 2.86  | 3.01  | 31.73 | 141.70 |
| SE5-16      | yolov8_bmcv.py   | yolov8s_int8_4b.bmodel | 2.51  | 2.82  | 21.48  | 135.00 |
| SE5-16      | yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel | 4.82  | 1.89  | 38.10 | 71.04 |
| SE5-16      | yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel | 4.90  | 1.89  | 26.03 | 66.09 |
| SE5-16      | yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel | 4.56  | 1.81  | 16.04 | 65.49 |
| SE7-32      | yolov8_opencv.py | yolov8s_fp32_1b.bmodel | 3.30   | 28.44 | 51.26 | 181.70 |
| SE7-32      | yolov8_opencv.py | yolov8s_fp16_1b.bmodel | 3.30   | 28.45 | 17.56 | 176.00 |
| SE7-32      | yolov8_opencv.py | yolov8s_int8_1b.bmodel | 3.33  | 27.99 | 13.74 | 161.50 |
| SE7-32      | yolov8_opencv.py | yolov8s_int8_4b.bmodel | 3.90   | 26.37 | 13.71 | 133.90 |
| SE7-32      | yolov8_bmcv.py   | yolov8s_fp32_1b.bmodel | 2.77  | 2.74  | 47.87 | 171.40 |
| SE7-32      | yolov8_bmcv.py   | yolov8s_fp16_1b.bmodel | 2.75  | 2.71  | 13.87 | 167.40 |
| SE7-32      | yolov8_bmcv.py   | yolov8s_int8_1b.bmodel | 2.73  | 2.70  | 10.32 | 153.50 |
| SE7-32      | yolov8_bmcv.py   | yolov8s_int8_4b.bmodel | 2.50  | 2.55  | 9.62  | 160.00 |
| SE7-32      | yolov8_bmcv.soc  | yolov8s_fp32_1b.bmodel | 4.66 | 0.99 | 41.55 | 70.40 |
| SE7-32      | yolov8_bmcv.soc  | yolov8s_fp16_1b.bmodel | 4.53 | 0.99 | 7.53 | 70.70 |
| SE7-32      | yolov8_bmcv.soc  | yolov8s_int8_1b.bmodel | 4.58 | 0.99 | 4.16 | 70.10 |
| SE7-32      | yolov8_bmcv.soc  | yolov8s_int8_4b.bmodel | 4.38 | 0.95 | 3.75 | 70.45 |
|   SE9-16    | yolov8_opencv.py  |      yolov8s_fp32_1b.bmodel       |      24.08      |      30.41      |     244.67      |     99.88       |
|   SE9-16    | yolov8_opencv.py  |      yolov8s_fp16_1b.bmodel       |      19.28      |      29.67      |      56.80      |     100.65      |
|   SE9-16    | yolov8_opencv.py  |      yolov8s_int8_1b.bmodel       |      11.62      |      30.25      |      30.19      |      92.74      |
|   SE9-16    | yolov8_opencv.py  |      yolov8s_int8_4b.bmodel       |      9.48       |      33.13      |      30.24      |      89.66      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp32_1b.bmodel       |      4.64       |      4.82       |     241.44      |     101.20      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_fp16_1b.bmodel       |      4.65       |      4.82       |      53.55      |     100.66      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_1b.bmodel       |      4.71       |      4.85       |      26.72      |      90.80      |
|   SE9-16    |  yolov8_bmcv.py   |      yolov8s_int8_4b.bmodel       |      4.17       |      4.39       |      24.50      |      88.34      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp32_1b.bmodel       |      5.82       |      1.79       |     233.42      |     128.11      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_fp16_1b.bmodel       |      5.85       |      1.78       |      45.59      |     127.78      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_1b.bmodel       |      5.82       |      1.77       |      18.79      |     116.44      |
|   SE9-16    |  yolov8_bmcv.soc  |      yolov8s_int8_4b.bmodel       |      5.79       |      1.68       |      17.84      |     113.72      |
|   SE9-16    | yolov8_opencv.py  |   yolov8s_fp32_1b_2core.bmodel    |      19.26      |      30.54      |     133.66      |     100.09      |
|   SE9-16    | yolov8_opencv.py  |   yolov8s_fp16_1b_2core.bmodel    |      19.23      |      29.67      |      37.49      |     101.03      |
|   SE9-16    | yolov8_opencv.py  |   yolov8s_int8_1b_2core.bmodel    |      9.45       |      30.45      |      23.81      |      89.63      |
|   SE9-16    | yolov8_opencv.py  |   yolov8s_int8_4b_2core.bmodel    |      9.44       |      32.46      |      22.07      |      88.25      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_fp32_1b_2core.bmodel    |      4.71       |      4.82       |     130.19      |      98.58      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_fp16_1b_2core.bmodel    |      4.68       |      4.83       |      34.37      |      96.20      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_int8_1b_2core.bmodel    |      4.74       |      4.84       |      20.31      |      92.89      |
|   SE9-16    |  yolov8_bmcv.py   |   yolov8s_int8_4b_2core.bmodel    |      4.19       |      4.39       |      17.26      |      91.08      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_fp32_1b_2core.bmodel    |      5.85       |      1.78       |     122.27      |     127.10      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_fp16_1b_2core.bmodel    |      5.87       |      1.77       |      26.34      |     125.09      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_int8_1b_2core.bmodel    |      5.84       |      1.77       |      12.11      |     114.18      |
|   SE9-16    |  yolov8_bmcv.soc  |   yolov8s_int8_4b_2core.bmodel    |      5.83       |      1.67       |      10.27      |     113.86      |

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
        
