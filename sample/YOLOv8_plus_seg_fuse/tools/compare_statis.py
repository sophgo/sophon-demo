import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
|    测试平台  |     测试程序      |             测试模型                |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ---------------- | ----------------------------------- | -------- | ---------     | ---------     | --------- |
|   SE7-32    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b.bmodel  | 3.04 | 1.17 | 44.73 | 9.52  |
|   SE7-32    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b.bmodel  | 3.04 | 1.17 | 9.76  | 10.60 |
|   SE7-32    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b.bmodel  | 3.01 | 1.17 | 7.04  | 7.46  |
|   SE7-32    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b.bmodel  | 2.70 | 0.45 | 44.51 | 6.50  |
|   SE7-32    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b.bmodel  | 2.69 | 0.45 | 9.55  | 6.63  |
|   SE7-32    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b.bmodel  | 2.68 | 0.45 | 6.84  | 2.78  |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b.bmodel  | 3.99 | 2.80 | 239.68 | 12.40 |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b.bmodel  | 3.97 | 2.80 | 50.64  | 13.17 |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b.bmodel  | 3.96 | 2.80 | 29.09  | 12.91 |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b.bmodel  | 3.40 | 1.20 | 239.26 | 8.73  |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b.bmodel  | 3.41 | 1.19 | 50.29  | 9.05  |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b.bmodel  | 3.35 | 1.20 | 28.75  | 6.56  |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b_2core.bmodel  | 3.97 | 2.80 | 129.88 | 12.40 |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b_2core.bmodel  | 3.97 | 2.80 | 32.14  | 13.17 |
|   SE9-16    |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b_2core.bmodel  | 3.95 | 2.80 | 21.25  | 12.91 |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b_2core.bmodel  | 3.40 | 1.20 | 129.50 | 9.04  |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b_2core.bmodel  | 3.39 | 1.19 | 31.79  | 9.00  |
|   SE9-16    |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b_2core.bmodel  | 3.37 | 1.19 | 20.91  | 6.55  |
|   SE9-8     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp32_1b.bmodel  | 3.86 | 2.82 | 239.71 | 13.38 |
|   SE9-8     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp32_1b.bmodel  | 4.35 | 1.20 | 239.26 | 10.16  |
|   SE9-8     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_fp16_1b.bmodel  | 3.87 | 2.81 | 50.64  | 14.55 |
|   SE9-8     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_fp16_1b.bmodel  | 3.39 | 1.20 | 50.27  | 9.74   |
|   SE9-8     |  yolov8_bmcv.py   |  yolov8s_seg_fuse_int8_1b.bmodel  | 3.85 | 2.81 | 29.09  | 13.52 |
|   SE9-8     |  yolov8_bmcv.soc  |  yolov8s_seg_fuse_int8_1b.bmodel  | 3.33 | 1.20 | 28.72  | 6.43   |
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
        
