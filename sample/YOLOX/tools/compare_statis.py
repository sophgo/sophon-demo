import re
import argparse
import math
import os
import sys

baseline = """
|    测试平台  |     测试程序      |     测试模型          |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | --------------------- | --------  | -------------- | ---------     | --------- |
| BM1684 SoC  | yolox_opencv.py  | yolox_s_fp32_1b.bmodel | 15.18    | 3.63           | 39.70         | 2.71     |
| BM1684 SoC  | yolox_opencv.py  | yolox_s_int8_1b.bmodel | 15.20    | 3.64           | 33.27         | 2.69     |
| BM1684 SoC  | yolox_opencv.py  | yolox_s_int8_4b.bmodel | 15.20    | 5.54           | 22.66         | 2.36     |
| BM1684 SoC  | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel | 3.70     | 2.88           | 28.16         | 2.72     |
| BM1684 SoC  | yolox_bmcv.py    | yolox_s_int8_1b.bmodel | 3.50     | 2.22           | 21.62         | 2.71     |
| BM1684 SoC  | yolox_bmcv.py    | yolox_s_int8_4b.bmodel | 3.38     | 2.06           | 10.68         | 2.35     |
| BM1684 SoC  | yolox_bmcv.soc   | yolox_s_fp32_1b.bmodel | 4.86     | 1.47           | 25.78         | 2.68     |
| BM1684 SoC  | yolox_bmcv.soc   | yolox_s_int8_1b.bmodel | 4.57     | 1.46           | 19.44         | 2.68     |
| BM1684 SoC  | yolox_bmcv.soc   | yolox_s_int8_4b.bmodel | 4.76     | 1.41           | 8.96          | 2.07     |
| BM1684 SoC  | yolox_sail.soc   | yolox_s_fp32_1b.bmodel | 3.22     | 3.11           | 26.16         | 2.08     |
| BM1684 SoC  | yolox_sail.soc   | yolox_s_int8_1b.bmodel | 3.21     | 3.11           | 19.83         | 2.07     |
| BM1684 SoC  | yolox_sail.soc   | yolox_s_int8_4b.bmodel | 3.14     | 2.73           | 9.26          | 2.10     |
| BM1684X SoC | yolox_opencv.py  | yolox_s_fp32_1b.bmodel | 13.87    | 3.40           | 44.02         | 2.84     |
| BM1684X SoC | yolox_opencv.py  | yolox_s_fp16_1b.bmodel | 13.88    | 3.24           | 22.04         | 2.84     |
| BM1684X SoC | yolox_opencv.py  | yolox_s_int8_1b.bmodel | 13.98    | 3.49           | 20.94         | 2.82     |
| BM1684X SoC | yolox_opencv.py  | yolox_s_int8_4b.bmodel | 13.90    | 5.17           | 20.87         | 2.52     |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel | 3.22     | 2.40           | 30.33         | 2.86     |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_fp16_1b.bmodel | 3.20     | 2.39           | 7.93          | 2.86     |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_int8_1b.bmodel | 3.19     | 2.38           | 6.81          | 2.87     |
| BM1684X SoC | yolox_bmcv.py    | yolox_s_int8_4b.bmodel | 3.03     | 2.19           | 6.18          | 2.51     |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_fp32_1b.bmodel | 4.51     | 0.75           | 27.86         | 2.72     |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_fp16_1b.bmodel | 4.55     | 0.75           | 5.49          | 2.75     |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_int8_1b.bmodel | 4.51     | 0.75           | 4.35          | 2.74     |
| BM1684X SoC | yolox_bmcv.soc   | yolox_s_int8_4b.bmodel | 4.28     | 0.72           | 4.26          | 2.73     |
| BM1684X SoC | yolox_sail.soc   | yolox_s_fp32_1b.bmodel | 2.92     | 2.72           | 28.29         | 2.12     |
| BM1684X SoC | yolox_sail.soc   | yolox_s_fp16_1b.bmodel | 2.88     | 2.71           | 5.91          | 2.10     |
| BM1684X SoC | yolox_sail.soc   | yolox_s_int8_1b.bmodel | 2.81     | 2.72           | 4.76          | 2.10     |
| BM1684X SoC | yolox_sail.soc   | yolox_s_int8_4b.bmodel | 2.67     | 2.63           | 4.55          | 2.12     |
| BM1688 SoC  | yolox_opencv.py  | yolox_s_fp32_1b.bmodel | 21.62    | 4.17           | 174.24        | 3.98     |
| BM1688 SoC  | yolox_opencv.py  | yolox_s_fp16_1b.bmodel | 20.42    | 4.09           | 54.62         | 3.98     |
| BM1688 SoC  | yolox_opencv.py  | yolox_s_int8_1b.bmodel | 20.19    | 4.14           | 40.05         | 3.97     |
| BM1688 SoC  | yolox_bmcv.py    | yolox_s_fp32_1b.bmodel | 4.68     | 5.18           | 157.73        | 4.01     |
| BM1688 SoC  | yolox_bmcv.py    | yolox_s_fp16_1b.bmodel | 5.16     | 5.25           | 38.22         | 4.03     |
| BM1688 SoC  | yolox_bmcv.py    | yolox_s_int8_1b.bmodel | 4.53     | 5.18           | 23.63         | 3.99     |
| BM1688 SoC  | yolox_bmcv.soc   | yolox_s_fp32_1b.bmodel | 5.84     | 1.96           | 154.59        | 3.76     |
| BM1688 SoC  | yolox_bmcv.soc   | yolox_s_fp16_1b.bmodel | 5.83     | 1.94           | 35.05         | 3.75     |
| BM1688 SoC  | yolox_bmcv.soc   | yolox_s_int8_1b.bmodel | 5.78     | 1.95           | 20.42         | 3.74     |
| BM1688 SoC  | yolox_sail.soc   | yolox_s_fp32_1b.bmodel | 3.94     | 5.22           | 155.22        | 2.91     | 
| BM1688 SoC  | yolox_sail.soc   | yolox_s_fp16_1b.bmodel | 3.92     | 5.21           | 5.65          | 2.91     |
| BM1688 SoC  | yolox_sail.soc   | yolox_s_int8_1b.bmodel | 4.01     | 5.23           | 21.04         | 2.91     |
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
    parser.add_argument('--bmodel', type=str, default='yolox_s_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolox_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolox_s_fp32_1b.bmodel_python_test.log')
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
        
