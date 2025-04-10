import re
import argparse
import math
import os
import sys

baseline = """
|   测试平台  |     测试程序        |             测试模型               |   decode_time   | preprocess_time | inference_time  |postprocess_time  |
| ----------  | -----------------   | -----------------------------------|-----------------|-----------------|-----------------|-----------------|
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel     |      1.87       |      18.88      |     766.08   |      71.21      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel     |      1.86       |      18.61      |     114.84   |      71.46      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel     |      1.85       |      18.60      |     344.06   |      71.41      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel     |      1.78       |      19.32      |     342.83    |      79.77      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel     |      1.81       |      1.96       |     723.02      |  109.33      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel     |      1.81       |      1.97       |      75.08      |  109.82      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel     |      1.83       |      1.55       |      35.46      |   58.71      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel     |      1.46       |      1.38       |      34.57      |   60.03      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel     |      1.32       |      1.11       |     724.83      |   97.69      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel     |      1.34       |      1.11       |      76.64      |   97.63      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel     |      1.30       |      0.69       |      33.92      |   3.10       |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel     |      1.14       |      0.63       |      33.13      |   3.09       |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel     |      4.09       |      24.75      |     3794.30     |      89.71      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel     |      3.38       |      24.39      |     503.90      |      89.40      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel     |      3.35       |      24.42      |     549.36      |      90.44      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel     |      3.11       |      24.95      |     546.77      |      88.84      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_int8_4b_2core.bmodel  |      3.12       |      24.91      |     142.10      |      89.03      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel     |      3.23       |      3.59       |     3746.51     |     138.87      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel     |      3.28       |      3.63       |     455.04      |     139.14      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel     |      3.25       |      3.24       |     119.96      |      75.59      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel     |      2.85       |      2.87       |     118.97      |      76.46      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_int8_4b_2core.bmodel  |      2.85       |      2.86       |      64.48      |      75.93      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel     |      2.49       |      2.14       |     3738.22     |     130.09      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel     |      2.50       |      2.13       |     446.65      |     129.98      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel     |      2.43       |      1.74       |     117.58      |      10.39      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel     |      2.18       |      1.61       |     116.93      |      10.40      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_int8_4b_2core.bmodel  |      2.18       |      1.61       |      62.47      |      10.37      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel     |      17.15      |      24.43      |     3813.90     |     132.27      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel     |      5.63       |      24.49      |     516.81      |     132.07      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel     |      5.66       |      24.60      |     550.77      |     132.19      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel     |      11.43      |      24.77      |     549.46      |     237.89      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel     |      12.47      |      3.60       |     3765.67     |     142.59      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel     |      3.41       |      3.57       |     466.11      |     142.27      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel     |      3.46       |      3.21       |     123.81      |      77.01      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel     |      9.51       |      2.85       |     122.29      |     157.92      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel     |      16.42      |      2.13       |     3757.22     |     164.12      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel     |      2.48       |      2.13       |     457.72      |     163.80      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel     |      2.46       |      1.74       |     121.44      |      10.39      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel     |      2.19       |      1.61       |     120.20      |      10.35      |
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
    parser.add_argument('--bmodel', type=str, default='real_esrgan_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='real_esrgan_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../python/log/bmcv_real_esrgan_fp32_1b.bmodel_debug.log')
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
        elif args.target == "CV186X":
            platform = "SE9-8"
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
        
