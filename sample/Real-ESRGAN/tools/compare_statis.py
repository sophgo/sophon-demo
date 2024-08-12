import re
import argparse
import math
import os
import sys

baseline = """
|    测试平台  |     测试程序      |             测试模型                 |   decode_time   | preprocess_time | inference_time  | postprocess_time| 
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      1.86       |      18.56      |     761.65      |      80.92      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      1.84       |      18.70      |     114.75      |      72.18      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      1.83       |      18.46      |     344.08      |      71.63      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      1.76       |      19.39      |     342.58      |      83.54      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      1.75       |      1.92       |     722.50      |     106.98      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      1.73       |      1.93       |      75.53      |     106.83      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      1.75       |      1.51       |      35.57      |      58.49      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      1.41       |      1.36       |      34.53      |      58.79      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      2.25       |      0.61       |     711.02      |      51.60      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      2.25       |      0.61       |      64.18      |      51.72      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      2.27       |      0.46       |      32.59      |      93.60      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      1.95       |      0.43       |      31.69      |      91.32      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      13.90      |      43.32      |     3803.27     |      90.92      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      3.21       |      42.89      |     512.60      |      87.89      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      3.20       |      42.98      |     537.23      |      87.68      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      8.46       |      47.41      |     541.77      |     213.41      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      13.33      |      3.73       |     3758.24     |     131.44      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      3.22       |      3.73       |     467.73      |     132.85      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      3.21       |      3.38       |     125.12      |      74.66      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      2.86       |      3.02       |     122.24      |     106.95      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      5.54       |      1.60       |     3741.92     |     108.41      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      4.82       |      1.59       |     451.58      |     108.37      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      5.02       |      1.58       |     120.78      |     121.39      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      4.53       |      1.49       |     118.20      |     121.00      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      3.32       |      24.22      |     3791.12     |      88.00      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      3.26       |      23.95      |     500.79      |      88.22      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      3.26       |      23.90      |     548.49      |      88.13      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      3.14       |      24.95      |     545.45      |     113.01      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      3.24       |      3.81       |     3746.02     |     132.69      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      3.26       |      3.85       |     455.66      |     132.52      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      3.24       |      3.46       |     120.55      |      76.49      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      2.93       |      3.08       |     118.63      |      76.70      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      3.64       |      1.52       |     3729.57     |      65.26      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      3.64       |      1.53       |     439.27      |      65.06      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      3.66       |      1.52       |     116.17      |      80.13      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      3.40       |      1.43       |     114.54      |      77.70      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_fp32_1b_2core.bmodel   |      3.30       |      24.39      |     1946.97     |      88.10      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_fp16_1b_2core.bmodel   |      3.28       |      24.25      |     292.25      |      88.05      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_int8_1b_2core.bmodel   |      3.30       |      24.10      |     520.30      |      88.24      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_int8_4b_2core.bmodel   |      3.17       |      24.98      |     490.98      |     106.71      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_fp32_1b_2core.bmodel     |      3.24       |      3.87       |     1901.67     |     131.68      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_fp16_1b_2core.bmodel     |      3.23       |      3.85       |     247.06      |     133.33      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_int8_1b_2core.bmodel     |      3.25       |      3.48       |      92.04      |      76.11      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_int8_4b_2core.bmodel     |      2.96       |      3.07       |      64.27      |      76.60      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_fp32_1b_2core.bmodel    |      3.65       |      1.52       |     1885.24     |      64.88      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_fp16_1b_2core.bmodel    |      3.65       |      1.52       |     230.78      |      64.88      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_int8_1b_2core.bmodel    |      3.64       |      1.51       |      87.69      |      77.74      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_int8_4b_2core.bmodel    |      3.44       |      1.43       |      60.18      |      77.54      |

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
        
