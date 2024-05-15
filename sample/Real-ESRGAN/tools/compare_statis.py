import re
import argparse
import math
import os
import sys

baseline = """
|    测试平台  |     测试程序      |             测试模型                 |   decode_time   | preprocess_time | inference_time  | postprocess_time| 
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      10.07      |      17.70      |     761.75      |      71.56      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      10.05      |      18.25      |     115.09      |      71.27      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      10.04      |      17.91      |     332.37      |      71.17      |
|   SE7-32    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      10.12      |      18.15      |     331.62      |      74.14      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      2.16       |      1.98       |     722.67      |     108.27      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      1.85       |      2.00       |      75.89      |     108.32      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      1.84       |      1.55       |      35.60      |      58.42      |
|   SE7-32    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      1.49       |      1.37       |      34.65      |      58.64      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      1.26       |      0.61       |     711.07      |      51.92      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      1.26       |      0.61       |      64.19      |      52.00      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      1.23       |      0.46       |      32.58      |     102.21      |
|   SE7-32    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      1.11       |      0.43       |      31.70      |      98.82      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      35.20      |      23.67      |     3803.45     |      94.61      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      33.24      |      23.37      |     512.38      |      96.68      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      23.94      |      23.70      |     536.92      |      96.74      |
|    SE9-8    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      21.02      |      24.59      |     544.08      |      190.15     |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      22.99      |      3.82       |     3758.30     |     132.90      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      20.30      |      3.79       |     467.83      |     135.42      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      3.46       |      3.47       |     125.32      |      76.28      |
|    SE9-8    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      8.95       |      3.01       |     121.99      |      99.81      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      15.14      |      1.60       |     3742.03     |     109.67      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      4.92       |      1.61       |     451.65      |     110.71      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      4.99       |      1.59       |     120.86      |     123.16      |
|    SE9-8    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      4.65       |      1.50       |     118.23      |     123.56      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp32_1b.bmodel      |      22.25      |      23.44      |     3791.06     |      87.81      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_fp16_1b.bmodel      |      14.34      |      23.80      |     500.06      |      87.83      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_1b.bmodel      |      14.30      |      23.49      |     548.05      |      87.49      |
|   SE9-16    |real_esrgan_opencv.py|    real_esrgan_int8_4b.bmodel      |      18.25      |      23.48      |     545.85      |     137.49      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp32_1b.bmodel        |      5.18       |      4.09       |     3746.08     |     131.82      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_fp16_1b.bmodel        |      3.61       |      4.09       |     455.72      |     132.93      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_1b.bmodel        |      3.57       |      3.69       |     120.61      |      76.14      |
|   SE9-16    |real_esrgan_bmcv.py|    real_esrgan_int8_4b.bmodel        |      3.10       |      3.27       |     118.65      |      92.00      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp32_1b.bmodel       |      4.29       |      1.61       |     3729.59     |      68.16      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_fp16_1b.bmodel       |      3.78       |      1.62       |     439.28      |      65.37      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_1b.bmodel       |      3.76       |      1.59       |     116.18      |      80.84      |
|   SE9-16    |real_esrgan_bmcv.soc|    real_esrgan_int8_4b.bmodel       |      3.67       |      1.49       |     114.54      |      78.07      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_fp32_1b_2core.bmodel   |      14.25      |      23.74      |     1946.79     |      87.87      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_fp16_1b_2core.bmodel   |      14.30      |      23.32      |     292.34      |      88.20      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_int8_1b_2core.bmodel   |      14.28      |      23.46      |     519.40      |      87.98      |
|   SE9-16    |real_esrgan_opencv.py| real_esrgan_int8_4b_2core.bmodel   |      22.25      |      23.60      |     492.16      |     153.62      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_fp32_1b_2core.bmodel     |      17.72      |      3.99       |     1901.91     |     132.33      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_fp16_1b_2core.bmodel     |      3.53       |      4.03       |     247.17      |     132.53      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_int8_1b_2core.bmodel     |      3.55       |      3.86       |      92.06      |      75.69      |
|   SE9-16    |real_esrgan_bmcv.py| real_esrgan_int8_4b_2core.bmodel     |      3.07       |      3.23       |      64.27      |      85.18      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_fp32_1b_2core.bmodel    |      4.68       |      1.61       |     1885.23     |      65.61      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_fp16_1b_2core.bmodel    |      4.14       |      1.62       |     230.77      |      67.16      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_int8_1b_2core.bmodel    |      4.12       |      1.59       |      87.69      |      81.85      |
|   SE9-16    |real_esrgan_bmcv.soc| real_esrgan_int8_4b_2core.bmodel    |      4.17       |      1.49       |      60.19      |      77.95      |

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
        
