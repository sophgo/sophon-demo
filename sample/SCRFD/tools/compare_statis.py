import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
|  platform   |      program      |              bmodel               |   decode_time   | preprocess_time | inference_time  |postprocess_time |
|   SE5-16    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      35.23      |      24.33      |      24.99      |      8.54       |
|   SE5-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      30.26      |      24.62      |      21.25      |      8.52       |
|   SE5-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      30.21      |      26.23      |      8.69       |      8.39       |
|   SE5-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.61       |      3.79       |      21.95      |      8.62       |
|   SE5-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      3.61       |      3.77       |      18.13      |      8.53       |
|   SE5-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      3.40       |      3.61       |      5.96       |      8.51       |
|   SE5-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      4.42       |      0.97       |      20.06      |      8.46       |
|   SE5-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      4.40       |      0.97       |      16.25      |      8.17       |
|   SE5-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      4.19       |      0.91       |      5.03       |      8.52       |
|   SE5-16    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.21       |      3.93       |      20.39      |      8.55       |
|   SE5-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      3.22       |      3.94       |      16.58      |      8.38       |
|   SE5-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      3.05       |      3.71       |      5.18       |      7.72       |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      30.32      |      24.94      |      40.36      |      8.45       |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_fp16_1b.bmodel    |      30.03      |      25.25      |      9.34       |      8.40       |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      30.07      |      25.90      |      8.19       |      8.42       |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      30.13      |      27.52      |      6.71       |      8.21       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.08       |      3.00       |      36.84      |      8.55       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp16_1b.bmodel    |      3.10       |      3.01       |      5.83       |      8.49       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      3.07       |      2.99       |      4.65       |      8.60       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      2.90       |      2.80       |      3.55       |      8.32       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.92       |      0.87       |      34.85      |      8.45       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      3.88       |      0.87       |      3.81       |      8.54       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      3.90       |      0.87       |      2.65       |      8.45       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      3.75       |      0.84       |      2.54       |      8.71       |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      2.67       |      3.15       |      35.16      |      8.71       |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      2.66       |      3.15       |      4.16       |      8.87       |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      2.66       |      3.16       |      3.00       |      8.76       |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      2.52       |      3.07       |      2.69       |      7.77       |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      45.48      |      33.19      |     170.02      |      11.41      |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_fp16_1b.bmodel    |      49.88      |      32.64      |      30.87      |      11.49      |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      50.62      |      32.52      |      14.32      |      11.57      |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      45.14      |      35.53      |      12.01      |      11.23      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      9.31       |      6.89       |     165.62      |      11.49      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp16_1b.bmodel    |      12.83      |      6.89       |      26.40      |      11.55      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      12.73      |      6.88       |      9.98       |      11.55      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      9.59       |      6.56       |      8.09       |      11.13      |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      8.38       |      2.43       |     162.74      |      11.78      |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      8.77       |      2.43       |      23.60      |      11.77      |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      6.41       |      2.43       |      7.19       |      11.58      |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      6.03       |      2.35       |      6.76       |      11.93      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      11.47      |      6.84       |     163.27      |      11.98      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      11.45      |      6.84       |      24.09      |      11.93      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      10.16      |      6.83       |      7.67       |      12.02      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      5.65       |      6.66       |      6.96       |      10.95      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_fp32_1b_2core.bmodel |      43.36      |      33.20      |     103.05      |      11.41      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_fp16_1b_2core.bmodel |      48.23      |      32.99      |      23.20      |      11.48      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_int8_1b_2core.bmodel |      50.92      |      33.15      |      13.04      |      11.50      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_int8_4b_2core.bmodel |      45.26      |      36.02      |      9.06       |      11.26      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_fp32_1b_2core.bmodel |      11.12      |      6.91       |      98.64      |      11.47      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_fp16_1b_2core.bmodel |      5.53       |      6.89       |      19.00      |      11.53      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_int8_1b_2core.bmodel |      5.43       |      6.88       |      8.74       |      11.53      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_int8_4b_2core.bmodel |      5.19       |      6.58       |      5.22       |      11.16      |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_fp32_1b_2core.bmodel |      7.16       |      2.43       |      95.78      |      11.67      |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_fp16_1b_2core.bmodel |      6.78       |      2.43       |      16.20      |      11.72      |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_int8_1b_2core.bmodel |      6.21       |      2.43       |      5.92       |      11.72      |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_int8_4b_2core.bmodel |      5.99       |      2.35       |      3.89       |      11.92      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_fp32_1b_2core.bmodel |      5.81       |      6.83       |      96.29      |      12.05      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_fp16_1b_2core.bmodel |      5.37       |      6.82       |      16.69      |      12.02      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_int8_1b_2core.bmodel |      4.83       |      6.82       |      6.41       |      11.95      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_int8_4b_2core.bmodel |      4.65       |      6.66       |      4.09       |      10.78      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      45.91      |      34.23      |     324.44      |      11.54      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_fp16_1b.bmodel    |      71.27      |      33.46      |      49.85      |      11.56      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      78.83      |      32.81      |      20.13      |      11.57      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      44.62      |      35.96      |      17.73      |      11.44      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      13.58      |      7.24       |     320.40      |      11.52      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp16_1b.bmodel    |      10.34      |      7.25       |      45.68      |      11.69      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      9.94       |      7.26       |      16.01      |      11.62      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      15.90      |      6.96       |      13.84      |      11.17      |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      9.60       |      2.57       |     317.28      |      11.74      |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      10.47      |      2.58       |      42.64      |      11.71      |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      9.99       |      2.57       |      13.00      |      11.63      |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      8.39       |      2.49       |      12.34      |      12.01      |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      8.49       |      7.06       |     317.85      |      12.10      |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      9.08       |      7.06       |      43.18      |      12.02      |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      10.51      |      7.06       |      13.52      |      12.02      |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      8.79       |      6.86       |      12.55      |      10.79      |
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
    parser.add_argument('--bmodel', type=str, default='scrfds_v6.1_3output_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='scrfd_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_scrfds_v6.1_3output_fp32_1b.bmodel_python_test.log')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    args = argsparser()
    benchmark_path = current_dir + "/benchmark.txt"
        
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
        
