import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
|  platform   |      program      |              bmodel               |   decode_time   | preprocess_time | inference_time  |postprocess_time |
|   SE5-16    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      35.41      |      24.20      |      25.04      |      8.57       |
|   SE5-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      30.14      |      24.38      |      21.21      |      8.40       |
|   SE5-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      30.24      |      26.42      |      8.70       |      8.33       |
|   SE5-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.57       |      3.78       |      21.97      |      8.68       |
|   SE5-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      3.56       |      3.78       |      18.13      |      8.51       |
|   SE5-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      3.38       |      3.63       |      5.97       |      8.43       |
|   SE5-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      4.41       |      0.97       |      20.06      |      6.42       |
|   SE5-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      4.40       |      0.97       |      16.24      |      6.39       |
|   SE5-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      4.20       |      0.91       |      5.03       |      6.93       |
|   SE5-16    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.26       |      3.92       |      20.37      |      24.64      |
|   SE5-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      3.21       |      3.92       |      16.55      |      24.65      |
|   SE5-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      3.03       |      3.67       |      5.13       |      24.95      |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      29.94      |      25.92      |      40.39      |      8.57       |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_fp16_1b.bmodel    |      30.02      |      25.67      |      9.38       |      8.67       |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      30.01      |      25.65      |      8.18       |      8.68       |
|   SE7-32    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      30.18      |      27.54      |      6.70       |      8.43       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.02       |      2.97       |      36.84      |      8.75       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp16_1b.bmodel    |      3.04       |      2.96       |      5.81       |      9.06       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      3.02       |      2.97       |      4.66       |      8.85       |
|   SE7-32    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      2.86       |      2.80       |      3.56       |      8.46       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      3.88       |      0.87       |      34.84      |      6.39       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      3.90       |      0.87       |      3.80       |      6.37       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      3.90       |      0.87       |      2.65       |      6.39       |
|   SE7-32    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      3.71       |      0.84       |      2.53       |      6.84       |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      2.69       |      3.15       |      35.15      |      24.78      |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      2.70       |      3.16       |      4.14       |      25.66      |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      2.69       |      3.16       |      2.98       |      25.04      |
|   SE7-32    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      2.55       |      3.10       |      2.70       |      25.02      |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      53.97      |      46.81      |     181.32      |      13.20      |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_fp16_1b.bmodel    |      54.74      |      46.58      |      43.43      |      14.68      |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      56.00      |      45.49      |      28.14      |      15.90      |
|   SE9-16    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      51.04      |      48.04      |      16.96      |      16.18      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      12.16      |      10.85      |     176.44      |      13.21      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp16_1b.bmodel    |      17.06      |      10.94      |      37.52      |      15.35      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      17.75      |      11.15      |      22.63      |      16.77      |
|   SE9-16    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      12.58      |      11.73      |      12.39      |      17.71      |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      11.99      |      3.08       |     171.05      |      9.60       |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      20.58      |      3.51       |      32.10      |      10.17      |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      19.33      |      3.61       |      16.78      |      10.93      |
|   SE9-16    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      14.92      |      3.65       |      9.49       |      12.73      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      13.56      |      10.02      |     171.81      |      39.02      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      17.07      |      10.41      |      34.34      |      42.58      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      17.73      |      10.36      |      18.59      |      43.38      |
|   SE9-16    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      8.59       |      6.86       |      6.96       |      35.79      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_fp32_1b_2core.bmodel |      54.07      |      46.95      |     114.79      |      13.30      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_fp16_1b_2core.bmodel |      56.41      |      45.42      |      35.99      |      15.72      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_int8_1b_2core.bmodel |      56.86      |      47.33      |      27.13      |      15.78      |
|   SE9-16    |  scrfd_opencv.py  |scrfd_10g_kps_int8_4b_2core.bmodel |      51.58      |      48.75      |      14.39      |      16.95      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_fp32_1b_2core.bmodel |      13.62      |      10.92      |     109.40      |      13.14      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_fp16_1b_2core.bmodel |      18.71      |      11.05      |      30.31      |      16.30      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_int8_1b_2core.bmodel |      19.53      |      11.38      |      22.17      |      16.47      |
|   SE9-16    |   scrfd_bmcv.py   |scrfd_10g_kps_int8_4b_2core.bmodel |      12.79      |      11.88      |      9.80       |      18.39      |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_fp32_1b_2core.bmodel |      14.26      |      3.21       |     103.61      |      9.63       |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_fp16_1b_2core.bmodel |      20.94      |      3.55       |      25.26      |      10.72      |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_int8_1b_2core.bmodel |      20.70      |      3.55       |      16.02      |      11.29      |
|   SE9-16    |  scrfd_bmcv.soc   |scrfd_10g_kps_int8_4b_2core.bmodel |      15.83      |      3.62       |      6.71       |      13.23      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_fp32_1b_2core.bmodel |      14.32      |      9.95       |     105.04      |      38.02      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_fp16_1b_2core.bmodel |      19.97      |      10.43      |      27.11      |      43.41      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_int8_1b_2core.bmodel |      20.05      |      10.37      |      17.78      |      43.43      |
|   SE9-16    |  scrfd_sail.soc   |scrfd_10g_kps_int8_4b_2core.bmodel |      10.92      |      6.85       |      4.10       |      35.62      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_fp32_1b.bmodel    |      50.64      |      33.73      |     324.48      |      11.60      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_fp16_1b.bmodel    |      68.15      |      32.86      |      49.81      |      11.71      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_1b.bmodel    |      72.49      |      33.02      |      20.37      |      11.67      |
|    SE9-8    |  scrfd_opencv.py  |   scrfd_10g_kps_int8_4b.bmodel    |      44.86      |      36.68      |      17.74      |      11.29      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp32_1b.bmodel    |      12.48      |      7.23       |     320.43      |      11.63      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_fp16_1b.bmodel    |      9.13       |      7.24       |      45.62      |      11.75      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_1b.bmodel    |      11.81      |      7.27       |      15.96      |      11.90      |
|    SE9-8    |   scrfd_bmcv.py   |   scrfd_10g_kps_int8_4b.bmodel    |      12.47      |      6.93       |      13.77      |      11.41      |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      11.96      |      2.58       |     317.28      |      9.02       |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      11.02      |      2.58       |      42.64      |      9.04       |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      11.93      |      2.57       |      13.00      |      8.95       |
|    SE9-8    |  scrfd_bmcv.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      8.88       |      2.49       |      12.33      |      9.64       |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_fp32_1b.bmodel    |      10.24      |      7.08       |     317.86      |      36.12      |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_fp16_1b.bmodel    |      10.47      |      7.06       |      43.17      |      36.09      |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_1b.bmodel    |      9.85       |      7.06       |      13.51      |      35.63      |
|    SE9-8    |  scrfd_sail.soc   |   scrfd_10g_kps_int8_4b.bmodel    |      8.94       |      6.86       |      12.55      |      35.29      |
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
        
