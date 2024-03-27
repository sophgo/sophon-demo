import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
|    测试平台  |     测试程序      |             测试模型                |decode_time    |preprocess_time  |inference_time   |postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | --------      | ---------       | ---------        | --------- |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      15.08      |      21.95      |      31.40      |     107.61      |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      15.07      |      26.16      |      34.45      |     110.48      |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      15.03      |      23.94      |      27.53      |     111.78      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.61       |      2.83       |      29.06      |     106.85      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      3.59       |      2.31       |      17.92      |     106.40      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      3.44       |      2.13       |      11.82      |     110.71      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.87       |      1.54       |      22.33      |      15.68      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      4.85       |      1.53       |      11.20      |      15.66      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      4.75       |      1.47       |      6.03       |      15.64      |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.23       |      3.04       |      23.31      |      14.07      |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.22       |      1.80       |      12.21      |      13.93      |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.08       |      1.71       |      6.88       |      13.80      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      15.09      |      27.82      |      33.27      |     108.98      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      15.01      |      27.27      |      19.10      |     109.18      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      15.08      |      27.02      |      15.18      |     109.33      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      14.99      |      25.01      |      13.31      |     108.20      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.09       |      2.35       |      28.98      |     103.87      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.09       |      2.34       |      14.75      |     103.75      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      3.08       |      2.34       |      10.92      |     103.89      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      2.93       |      2.16       |      9.82       |     108.36      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.32       |      0.74       |      21.63      |      15.91      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.32       |      0.74       |      7.38       |      15.94      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      4.33       |      0.74       |      3.48       |      15.94      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      4.17       |      0.71       |      3.32       |      15.73      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      2.71       |      2.58       |      22.61      |      14.15      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      2.71       |      2.59       |      8.35       |      14.19      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      2.70       |      2.59       |      4.45       |      14.18      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      2.56       |      2.50       |      4.20       |      14.06      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      23.43      |      35.69      |     112.71      |     151.54      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      19.48      |      36.12      |      42.15      |     149.94      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      19.24      |      34.85      |      21.46      |     148.00      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      19.21      |      33.15      |      19.40      |     150.54      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.36       |      5.04       |     107.62      |     143.80      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.35       |      5.06       |      36.97      |     143.51      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      4.36       |      5.04       |      16.47      |     143.20      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      4.22       |      4.74       |      14.96      |     149.54      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.69       |      1.88       |      97.91      |      22.33      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      5.80       |      1.88       |      27.36      |      22.33      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      5.75       |      1.88       |      7.12       |      22.40      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      5.57       |      1.79       |      7.02       |      22.07      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.76       |      5.06       |     100.46      |      19.93      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.81       |      5.05       |      29.87      |      19.84      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.74       |      5.04       |      9.60       |      19.91      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.69       |      4.82       |      9.32       |      19.73      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      19.35      |      35.64      |      78.13      |     150.70      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      19.32      |      35.89      |      32.72      |     150.59      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      19.22      |      36.24      |      21.08      |     148.49      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      19.28      |      32.70      |      17.31      |     150.59      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      4.36       |      5.05       |      72.89      |     143.53      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      4.38       |      5.08       |      27.64      |     143.77      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      4.38       |      5.06       |      15.90      |     143.50      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      4.21       |      4.74       |      13.45      |     149.79      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      5.79       |      1.87       |      63.28      |      22.35      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      5.78       |      1.88       |      18.07      |      22.35      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      5.79       |      1.88       |      6.32       |      22.37      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      5.61       |      1.79       |      4.99       |      22.12      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      3.83       |      5.06       |      65.81      |      19.84      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      3.80       |      5.05       |      20.57      |      19.86      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      3.79       |      5.05       |      8.81       |      19.88      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      3.61       |      4.81       |      7.27       |      19.80      |
"""
baseline_cpu_opt = """
|    测试平台  |     测试程序      |             测试模型                |decode_time    |preprocess_time  |inference_time   |postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | --------      | ---------       | ---------        | ---------      |
|   SE5-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      14.99      |      21.52      |      43.84      |      16.83      |
|   SE5-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.60       |      2.85       |      24.29      |      16.87      |
|   SE5-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.88       |      1.54       |      22.33      |      6.17       |
|   SE5-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.23       |      3.03       |      23.31      |      4.49       |
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
    parser.add_argument('--bmodel', type=str, default='yolov5s_v6.1_3output_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov5_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov5s_v6.1_3output_fp32_1b.bmodel_python_test.log')
    parser.add_argument('--use_cpu_opt', action="store_true", default=False, help='accelerate cpu postprocess')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    args = argsparser()
    if args.use_cpu_opt:
        benchmark_path = current_dir + "/benchmark_cpu_opt.txt"
        baseline = baseline_cpu_opt
    else:
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
        
