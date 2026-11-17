#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import re
import argparse
import math
import os
import sys

baseline = """
|    测试平台  |     测试程序        |        测试模型       |decode_time|preprocess_time|inference_time|postprocess_time|
| ----------- | ------------------ | --------------------- | -------- | --------- | --------- | --------- |
|   SE9-16    | dinov2_opencv.py   | dinov2_vitb14_reg_fp32_1b.bmodel |     7.680 |    12.350 |      294.500 |     6.000 |
|   SE9-16    | dinov2_opencv.py   | dinov2_vitb14_reg_fp32_1b_2core.bmodel |     7.630 |    12.360 |      194.930 |     5.930 |
|   SE9-16    | dinov2_opencv.py   | dinov2_vitb14_reg_fp16_1b.bmodel |     7.620 |    12.350 |       53.710 |     5.950 |
|   SE9-16    | dinov2_opencv.py   | dinov2_vitb14_reg_fp16_1b_2core.bmodel |     7.650 |    12.360 |       41.720 |     5.940 |
|   SE9-16    | dinov2_opencv.py   | dinov2_vitb14_reg_int8_1b.bmodel |     7.640 |    12.330 |       34.850 |     5.950 |
|   SE9-16    | dinov2_opencv.py   | dinov2_vitb14_reg_int8_1b_2core.bmodel |     7.660 |    12.360 |       29.180 |     5.930 |
|   SE9-16    | dinov2_bmcv.py     | dinov2_vitb14_reg_fp32_1b.bmodel |     6.890 |     1.670 |      294.010 |     5.910 |
|   SE9-16    | dinov2_bmcv.py     | dinov2_vitb14_reg_fp32_1b_2core.bmodel |     3.530 |     1.650 |      194.570 |     5.970 |
|   SE9-16    | dinov2_bmcv.py     | dinov2_vitb14_reg_fp16_1b.bmodel |     3.450 |     1.660 |       53.190 |     5.960 |
|   SE9-16    | dinov2_bmcv.py     | dinov2_vitb14_reg_fp16_1b_2core.bmodel |     3.450 |     1.640 |       41.180 |     5.900 |
|   SE9-16    | dinov2_bmcv.py     | dinov2_vitb14_reg_int8_1b.bmodel |     3.430 |     1.660 |       34.310 |     5.890 |
|   SE9-16    | dinov2_bmcv.py     | dinov2_vitb14_reg_int8_1b_2core.bmodel |     3.440 |     1.650 |       28.650 |     5.890 |
|   SE9-16    | dinov2_bmcv.soc    | dinov2_vitb14_reg_fp32_1b.bmodel |     3.229 |     0.852 |      293.535 |     4.058 |
|   SE9-16    | dinov2_bmcv.soc    | dinov2_vitb14_reg_fp32_1b_2core.bmodel |     3.216 |     0.855 |      193.962 |     4.041 |
|   SE9-16    | dinov2_bmcv.soc    | dinov2_vitb14_reg_fp16_1b.bmodel |     3.218 |     0.850 |       52.786 |     4.040 |
|   SE9-16    | dinov2_bmcv.soc    | dinov2_vitb14_reg_fp16_1b_2core.bmodel |     3.188 |     0.851 |       40.797 |     4.030 |
|   SE9-16    | dinov2_bmcv.soc    | dinov2_vitb14_reg_int8_1b.bmodel |     3.210 |     0.844 |       33.908 |     4.029 |
|   SE9-16    | dinov2_bmcv.soc    | dinov2_vitb14_reg_int8_1b_2core.bmodel |     3.183 |     0.848 |       28.260 |     4.043 |
|   SE7-32    | dinov2_opencv.py   | dinov2_vitb14_reg_fp32_1b.bmodel |     5.890 |     9.280 |       81.900 |     2.240 |
|   SE7-32    | dinov2_opencv.py   | dinov2_vitb14_reg_fp16_1b.bmodel |     5.830 |     9.070 |       19.110 |     1.850 |
|   SE7-32    | dinov2_opencv.py   | dinov2_vitb14_reg_int8_1b.bmodel |     5.820 |     9.170 |       17.540 |     1.860 |
|   SE7-32    | dinov2_bmcv.py     | dinov2_vitb14_reg_fp32_1b.bmodel |     2.830 |     0.770 |       81.450 |     2.060 |
|   SE7-32    | dinov2_bmcv.py     | dinov2_vitb14_reg_fp16_1b.bmodel |     2.640 |     0.760 |       18.690 |     1.940 |
|   SE7-32    | dinov2_bmcv.py     | dinov2_vitb14_reg_int8_1b.bmodel |     2.870 |     0.870 |       17.170 |     2.380 |
|   SE7-32    | dinov2_bmcv.soc    | dinov2_vitb14_reg_fp32_1b.bmodel |     2.365 |     0.350 |       81.175 |     3.265 |
|   SE7-32    | dinov2_bmcv.soc    | dinov2_vitb14_reg_fp16_1b.bmodel |     2.362 |     0.345 |       18.375 |     3.269 |
|   SE7-32    | dinov2_bmcv.soc    | dinov2_vitb14_reg_int8_1b.bmodel |     2.353 |     0.344 |       16.788 |     3.277 |
|   SE9-8     | dinov2_opencv.py   | dinov2_vitb14_reg_fp32_1b.bmodel |     8.690 |    13.280 |      294.380 |     1.750 |
|   SE9-8     | dinov2_opencv.py   | dinov2_vitb14_reg_fp16_1b.bmodel |     8.990 |    13.610 |       54.000 |     2.380 |
|   SE9-8     | dinov2_opencv.py   | dinov2_vitb14_reg_int8_1b.bmodel |     8.790 |    13.600 |       35.100 |     1.970 |
|   SE9-8     | dinov2_bmcv.py     | dinov2_vitb14_reg_fp32_1b.bmodel |     5.440 |     1.780 |      293.830 |     1.790 |
|   SE9-8     | dinov2_bmcv.py     | dinov2_vitb14_reg_fp16_1b.bmodel |     4.240 |     1.830 |       53.590 |     2.160 |
|   SE9-8     | dinov2_bmcv.py     | dinov2_vitb14_reg_int8_1b.bmodel |     4.170 |     1.890 |       34.480 |     1.890 |
|   SE9-8     | dinov2_bmcv.soc    | dinov2_vitb14_reg_fp32_1b.bmodel |     3.363 |     0.859 |      293.319 |     4.100 |
|   SE9-8     | dinov2_bmcv.soc    | dinov2_vitb14_reg_fp16_1b.bmodel |     3.187 |     0.855 |       52.654 |     4.091 |
|   SE9-8     | dinov2_bmcv.soc    | dinov2_vitb14_reg_int8_1b.bmodel |     3.157 |     0.862 |       33.808 |     4.068 |
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
    parser.add_argument('--bmodel', type=str, default='dinov2_vitb14_reg_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='dinov2_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_dinov2_vitb14_reg_fp32_1b.bmodel_python_test.log')
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
        print("Unmatched case: no baseline for platform={} program={} bmodel={}, skip comparison.".format(
            platform, args.program, args.bmodel))
    else:
        baseline_data["decode"] = table_data["decode"][match_index]
        baseline_data["preprocess"] = table_data["preprocess"][match_index]
        baseline_data["inference"] = table_data["inference"][match_index]
        baseline_data["postprocess"] = table_data["postprocess"][match_index]
    for key, statis in baseline_data.items():
        if key not in extracted_data:
            print("{:} time not found in log, cannot compare.".format(key))
            compare_pass = False
            continue
        if statis < 1:
            if abs(statis - extracted_data[key]) > 0.5:
                print("{:} time, diff > 0.5".format(key))
                print("Baseline is:", statis)
                print("Now is: ", extracted_data[key])
                compare_pass = False
        elif abs(statis - extracted_data[key]) / statis > 0.4:
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
