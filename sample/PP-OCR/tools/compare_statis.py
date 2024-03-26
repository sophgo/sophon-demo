import re
import argparse
import math
import os
import sys
import re

baseline = """
|    测试平台  |     测试程序             |            测试模型               |decode_time/crop_time|    preprocess_time  |inference_time      |  postprocess_time    | 
| ----------- | ----------------        |     --------------------------    |  --------           | ---------           | ---------           |  ---------          |
| SE5-16      | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp32.bmodel    |        37.96        |        25.51        |        25.08        |        13.04        |
|     ^       |         ^               |    ch_PP-OCRv3_rec_fp32.bmodel    |        1.67         |        0.60         |        4.11         |        1.37         |
| SE5-16      | ppocr_bmcv.soc          |    ch_PP-OCRv3_det_fp32.bmodel    |        7.31         |        5.26         |        14.65        |        3.16         |
|     ^       |         ^               |    ch_PP-OCRv3_rec_fp32.bmodel    |        1.45         |        0.96         |        3.05         |        3.11         |
|   SE7-32    | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp32.bmodel    |        33.78        |        25.84        |        25.66        |        13.26        |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp32.bmodel    |        1.66         |        0.59         |        3.22         |        1.60         |
|   SE7-32    | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp16.bmodel    |        33.79        |        26.05        |        14.32        |        13.17        |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp16.bmodel    |        1.67         |        0.60         |        2.18         |        1.60         |
|   SE7-32    |     ppocr_bmcv.soc      |    ch_PP-OCRv3_det_fp32.bmodel    |        5.66         |        1.32         |        14.21        |        3.38         |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp32.bmodel    |        0.67         |        0.18         |        1.74         |        2.97         |
|   SE7-32    |     ppocr_bmcv.soc      |    ch_PP-OCRv3_det_fp16.bmodel    |        5.66         |        1.32         |        2.87         |        3.38         |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp16.bmodel    |        0.67         |        0.18         |        0.61         |        2.98         |
|   SE9-16    | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp32.bmodel    |        53.49        |        33.75        |        60.79        |        18.13        |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp32.bmodel    |        2.31         |        0.81         |        11.78        |        1.92         |
|   SE9-16    | ppocr_system_opencv.py  |    ch_PP-OCRv3_det_fp16.bmodel    |        53.15        |        33.64        |        26.31        |        18.06        |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp16.bmodel    |        2.32         |        0.81         |        4.47         |        1.92         |
|   SE9-16    |     ppocr_bmcv.soc      |    ch_PP-OCRv3_det_fp32.bmodel    |        16.04        |        3.22         |        46.75        |        4.72         |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp32.bmodel    |        0.80         |        0.61         |        10.42        |        4.16         |
|   SE9-16    |     ppocr_bmcv.soc      |    ch_PP-OCRv3_det_fp16.bmodel    |        16.58        |        3.22         |        11.64        |        4.75         |
|      ^      |            ^            |    ch_PP-OCRv3_rec_fp16.bmodel    |        0.80         |        0.61         |        2.62         |        4.18         |
|   SE9-16    | ppocr_system_opencv.py  | ch_PP-OCRv3_det_fp32_2core.bmodel |        52.67        |        33.78        |        40.89        |        18.09        |
|      ^      |            ^            | ch_PP-OCRv3_rec_fp32_2core.bmodel |        2.32         |        0.80         |        8.01         |        1.92         |
|   SE9-16    | ppocr_system_opencv.py  | ch_PP-OCRv3_det_fp16_2core.bmodel |        53.29        |        33.72        |        22.14        |        18.04        |
|      ^      |            ^            | ch_PP-OCRv3_rec_fp16_2core.bmodel |        2.31         |        0.81         |        3.94         |        1.92         |
|   SE9-16    |     ppocr_bmcv.soc      | ch_PP-OCRv3_det_fp32_2core.bmodel |        16.63        |        3.22         |        25.97        |        4.92         |
|      ^      |            ^            | ch_PP-OCRv3_rec_fp32_2core.bmodel |        0.80         |        0.60         |        6.84         |        4.17         |
|   SE9-16    |     ppocr_bmcv.soc      | ch_PP-OCRv3_det_fp16_2core.bmodel |        17.66        |        3.22         |        7.61         |        4.76         |
|      ^      |            ^            | ch_PP-OCRv3_rec_fp16_2core.bmodel |        0.80         |        0.60         |        2.15         |        4.18         |
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

patterns_cpp_det = {
    'decode': re.compile(r'\[.*decode time.*\]  loops:.*avg: ([\d.]+) ms'),
    'preprocess': re.compile(r'\[.*Det preprocess.*\]  loops:.*avg: ([\d.]+) ms'),
    'inference': re.compile(r'\[.*Det inference.*\]  loops:.*avg: ([\d.]+) ms'),
    'postprocess': re.compile(r'\[.*Det postprocess.*\]  loops:.*avg: ([\d.]+) ms'),
}
patterns_cpp_rec = {
    'crop': re.compile(r'\[.*get crop time.*\]  loops:.*avg: ([\d.]+) ms'),
    'preprocess': re.compile(r'\[.*Rec preprocess.*\]  loops:.*avg: ([\d.]+) ms'),
    'inference': re.compile(r'\[.*Rec inference.*\]  loops:.*avg: ([\d.]+) ms'),
    'postprocess': re.compile(r'\[.*Rec postprocess.*\]  loops:.*avg: ([\d.]+) ms'),
}
patterns_python = r"INFO:root:(\w+)_time\(ms\): (\d+\.\d+)"

def extract_times(text, patterns):
    results = {}
    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            results[key] = round(float(match.group(1)),2)
    return results

def compare(baseline_data, extracted_data, name_str="det"):
    for key, statis in baseline_data.items():
        if abs(statis - extracted_data[key]) / statis > 0.2:
            print("{:} {:} time, diff ratio > 0.2".format(name_str, key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--target', type=str, default='BM1684X', help='path of label json')
    parser.add_argument('--platform', type=str, default='soc', help='path of result json')
    parser.add_argument('--bmodel_det', type=str, default='ch_PP-OCRv3_det_fp32.bmodel')
    parser.add_argument('--bmodel_rec', type=str, default='ch_PP-OCRv3_rec_fp32.bmodel')
    parser.add_argument('--program', type=str, default='ppocr_system_opencv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='')
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
    else:
        platform = args.target + " SoC" if args.platform == "soc" else args.target + " PCIe"
    min_width = 21
    
    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^13}|{:^25}|{:^35}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
           "platform", "program", "bmodel", "decode_time/crop_time", "preprocess_time", "inference_time", "postprocess_time", width=min_width)
            f.write(benchmark_str)
            
    with open(args.input, "r") as f:
        data = f.read()
        
    if args.language == "python":    
        matches = re.findall(patterns_python, data)
        extracted_data_det = dict((name, float(value)) for name, value in matches[:4])
        extracted_data_rec = dict((name, float(value)) for name, value in matches[4:])
    elif args.language == "cpp":
        extracted_data_det = extract_times(data, patterns_cpp_det)
        extracted_data_rec = extract_times(data, patterns_cpp_rec)
    else:
        print("unsupport code language")
    match_index = -1
    for i in range(0, len(table_data["platform"]), 2):
        if platform == table_data["platform"][i] and args.program == table_data["program"][i] \
        and args.bmodel_det == table_data["bmodel"][i] and args.bmodel_rec == table_data["bmodel"][i+1]:
            match_index = i
            break
    baseline_data_det = {}
    baseline_data_rec = {}
    if match_index == -1:
        print("Unmatched case.")
    else:
        baseline_data_det["decode"] = table_data["decode"][match_index]
        baseline_data_det["preprocess"] = table_data["preprocess"][match_index]
        baseline_data_det["inference"] = table_data["inference"][match_index]
        baseline_data_det["postprocess"] = table_data["postprocess"][match_index]
        baseline_data_rec["crop"] = table_data["decode"][match_index + 1]
        baseline_data_rec["preprocess"] = table_data["preprocess"][match_index + 1]
        baseline_data_rec["inference"] = table_data["inference"][match_index + 1]
        baseline_data_rec["postprocess"] = table_data["postprocess"][match_index + 1]
        
    compare(baseline_data_det, extracted_data_det)
    compare(baseline_data_rec, extracted_data_rec)

    benchmark_str = "|{:^13}|{:^25}|{:^35}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel_det, **extracted_data_det, width=min_width)
    benchmark_str += "|{:^13}|{:^25}|{:^35}|{crop:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     "^", "^", args.bmodel_rec, **extracted_data_rec, width=min_width)
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
