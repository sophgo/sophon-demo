import re
import argparse
import math
import os
import sys
import re

baseline = """
|    测试平台  |     测试程序            |        测试模型            |decode_time/crop_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ----------------       | -------------------------- |  --------           | ---------     | ---------     | --------- |
| SE5-16      | ppocr_system_opencv.py | ch_PP-OCRv3_det_fp32.bmodel| 37.96               |  25.51        |  25.08        |  13.04    |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp32.bmodel| 1.67                |  0.60         |  4.11         |  1.37     |
| SE5-16      | ppocr_bmcv.soc         | ch_PP-OCRv3_det_fp32.bmodel| 7.31                |  5.26         |  14.65        |  3.16     |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp32.bmodel| 1.45                |  0.96         |  3.05         |  3.11     |
| SE7-32      | ppocr_system_opencv.py | ch_PP-OCRv3_det_fp32.bmodel| 37.79               |  25.99        |  24.90        |  13.50    |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp32.bmodel| 1.67                |  0.59         |  3.08         |  1.59     |
| SE7-32      | ppocr_system_opencv.py | ch_PP-OCRv3_det_fp16.bmodel| 37.58               |  25.79        |  14.16        |  13.30    |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp16.bmodel| 1.67                |  0.59         |  2.10         |  1.59     |
| SE7-32      | ppocr_bmcv.soc         | ch_PP-OCRv3_det_fp32.bmodel| 6.72                |  1.31         |  13.59        |  3.44     |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp32.bmodel| 2.08                |  0.59         |  1.75         |  3.08     |
| SE7-32      | ppocr_bmcv.soc         | ch_PP-OCRv3_det_fp16.bmodel| 6.72                |  1.30         |  2.78         |  3.53     |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp16.bmodel| 2.00                |  0.38         |  0.64         |  3.08     |
| SE9-16      | ppocr_system_opencv.py | ch_PP-OCRv3_det_fp32.bmodel| 49.03               |  34.46        |  83.09        |  19.21    |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp32.bmodel| 2.40                |  0.83         |  46.75        |  2.03     |
| SE9-16      | ppocr_system_opencv.py | ch_PP-OCRv3_det_fp16.bmodel| 48.06               |  34.10        |  32.76        |  18.87    |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp16.bmodel| 2.42                |  0.82         |  8.96         |  1.92     |
| SE9-16      | ppocr_bmcv.soc         | ch_PP-OCRv3_det_fp32.bmodel| 8.78                |  3.38         |  46.82        |  8.19     |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp32.bmodel| 0.78                |  0.83         |  10.62        |  4.24     |
| SE9-16      | ppocr_bmcv.soc         | ch_PP-OCRv3_det_fp16.bmodel| 8.70                |  3.36         |  11.65        |  8.22     |
|     ^       |         ^              | ch_PP-OCRv3_rec_fp16.bmodel| 0.80                |  0.80         |  2.72         |  4.20     |

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
        
