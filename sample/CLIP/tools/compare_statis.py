import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
| 测试平台 | 测试程序            | image_encode模型                        | text_encode模型                        | Preprocess_Time | Image_Encoding_Time | Text_Encoding_Time |
| -------- | ------------------- | --------------------------------------- | -------------------------------------- | --------------- | ------------------- | ------------------ |
| SE7-32   | zeroshot_predict.py | clip_image_vitb32_bm1684x_f16_1b.bmodel | clip_text_vitb32_bm1684x_f16_1b.bmodel | 12.17           | 9.63                | 18.90              |
| SE9-16   | zeroshot_predict.py | clip_image_vitb32_bm1688_f16_1b.bmodel  | clip_text_vitb32_bm1688_f16_1b.bmodel  | 16.92           | 25.04               | 49.61              |
| SE9-8    | zeroshot_predict.py | clip_image_vitb32_cv186x_f16_1b.bmodel  | clip_text_vitb32_cv186x_f16_1b.bmodel  | 17.09           | 30.59               | 59.56              |
"""

table_data = {
    "platform": [],
    "program": [],
    "image_model": [],
    "text_model": [],
    "Preprocess_Time": [],
    "Text_Encoding_Time": [],
    "Image_Encoding_Time": []
}

for line in baseline.strip().split("\n")[2:]:
    match = re.search(r'\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', line)
    if match:
        table_data["platform"].append(match.group(1))
        table_data["program"].append(match.group(2))
        table_data["image_model"].append(match.group(3))
        table_data["text_model"].append(match.group(4))
        table_data["Preprocess_Time"].append(float(match.group(5)))
        table_data["Image_Encoding_Time"].append(float(match.group(6)))
        table_data["Text_Encoding_Time"].append(float(match.group(7)))
        

patterns_python = {
    'Preprocess_Time': re.compile(r'preprocess\(ms\): ([\d.]+)'),
    'Text_Encoding_Time': re.compile(r'text_encode\(ms\): ([\d.]+)'),
    'Image_Encoding_Time': re.compile(r'image_encode\(ms\): ([\d.]+)'),
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
    parser.add_argument('--program', type=str, default='yolov7_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov7_v0.1_3output_fp32_1b.bmodel_python_test.log')
    parser.add_argument('--image_model', type=str, default='clip_image_vitb32_bm1684x_f16_1b.bmodel')
    parser.add_argument('--text_model', type=str, default='clip_text_vitb32_bm1684x_f16_1b.bmodel')
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
           "platform", "program", "image_model", "text_model", "Preprocess_Time", "Image_Encoding_Time", "Text_Encoding_Time", width=min_width)
            f.write(benchmark_str)
    
    with open(args.input, "r") as f:
        data = f.read()
    if args.language == "python":    
        extracted_data = extract_times(data, patterns_python)
    else:
        print("unsupport code language")

    match_index = -1
    for i in range(0, len(table_data["platform"])):
        if platform == table_data["platform"][i] and args.program == table_data["program"][i] and args.image_model == table_data["image_model"][i] and args.text_model == table_data["text_model"][i]:
            match_index = i
            break
    baseline_data = {}
    if match_index == -1:
        print("Unmatched case.")
    else:
        baseline_data["Preprocess_Time"] = table_data["Preprocess_Time"][match_index]
        baseline_data["Image_Encoding_Time"] = table_data["Image_Encoding_Time"][match_index]
        baseline_data["Text_Encoding_Time"] = table_data["Text_Encoding_Time"][match_index]
    for key, statis in baseline_data.items():
        if abs(statis - extracted_data[key]) / statis > 0.4:
            print("{:} time, diff ratio > 0.4".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^35}|{:^35}|{Preprocess_Time:^{width}.2f}|{Image_Encoding_Time:^{width}.2f}|{Text_Encoding_Time:^{width}.2f}|\n".format(
                     platform, args.program, args.image_model, args.text_model, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
