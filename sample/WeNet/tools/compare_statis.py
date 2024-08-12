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
import multiprocessing

baseline = """
|    测试平台  |  测试程序 |             测试模型                                   |preprocess_time|encoder_inference_time|decoder_inference_time|postprocess_time| 
| ----------- | --------- | ----------------------------------------------------- | ------------- | -------------------- | -------------------- | ----------------- |
|   SE5-16    |     wenet.py      |                 wenet_encoder_streaming_fp32.bmodel                  |      4.02       |      46.34      |      none       |      8.70       |
|   SE5-16    |     wenet.py      |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      3.96       |      46.11      |     186.68      |      10.14      |
|   SE5-16    |     wenet.soc     |                 wenet_encoder_streaming_fp32.bmodel                  |      88.20      |      38.71      |      none       |      1.12       |
|   SE5-16    |     wenet.soc     |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      88.95      |      38.72      |     186.51      |      1.29       |
|   SE5-16    |     wenet.py      |               wenet_encoder_non_streaming_fp32.bmodel                |      3.35       |      30.70      |      none       |      1.53       |
|   SE5-16    |     wenet.py      | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      3.11       |      30.77      |     186.99      |      3.34       |
|   SE5-16    |     wenet.soc     |               wenet_encoder_non_streaming_fp32.bmodel                |      88.09      |      29.60      |      none       |      0.36       |
|   SE5-16    |     wenet.soc     | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      88.92      |      29.62      |     186.58      |      0.46       |
|   SE7-32    |     wenet.py      |                 wenet_encoder_streaming_fp32.bmodel                  |      3.32       |      23.69      |      none       |      8.69       |
|   SE7-32    |     wenet.py      |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      3.24       |      23.70      |      66.98      |      10.34      |
|   SE7-32    |     wenet.soc     |                 wenet_encoder_streaming_fp32.bmodel                  |      25.73      |      15.64      |      none       |      0.99       |
|   SE7-32    |     wenet.soc     |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      26.43      |      15.69      |      66.64      |      1.09       |
|   SE7-32    |     wenet.py      |                 wenet_encoder_streaming_fp16.bmodel                  |      3.70       |      13.00      |      none       |      8.64       |
|   SE7-32    |     wenet.py      |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      3.86       |      13.07      |      13.89      |      10.54      |
|   SE7-32    |     wenet.soc     |                 wenet_encoder_streaming_fp16.bmodel                  |      25.73      |      5.13       |      none       |      1.00       |
|   SE7-32    |     wenet.soc     |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      26.43      |      5.10       |      13.55      |      1.07       |
|   SE7-32    |     wenet.py      |               wenet_encoder_non_streaming_fp32.bmodel                |      3.04       |      16.72      |      none       |      1.57       |
|   SE7-32    |     wenet.py      | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      3.14       |      16.71      |      66.97      |      3.39       |
|   SE7-32    |     wenet.soc     |               wenet_encoder_non_streaming_fp32.bmodel                |      25.77      |      15.53      |      none       |      0.32       |
|   SE7-32    |     wenet.soc     | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      26.24      |      15.53      |      66.63      |      0.40       |
|   SE7-32    |     wenet.py      |               wenet_encoder_non_streaming_fp16.bmodel                |      3.07       |      4.19       |      none       |      1.61       |
|   SE7-32    |     wenet.py      | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      3.29       |      4.19       |      13.87      |      3.30       |
|   SE7-32    |     wenet.soc     |               wenet_encoder_non_streaming_fp16.bmodel                |      25.76      |      3.03       |      none       |      0.51       |
|   SE7-32    |     wenet.soc     | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      26.50      |      3.02       |      13.55      |      0.59       |
|   SE9-16    |     wenet.py      |                 wenet_encoder_streaming_fp32.bmodel                  |      5.25       |      42.66      |      none       |      12.07      |
|   SE9-16    |     wenet.py      |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      5.69       |      42.44      |     157.15      |      14.48      |
|   SE9-16    |     wenet.soc     |                 wenet_encoder_streaming_fp32.bmodel                  |      35.87      |      32.08      |      none       |      1.82       |
|   SE9-16    |     wenet.soc     |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      36.90      |      32.09      |     156.63      |      1.99       |
|   SE9-16    |     wenet.py      |                 wenet_encoder_streaming_fp16.bmodel                  |      4.55       |      20.89      |      none       |      12.10      |
|   SE9-16    |     wenet.py      |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      6.04       |      20.85      |      39.46      |      14.38      |
|   SE9-16    |     wenet.soc     |                 wenet_encoder_streaming_fp16.bmodel                  |      35.87      |      10.46      |      none       |      1.55       |
|   SE9-16    |     wenet.soc     |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      36.83      |      10.48      |      39.01      |      1.69       |
|   SE9-16    |     wenet.py      |               wenet_encoder_non_streaming_fp32.bmodel                |      5.59       |      47.30      |      none       |      2.19       |
|   SE9-16    |     wenet.py      | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      5.32       |      47.24      |     157.15      |      4.68       |
|   SE9-16    |     wenet.soc     |               wenet_encoder_non_streaming_fp32.bmodel                |      35.97      |      45.74      |      none       |      0.62       |
|   SE9-16    |     wenet.soc     | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      36.77      |      45.74      |     156.65      |      0.75       |
|   SE9-16    |     wenet.py      |               wenet_encoder_non_streaming_fp16.bmodel                |      4.47       |      11.12      |      none       |      2.17       |
|   SE9-16    |     wenet.py      | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      4.38       |      11.03      |      39.48      |      4.63       |
|   SE9-16    |     wenet.soc     |               wenet_encoder_non_streaming_fp16.bmodel                |      35.88      |      9.53       |      none       |      0.49       |
|   SE9-16    |     wenet.soc     | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      36.65      |      9.53       |      39.01      |      0.61       |
|    SE9-8    |     wenet.py      |                 wenet_encoder_streaming_fp32.bmodel                  |      5.81       |      42.55      |      none       |      11.77      |
|    SE9-8    |     wenet.py      |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      5.46       |      42.47      |     157.13      |      14.16      |
|    SE9-8    |     wenet.soc     |                 wenet_encoder_streaming_fp32.bmodel                  |      35.94      |      32.17      |      none       |      1.72       |
|    SE9-8    |     wenet.soc     |   wenet_encoder_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel    |      36.96      |      32.21      |     156.64      |      1.88       |
|    SE9-8    |     wenet.py      |                 wenet_encoder_streaming_fp16.bmodel                  |      6.50       |      20.65      |      none       |      12.07      |
|    SE9-8    |     wenet.py      |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      5.54       |      20.66      |      38.89      |      14.59      |
|    SE9-8    |     wenet.soc     |                 wenet_encoder_streaming_fp16.bmodel                  |      35.94      |      10.42      |      none       |      1.54       |
|    SE9-8    |     wenet.soc     |   wenet_encoder_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel    |      36.77      |      10.44      |      38.42      |      1.65       |
|    SE9-8    |     wenet.py      |               wenet_encoder_non_streaming_fp32.bmodel                |      5.31       |      47.39      |      none       |      2.16       |
|    SE9-8    |     wenet.py      | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      5.19       |      47.38      |     157.14      |      4.59       |
|    SE9-8    |     wenet.soc     |               wenet_encoder_non_streaming_fp32.bmodel                |      35.90      |      45.87      |      none       |      0.62       |
|    SE9-8    |     wenet.soc     | wenet_encoder_non_streaming_fp32.bmodel + wenet_decoder_fp32.bmodel  |      36.77      |      45.87      |     156.64      |      0.76       |
|    SE9-8    |     wenet.py      |               wenet_encoder_non_streaming_fp16.bmodel                |      4.43       |      10.97      |      none       |      2.13       |
|    SE9-8    |     wenet.py      | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      4.40       |      10.97      |      38.89      |      4.68       |
|    SE9-8    |     wenet.soc     |               wenet_encoder_non_streaming_fp16.bmodel                |      35.91      |      9.48       |      none       |      0.49       |
|    SE9-8    |     wenet.soc     | wenet_encoder_non_streaming_fp16.bmodel + wenet_decoder_fp16.bmodel  |      36.67      |      9.49       |      38.42      |      0.63       |

"""

table_data = {
    "platform": [],
    "program": [],
    "bmodel": [],
    "preprocess": [],
    "encoder_inference": [],
    "decoder_inference": [],
    "postprocess": []
}

patterns_cpp = {
    'preprocess': re.compile(r'\[.*preprocess*\]  loops:.*avg: ([\d.]+) ms'),
    'encoder_inference': re.compile(r'\[.*encoder inference.*\]  loops:.*avg: ([\d.]+) ms'),
    'decoder_inference': re.compile(r'\[.*decoder inference.*\]  loops:.*avg: ([\d.]+) ms'),
    'postprocess': re.compile(r'\[.*postprocess.*\]  loops:.*avg: ([\d.]+) ms'),
}

patterns_python = {
    'preprocess': re.compile(r'preprocess_time\(ms\): ([\d.]+)'),
    'encoder_inference': re.compile(r'encoder_inference_time\(ms\): ([\d.]+)'),
    'decoder_inference': re.compile(r'decoder_inference_time\(ms\): ([\d.]+)'),
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
    parser.add_argument('--target', type=str, default='BM1684X', help='')
    parser.add_argument('--platform', type=str, default='soc', help='')
    parser.add_argument('--encoder_bmodel', type=str, default='wenet_encoder_streaming_fp32.bmodel')
    parser.add_argument('--decoder_bmodel', type=str, default='wenet_decoder_fp32.bmodel')
    parser.add_argument('--program', type=str, default='wenet.soc')
    parser.add_argument('--language', type=str, default='cpp')
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--mode', type=str, default='ctc_prefix_beam_search')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    args = argsparser()
    benchmark_path = current_dir + "/benchmark.txt"
        
    for line in baseline.strip().split("\n")[2:]:
        match = re.search(r'\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*([^|]+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', line)
        if match:
            table_data["platform"].append(match.group(1))
            table_data["program"].append(match.group(2))
            table_data["bmodel"].append(match.group(3))
            table_data["preprocess"].append(match.group(4))
            table_data["encoder_inference"].append(float(match.group(5)))
            if match.group(6) == "none":
                table_data["decoder_inference"].append(match.group(6))
            else:
                table_data["decoder_inference"].append(float(match.group(6)))
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
            benchmark_str = "|{:^13}|{:^19}|{:^70}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
           "platform", "program", "bmodel", "preprocess", "encoder_inference", "decoder_inference", "postprocess", width=min_width)
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
    bmodel = args.encoder_bmodel
    if args.mode == "attention_rescoring":
        bmodel = args.encoder_bmodel + " + " + args.decoder_bmodel
    for i in range(0, len(table_data["platform"])):
        if platform == table_data["platform"][i] and args.program == table_data["program"][i] and bmodel == table_data["bmodel"][i]:
            match_index = i
            break
    baseline_data = {}
    if match_index == -1:
        print("Unmatched case.")
    else:
        baseline_data["preprocess"] = table_data["preprocess"][match_index]
        baseline_data["encoder_inference"] = table_data["encoder_inference"][match_index]
        baseline_data["decoder_inference"] = table_data["decoder_inference"][match_index]
        baseline_data["postprocess"] = table_data["postprocess"][match_index]
        
    for key, statis in baseline_data.items():
        if abs(statis - extracted_data[key]) / statis > 0.4:
            print("{:} time, diff ratio > 0.4".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
    
    if args.mode != "attention_rescoring":
        extracted_data["decoder_inference"] = "none"
        benchmark_str = "|{:^13}|{:^19}|{:^70}|{preprocess:^{width}.2f}|{encoder_inference:^{width}.2f}|{decoder_inference:^{width}}|{postprocess:^{width}.2f}|\n".format(
                        platform, args.program, bmodel, **extracted_data, width=min_width)
    else:
        benchmark_str = "|{:^13}|{:^19}|{:^70}|{preprocess:^{width}.2f}|{encoder_inference:^{width}.2f}|{decoder_inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                        platform, args.program, bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)
        
    if compare_pass == False:
        sys.exit(1)
        
