# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
import argparse
import multiprocessing
import os
import re
import sys

baseline = """
|    测试平台 |     测试程序      |             encoder_bmodel        |        decoder_bmodel             |   decode_time   | preprocess_time | inference_time  | postprocess_time| 
| ----------- | ----------------- | --------------------------------- | --------------------------------- | --------------- | --------------- | --------------  | --------------- |
|   SE9-16    |  sam2_opencv.py   | sam2_encoder_f32_1b_1core.bmodel  | sam2_decoder_f32_1b_1core.bmodel  |      95.91      |     2394.54     |      74.43      |      1.07       |
|   SE9-16    |  sam2_opencv.py   | sam2_encoder_f32_1b_2core.bmodel  | sam2_decoder_f32_1b_2core.bmodel  |      99.32      |     1472.30     |      58.43      |      1.14       |
|   SE9-16    |  sam2_opencv.py   | sam2_encoder_f16_1b_1core.bmodel  | sam2_decoder_f16_1b_1core.bmodel  |      96.10      |     457.77      |      37.05      |      2.77       |
|   SE9-16    |  sam2_opencv.py   | sam2_encoder_f16_1b_2core.bmodel  | sam2_decoder_f16_1b_2core.bmodel  |     100.79      |     311.52      |      34.60      |      1.11       |


"""

table_data = {
    "platform": [],
    "program": [],
    "encoder_bmodel": [],
    "decoder_bmodel": [],
    "preprocess_time": [],
    "encoder_time": [],
    "decoder_time": [],
    "postprocess_time": [],
}


patterns_python = {
    "preprocess_time": re.compile(r"Preprocess time\(ms\): ([\d.]+)"),
    "encoder_time": re.compile(r"Encoder time\(ms\): ([\d.]+)"),
    "decoder_time": re.compile(r"Decoder time\(ms\): ([\d.]+)"),
    "postprocess_time": re.compile(r"Postprocess time\(ms\): ([\d.]+)"),
}


def extract_times(text, patterns):
    results = {}
    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            results[key] = round(float(match.group(1)), 2)
    return results


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument(
        "--target", type=str, default="BM1688", help="path of label json"
    )
    parser.add_argument(
        "--platform", type=str, default="soc", help="path of result json"
    )
    parser.add_argument(
        "--encoder_bmodel",
        type=str,
        default="../models/BM1688/image_encoder/sam2_encoder_f16_1b_2core.bmodel",
        help="Path of encoder bmodel",
    )
    parser.add_argument(
        "--decoder_bmodel",
        type=str,
        default="../models/BM1688/image_decoder/sam2_decoder_f16_1b_2core.bmodel",
        help="Path of decoder bmodel",
    )
    parser.add_argument("--program", type=str, default="sam2_opencv.py")
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument(
        "--input", type=str, default="../log/sam2_opencv_decoder_f16_1b_2core_python_test.log"
    )
    parser.add_argument(
        "--use_cpu_opt",
        action="store_true",
        default=False,
        help="accelerate cpu postprocess",
    )
    return parser.parse_args()


if __name__ == "__main__":
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
        match = re.search(
            r"\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|",
            line,
        )
        if match:
            table_data["platform"].append(match.group(1))
            table_data["program"].append(match.group(2))
            table_data["encoder_bmodel"].append(match.group(3))
            table_data["decoder_bmodel"].append(match.group(4))
            table_data["preprocess_time"].append(float(match.group(5)))
            table_data["encoder_time"].append(float(match.group(6)))
            table_data["decoder_time"].append(float(match.group(7)))
            table_data["postprocess_time"].append(float(match.group(8)))

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
        platform = (
            args.target + " SoC" if args.platform == "soc" else args.target + " PCIe"
        )
    min_width = 17

    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^13}|{:^20}|{:^35}|{:^35}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
                "platform",
                "program",
                "encoder_bmodel",
                "decoder_bmodel",
                "preprocess_time",
                "encoder_time",
                "decoder_time",
                "postprocess_time",
                width=min_width,
            )
            f.write(benchmark_str)

    with open(args.input, "r") as f:
        data = f.read()
    if args.language == "python":
        extracted_data = extract_times(data, patterns_python)
    else:
        print("unsupport code language")
    match_index = -1
    for i in range(0, len(table_data["platform"])):
        if (
            platform == table_data["platform"][i]
            and args.program == table_data["program"][i]
        ):
            match_index = i
            break
    baseline_data = {}
    if match_index == -1:
        print("Unmatched case.")
    else:
        baseline_data["preprocess_time"] = table_data["preprocess_time"][match_index]
        baseline_data["encoder_time"] = table_data["encoder_time"][match_index]
        baseline_data["decoder_time"] = table_data["decoder_time"][match_index]
        baseline_data["postprocess_time"] = table_data["postprocess_time"][match_index]
    for key, statis in baseline_data.items():
        threhold = 0.2
        if key == "encoder_time" and args.program == "sam2_opencv.py":
            threhold = 0.4
        if key == "preprocess_time":
            threhold = 0.4
        if abs(statis - extracted_data[key]) / statis > threhold:
            print("{:} time, diff ratio > {:}".format(key, str(threhold)))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False

    benchmark_str = "|{:^13}|{:^19}|{:^35}|{:^35}|{preprocess_time:^{width}.2f}|{encoder_time:^{width}.2f}|{decoder_time:^{width}.2f}|{postprocess_time:^{width}.2f}|\n".format(
        platform,
        args.program,
        os.path.split(args.encoder_bmodel)[-1],
        os.path.split(args.decoder_bmodel)[-1],
        **extracted_data,
        width=min_width
    )

    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)

    if compare_pass is False:
        sys.exit(1)
