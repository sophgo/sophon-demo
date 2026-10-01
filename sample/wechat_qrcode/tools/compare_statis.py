#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# Parse the benchmark output of a wechat_qrcode program (C++ / Python, any of
# the 4 variants) and compare end-to-end FPS against a hardcoded baseline.
#
# Every variant's bench mode prints the same block:
#   e2e    : <ms> ms/img, <fps> fps
#   detect : <ms> ms/img (<n> calls, TPU)
#   sr     : <ms> ms/img (<n> calls, TPU)
#   zxing  : <ms> ms/img (<n> calls, CPU)
# so a single parser covers all four.
#
# Usage:
#   python3 tools/compare_statis.py --target BM1684X --platform SE7-32 \
#       --program wechat_qrcode_sail.soc --language cpp --input log/xx.log
#   python3 tools/compare_statis.py ... --threshold 0.7   # change regress guard

import argparse
import re

# Baseline end-to-end FPS per platform (same algorithm across implementations;
# measured with the C++ sail demo on detect_f32_fused + sr_f16_fused, device
# pass-through: hardware decode -> toBMI -> detectAndDecode(bm_image)). The
# bmcv / python variants run the identical detect/sr/zxing pipeline, so they
# must stay in this order.
BASELINE_FPS = {
    "SE7-32": 205.3,   # BM1684X
    "SE9-8": 113.3,    # CV186X
    "SE9-16": 114.2,   # BM1688
}

PATTERNS = {
    "e2e_fps": re.compile(r"e2e\s*:\s*[\d.]+\s*ms/img,\s*([\d.]+)\s*fps"),
    "e2e_ms": re.compile(r"e2e\s*:\s*([\d.]+)\s*ms/img"),
    "detect_ms": re.compile(r"detect\s*:\s*([\d.]+)\s*ms/img"),
    "sr_ms": re.compile(r"sr\s*:\s*([\d.]+)\s*ms/img"),
    "zxing_ms": re.compile(r"zxing\s*:\s*([\d.]+)\s*ms/img"),
}


def parse_bench(text):
    """Return a dict of metrics found in the bench log, None if nothing found."""
    metrics = {}
    for key, pat in PATTERNS.items():
        m = pat.search(text)
        if m:
            metrics[key] = float(m.group(1))
    return metrics or None


def main():
    parser = argparse.ArgumentParser(prog="compare_statis.py")
    parser.add_argument("--target", type=str, default="BM1684X")
    parser.add_argument("--platform", type=str, default="SE7-32",
                        help="SE7-32 | SE9-16 | SE9-8")
    parser.add_argument("--program", type=str, default="wechat_qrcode_sail.soc")
    parser.add_argument("--language", type=str, default="cpp",
                        choices=["cpp", "python"])
    parser.add_argument("--input", type=str, required=True,
                        help="bench log file")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="fps >= baseline*threshold to pass")
    parser.add_argument("--bmodel", type=str,
                        default="detect_f32_fused+sr_f16_fused")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        text = f.read()

    metrics = parse_bench(text)
    if metrics is None:
        print("FAIL: no bench metrics found in {}".format(args.input))
        return 1

    fps = metrics.get("e2e_fps", 0.0)
    header = ("| {:^11s} | {:^24s} | {:^34s} | {:^10s} | {:^9s} | {:^9s} "
              "| {:^9s} |".format("platform", "program", "model",
                                  "e2e_fps", "detect_ms", "sr_ms", "zxing_ms"))
    sep = ("| " + "-" * 11 + " | " + "-" * 24 + " | " + "-" * 34 + " | "
           + "-" * 10 + " | " + "-" * 9 + " | " + "-" * 9 + " | " + "-" * 9 + " |")
    row = ("| {:^11s} | {:^24s} | {:^34s} | {:>9.2f} | {:>8.2f} | {:>8.2f} "
           "| {:>8.2f} |".format(
               args.platform, args.program, args.bmodel[:34], fps,
               metrics.get("detect_ms", 0.0),
               metrics.get("sr_ms", 0.0),
               metrics.get("zxing_ms", 0.0)))
    print(header)
    print(sep)
    print(row)

    base = BASELINE_FPS.get(args.platform)
    if base is None:
        print("WARN: no baseline for platform {}, skipping fps guard".format(
            args.platform))
        return 0

    if fps >= base * args.threshold:
        print("PASS: {:.2f} fps >= {:.0f}*{:.1f}={:.1f} ({} / {})".format(
            fps, base, args.threshold, base * args.threshold,
            args.program, args.platform))
        return 0
    print("FAIL: {:.2f} fps < {:.0f}*{:.1f}={:.1f} ({} / {})".format(
        fps, base, args.threshold, base * args.threshold,
        args.program, args.platform))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())