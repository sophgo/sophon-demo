#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# WeChatQRCode QR detection + recognition via the sail delivery API, with
# OpenCV (cv2) image loading: decode the image on the host with cv2.imdecode,
# then feed the resulting uint8 numpy array straight into
# sail.wechat_qrcode.WeChatQRCode.detectAndDecode(ndarray).
#
# Usage:
#   python3 wechat_qrcode_opencv.py --detect ... --sr ... --input <img|dir> \
#       [--dev_id 0] [--core_id -1] [--iters N]
#
# iters <= 0: decode each image (a file, or every image in a directory), print
#             text + 4 corners per QR, and dump a result JSON under ./results;
# iters >  0: bench mode -- decode one image iters times (3 warmups), print
#             end-to-end and detect/sr/zxing per-stage timings.

import argparse
import json
import logging
import os
import time

import cv2
import numpy as np
import sophon.sail as sail

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(input_dir):
    """Yield (full_path, filename) for every supported image under input_dir."""
    for root, _, filenames in os.walk(input_dir):
        for name in sorted(filenames):
            if os.path.splitext(name)[-1].lower() in IMG_EXTS:
                yield os.path.join(root, name), name


def load_image_bgr(path):
    """Decode one image to a uint8 BGR ndarray (H,W,3), or None on failure."""
    # IGNORE_ORIENTATION pins the read to the stored (raw EXIF) frame so the
    # four detected corners land in the same coordinate frame as the BoofCV
    # GT, independent of the cv2 version (matches the C++ host-decode path).
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def format_points(corners):
    """corners: (4,2) float32 -> human-readable string for logging/printing."""
    return " ".join("({:.1f},{:.1f})".format(float(p[0]), float(p[1])) for p in corners)


def bench(qr, image_path, iters):
    img = load_image_bgr(image_path)
    if img is None:
        raise FileNotFoundError("cannot decode image: {}".format(image_path))

    # The first TPU inference includes kernel loading, so warm up first.
    for _ in range(3):
        qr.detectAndDecode(img)

    qr.resetBenchStats()
    t0 = time.perf_counter()
    decoded = 0
    for _ in range(iters):
        texts, _ = qr.detectAndDecode(img)
        if texts:
            decoded += 1
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0

    s = qr.getBenchStats()
    def avg(ms, calls):
        return ms / calls if calls else 0.0

    print("=== bench: {} ({}x{}, iters={}, decoded={}) ===".format(
        image_path, img.shape[1], img.shape[0], iters, decoded))
    print("e2e    : {:.2f} ms/img, {:.2f} fps".format(
        total_ms / iters, 1000.0 * iters / total_ms))
    print("detect : {:.2f} ms/img ({} calls, TPU)".format(
        avg(s["detect_ms"], s["detect_calls"]), s["detect_calls"]))
    print("sr     : {:.2f} ms/img ({} calls, TPU)".format(
        avg(s["sr_ms"], s["sr_calls"]), s["sr_calls"]))
    print("zxing  : {:.2f} ms/img ({} calls, CPU)".format(
        avg(s["zxing_ms"], s["zxing_calls"]), s["zxing_calls"]))


def run(qr, input_path, tag, detect_path):
    """Single-frame mode: decode a file or directory, dump JSON results."""
    os.makedirs("results", exist_ok=True)

    if os.path.isdir(input_path):
        images = list(list_images(input_path))
    else:
        images = [(input_path, os.path.basename(input_path))]

    results = {}
    n_total = 0
    for img_file, name in images:
        img = load_image_bgr(img_file)
        if img is None:
            logging.error("{} imdecode is None.".format(img_file))
            continue

        texts, points = qr.detectAndDecode(img)
        logging.info("[{}] decoded {} code(s)".format(img_file, len(texts)))
        n_total += len(texts)

        items = []
        for i, text in enumerate(texts):
            corners = points[i] if i < len(points) else np.zeros((0, 2), np.float32)
            logging.info("  [{}] text={} corners={}".format(
                i, text, format_points(corners)))
            items.append({
                "text": text,
                "points": corners.tolist(),
            })
        results[img_file] = items

    bname = os.path.splitext(os.path.basename(detect_path))[0]
    iname = os.path.splitext(os.path.basename(input_path.rstrip("/")))[0]
    json_name = "{}_{}_{}_python_result.json".format(bname, iname, tag)
    with open(os.path.join("results", json_name), "w") as jf:
        json.dump(results, jf, indent=4, ensure_ascii=False)
    logging.info("result saved in results/{} ({}/{} images decoded)".format(
        json_name, n_total, len(images)))

    if len(images) == 1:
        return 0 if n_total > 0 else 1
    return 0


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument("--detect", type=str,
                        default="../models/BM1684X/detect_f32_fused.bmodel",
                        help="path of the detect bmodel")
    parser.add_argument("--sr", type=str,
                        default="../models/BM1684X/sr_f16_fused.bmodel",
                        help="path of the super-resolution bmodel")
    parser.add_argument("--input", type=str, default="../images/qr_small.png",
                        help="path of input (image file or directory)")
    parser.add_argument("--dev_id", type=int, default=0, help="TPU device id")
    parser.add_argument("--core_id", type=int, default=-1,
                        help="BM1688 dual-core pin (0/1), -1 = auto")
    parser.add_argument("--iters", type=int, default=0,
                        help=">0 runs bench mode with this many iterations")
    return parser.parse_args()


if __name__ == "__main__":
    args = argsparser()
    if not os.path.exists(args.input):
        raise FileNotFoundError("{} is not existed.".format(args.input))

    qr = sail.wechat_qrcode.WeChatQRCode(
        args.detect, args.sr, args.dev_id, args.core_id)
    logging.info("load {} + {} success!".format(args.detect, args.sr))

    if args.iters > 0:
        bench(qr, args.input, args.iters)
    else:
        run(qr, args.input, "opencv", args.detect)

    print("all done.")