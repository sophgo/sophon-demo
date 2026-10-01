#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# Evaluate wechat_qrcode results against the BoofCV QR Code V4 dataset.
#
# The dataset (download with scripts/download_datasets.sh) has two subsets:
#   qrcodes/detection/<16 categories>/imageNNN.{jpg,txt}
#       ground truth = hand-selected 2D corner sets, two on-disk layouts:
#         * "SETS" layout : a "# comment" line, a "SETS" marker line, then
#           one QR per line (8 floats = 4 corners).
#         * "pairs" layout: a "# comment" line, then one corner per line
#           (2 floats), 4 consecutive lines = one QR.
#   qrcodes/decoding/<name>.{png,txt}
#       <name>.txt = the exact expected text of the clean synthetic code.
#
# A program's --input=dataset run dumps a result JSON {image_path: [{text,
# points}]}; points[i] is a (4,2) float32 corner array in image coordinates.
#
# Metrics (aligned with the plan's detection IoU>=0.5 criterion):
#   detection: per-code greedy one-to-one match by polygon IoU, then
#              recall = matched / #GT, precision = matched / #pred,
#              plus per-category recall.
#   decoding : a code is correct iff its decoded text equals the expected text.
#
# Usage:
#   python3 tools/eval_qrcode.py --gt_path datasets/BoofCV_qrcode_v4/qrcodes \
#       --result_json results/xxx_result.json [--iou 0.5]

import argparse
import glob
import json
import os

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def order_points(pts):
    """Sort 4 (x,y) points CCW around their centroid (makes a simple quad)."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_polygon(subject, clip):
    """Sutherland-Hodgman clip of subject polygon against a convex clip poly."""
    output = subject
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        if len(output) == 0:
            break
        input_list = output
        output = []
        for j in range(len(input_list)):
            p = input_list[j]
            q = input_list[(j + 1) % len(input_list)]
            # inside == left of edge a->b (CCW clip polygon)
            inside_p = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-9
            inside_q = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) >= -1e-9
            if inside_p:
                if inside_q:
                    output.append(q)  # both inside: keep the far vertex
                else:
                    output.append(line_intersect(p, q, a, b))
            elif inside_q:
                # entering the clip polygon: emit intersection point and the
                # now-inside vertex (Sutherland-Hodgman)
                output.append(line_intersect(p, q, a, b))
                output.append(q)
    return np.array(output, dtype=np.float64) if len(output) else np.zeros((0, 2))


def line_intersect(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-12:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return np.array([px, py])


def polygon_iou(a, b):
    """IoU of two quads, order-independent (both ordered CCW first)."""
    a = order_points(np.asarray(a, dtype=np.float64))
    b = order_points(np.asarray(b, dtype=np.float64))
    inter = polygon_area(clip_polygon(a, b))
    area_a = polygon_area(a)
    area_b = polygon_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def parse_detection_gt(text):
    """Return a list of corner arrays (each (4,2) float)."""
    lines = [ln.strip() for ln in text.splitlines()]
    content = [ln for ln in lines if ln and not ln.startswith("#")]
    if not content:
        return []
    sets = []
    if content[0].upper() == "SETS":
        # 8 floats per line = one QR
        for ln in content[1:]:
            vals = [float(v) for v in ln.split()]
            if len(vals) == 8:
                sets.append(np.array(vals, dtype=np.float64).reshape(4, 2))
    else:
        # 2 floats per line, 4 lines per QR
        buf = []
        for ln in content:
            vals = [float(v) for v in ln.split()]
            if len(vals) == 2:
                buf.append(vals)
        for i in range(0, len(buf) - 3, 4):
            sets.append(np.array(buf[i:i + 4], dtype=np.float64))
    return sets


def lookup_result(result_map, category, stem):
    """Find the result entry for image category/stem.jpg by path suffix."""
    wanted = os.path.join(category, stem)
    for path, items in result_map.items():
        norm = os.path.splitext(os.path.normpath(path))[0]
        if norm.endswith(wanted):
            return items
    return None


def eval_detection(result_map, detection_root, iou_thresh):
    gt_files = sorted(glob.glob(os.path.join(detection_root, "*", "*.txt")))
    if not gt_files:
        return None, "no detection ground truth under {}".format(detection_root)

    total_gt = total_pred = matched = 0
    per_cat = {}
    for gt_file in gt_files:
        category = os.path.basename(os.path.dirname(gt_file))
        stem = os.path.splitext(os.path.basename(gt_file))[0]
        with open(gt_file) as f:
            gt_sets = parse_detection_gt(f.read())
        if not gt_sets:
            continue

        items = lookup_result(result_map, category, stem) or []
        pred = []
        for it in items:
            pts = np.asarray(it.get("points", []), dtype=np.float64)
            if pts.shape == (4, 2) and pts.size == 8:
                pred.append(pts)

        # greedy one-to-one match by IoU (order-independent)
        used_gt = [False] * len(gt_sets)
        used_pred = [False] * len(pred)
        n_match = 0
        pairs = []
        for i, g in enumerate(gt_sets):
            for j, p in enumerate(pred):
                pairs.append((polygon_iou(g, p), i, j))
        for iou, i, j in sorted(pairs, reverse=True):
            if iou < iou_thresh:
                break
            if used_gt[i] or used_pred[j]:
                continue
            used_gt[i] = used_pred[j] = True
            n_match += 1

        total_gt += len(gt_sets)
        total_pred += len(pred)
        matched += n_match
        d = per_cat.setdefault(category, [0, 0, 0])
        d[0] += len(gt_sets); d[1] += n_match; d[2] += len(pred)

    recall = matched / total_gt if total_gt else 0.0
    precision = matched / total_pred if total_pred else 0.0
    return {
        "total_gt": total_gt, "total_pred": total_pred, "matched": matched,
        "recall": recall, "precision": precision, "per_cat": per_cat,
    }, None


def eval_decoding(result_map, decoding_root):
    gt_files = sorted(glob.glob(os.path.join(decoding_root, "*.txt")))
    total = correct = 0
    detail = []
    for gt_file in gt_files:
        base = os.path.basename(gt_file)
        if base in ("readme.txt",) or base.endswith("~"):
            continue
        stem = os.path.splitext(base)[0]
        with open(gt_file) as f:
            expected = f.read().strip()
        items = lookup_result(result_map, "decoding", stem) or []
        texts = [it.get("text", "") for it in items]
        total += 1
        # vCard / vEvent payloads use CRLF line breaks per the QR spec, while the
        # BoofCV .txt ground truth was saved with LF; normalize both sides so a
        # correct decode is not penalised for the line-ending convention alone.
        norm = lambda s: s.replace("\r\n", "\n").replace("\r", "\n").strip()
        if any(norm(t) == norm(expected) for t in texts):
            correct += 1
        detail.append((stem, expected, texts))
    acc = correct / total if total else 0.0
    return {"total": total, "correct": correct, "accuracy": acc, "detail": detail}


def main():
    parser = argparse.ArgumentParser(prog="eval_qrcode.py")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="dataset root (contains qrcodes/detection and "
                             "qrcodes/decoding)")
    parser.add_argument("--result_json", type=str, required=True,
                        help="program result JSON {image_path: [{text, points}]}")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="detection IoU threshold (default 0.5)")
    parser.add_argument("--subset", type=str, default="all",
                        choices=["all", "detection", "decoding"])
    args = parser.parse_args()

    with open(args.result_json) as f:
        result_map = json.load(f)

    detection_root = os.path.join(args.gt_path, "detection")
    decoding_root = os.path.join(args.gt_path, "decoding")

    if args.subset in ("all", "detection"):
        det, err = eval_detection(result_map, detection_root, args.iou)
        if err:
            print("SKIP detection:", err)
        else:
            print("== detection (IoU >= {}) ==".format(args.iou))
            print("  images GT codes: {:4d}, predicted codes: {:4d}, "
                  "matched: {:4d}".format(
                      det["total_gt"], det["total_pred"], det["matched"]))
            print("  recall    : {:.4f} ({}/{})".format(
                det["recall"], det["matched"], det["total_gt"]))
            print("  precision : {:.4f} ({}/{})".format(
                det["precision"], det["matched"], det["total_pred"]))
            print("  per-category recall:")
            for cat in sorted(det["per_cat"]):
                gt, m, p = det["per_cat"][cat]
                print("    {:16s} {:4d}/{:4d} = {:.4f}".format(
                    cat, m, gt, m / gt if gt else 0.0))

    if args.subset in ("all", "decoding"):
        dec = eval_decoding(result_map, decoding_root)
        print("== decoding ==")
        print("  images: {:3d}, correct: {:3d}, accuracy: {:.4f}".format(
            dec["total"], dec["correct"], dec["accuracy"]))
        for stem, expected, texts in dec["detail"]:
            norm = lambda s: s.replace("\r\n", "\n").replace("\r", "\n").strip()
            flag = "OK " if any(norm(expected) == norm(t) for t in texts) else "MISS"
            got = texts[0] if texts else "<no decode>"
            print("    [{:2s}] {:<28s} want={!r} got={!r}".format(
                flag, stem, expected, got))


if __name__ == "__main__":
    main()