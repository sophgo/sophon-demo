#!/usr/bin/env python3
"""
FEARTracker single-object visual tracking demo using SAIL TPU inference.

FEAR: Fast, Efficient, Accurate and Robust Visual Tracker (ECCV 2022)
Model: FBNet-C backbone + Siamese cross-correlation tracker.
"""

import argparse
import os
import time
import numpy as np
import cv2
from collections import deque

import sophon.sail as sail

# ============================================================
# Default tracking configuration
# ============================================================
DEFAULT_CONFIG = {
    "penalty_k": 0.062,
    "window_influence": 0.38,
    "lr": 0.765,
    "windowing": "cosine",
    "total_stride": 16,
    "score_size": 16,
    "template_bbox_offset": 0.2,
    "search_context": 2,
    "instance_size": 256,
    "template_size": 128,
    "smooth": False,
}

# ============================================================
# Utility functions (self-contained, no torch/hydra dependency)
# ============================================================

def make_grid(score_size, total_stride, instance_size):
    x, y = np.meshgrid(
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
    )
    grid_x = x * total_stride + instance_size // 2
    grid_y = y * total_stride + instance_size // 2
    return grid_x[np.newaxis, :, :], grid_y[np.newaxis, :, :]


def limit(radius):
    return np.maximum(radius, 1.0 / radius)


def squared_size(w, h):
    pad = (w + h) * 0.5
    size = (w + pad) * (h + pad)
    return np.sqrt(size)


def ensure_bbox_boundaries(bbox, img_shape):
    x1, y1, w, h = bbox.astype("int32")
    x1, y1 = min(max(0, x1), img_shape[1]), min(max(0, y1), img_shape[0])
    x2, y2 = min(max(0, x1 + w), img_shape[1]), min(max(0, y1 + h), img_shape[0])
    w, h = x2 - x1, y2 - y1
    return np.array([x1, y1, w, h])


def clamp_bbox(bbox, shape, min_side=3):
    bbox = ensure_bbox_boundaries(bbox, shape)
    x, y, w, h = bbox
    img_h, img_w = shape[0], shape[1]
    if w < min_side:
        w = min_side
        x -= max(0, x + w - img_w)
    if h < min_side:
        h = min_side
        y -= max(0, y + h - img_h)
    return np.array([x, y, w, h])


def extend_bbox(bbox, offset=0.1):
    x, y, w, h = bbox
    if isinstance(offset, tuple):
        if len(offset) == 4:
            left, right, top, bottom = offset
        elif len(offset) == 2:
            w_offset, h_offset = offset
            left = right = w_offset
            top = bottom = h_offset
    else:
        left = right = top = bottom = offset
    return np.array([
        x - w * left, y - h * top,
        w * (1.0 + right + left), h * (1.0 + top + bottom)
    ]).astype("int32")


def get_extended_crop(image, bbox, crop_size, offset, padding_value=None):
    if padding_value is None:
        padding_value = np.mean(image, axis=(0, 1))
    context = extend_bbox(bbox, offset)
    pad_left = max(-context[0], 0)
    pad_top = max(-context[1], 0)
    pad_right = max(context[0] + context[2] - image.shape[1], 0)
    pad_bottom = max(context[1] + context[3] - image.shape[0], 0)

    crop = image[
        context[1] + pad_top: context[1] + context[3] - pad_bottom,
        context[0] + pad_left: context[0] + context[2] - pad_right,
    ].copy()

    padded_crop = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=padding_value.tolist()
    )
    padded_bbox = np.array([bbox[0] - context[0], bbox[1] - context[1], bbox[2], bbox[3]])
    padded_bbox = ensure_bbox_boundaries(padded_bbox, padded_crop.shape[:2])

    resized = cv2.resize(padded_crop, (crop_size, crop_size))
    w_scale = crop_size / padded_crop.shape[1]
    h_scale = crop_size / padded_crop.shape[0]
    resized_bbox = np.array([
        padded_bbox[0] * w_scale, padded_bbox[1] * h_scale,
        padded_bbox[2] * w_scale, padded_bbox[3] * h_scale,
    ])
    return resized, resized_bbox, context


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def draw_bbox(image, bbox, width=5):
    image = image.copy()
    x, y, w, h = bbox
    return cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), width)


# ============================================================
# FEARTracker with SAIL TPU backend
# ============================================================

class TrackingState:
    def __init__(self):
        self.bbox = None
        self.mapping = None
        self.prev_size = None
        self.mean_color = None


class FEARTracker:
    def __init__(self, bmodel_path, dev_id=0):
        self.config = dict(DEFAULT_CONFIG)
        self.state = TrackingState()

        self.handle = sail.Handle(dev_id)
        self.engine = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_names = self.engine.get_input_names(self.graph_name)
        self.output_names = self.engine.get_output_names(self.graph_name)

        print(f"Model loaded: {bmodel_path}")
        print(f"  inputs : {self.input_names}")
        print(f"  outputs: {self.output_names}")

        self.template_img = None

        grid_x, grid_y = make_grid(
            self.config["score_size"],
            self.config["total_stride"],
            self.config["instance_size"],
        )
        self.grid_x = grid_x
        self.grid_y = grid_y

        self.window = self._make_window(
            self.config["windowing"], self.config["score_size"]
        )

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _make_window(self, windowing, score_size):
        if windowing == "cosine":
            return np.outer(np.hanning(score_size), np.hanning(score_size))
        return np.ones((score_size, score_size))

    def _preprocess(self, image):
        img = image[:, :, :3].astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        arr = np.transpose(img, (2, 0, 1))
        arr = np.expand_dims(arr, 0).astype(np.float32)
        return arr

    def initialize(self, image, bbox):
        bbox = clamp_bbox(np.array(bbox), image.shape)
        self.state.bbox = bbox
        self.state.mean_color = np.mean(image, axis=(0, 1))

        template_crop, _, _ = get_extended_crop(
            image=image, bbox=bbox,
            offset=self.config["template_bbox_offset"],
            crop_size=self.config["template_size"],
        )
        self.template_img = self._preprocess(template_crop)

    def update(self, image):
        search_crop, search_bbox, padded_bbox = get_extended_crop(
            image=image, bbox=self.state.bbox,
            crop_size=self.config["instance_size"],
            offset=self.config["search_context"],
            padding_value=self.state.mean_color,
        )
        self.state.mapping = padded_bbox
        self.state.prev_size = search_bbox[2:]

        search_img = self._preprocess(search_crop)

        input_data = {
            self.input_names[0]: self.template_img,
            self.input_names[1]: search_img,
        }
        output = self.engine.process(self.graph_name, input_data)

        bbox_pred = output[self.output_names[0]]    # [1, 4, 16, 16]
        cls_pred = output[self.output_names[1]]      # [1, 1, 16, 16]

        pred_bbox, _ = self._postprocess(bbox_pred, cls_pred)
        pred_bbox = self._rescale_bbox(pred_bbox, self.state.mapping)
        pred_bbox = clamp_bbox(pred_bbox, image.shape)
        self.state.bbox = pred_bbox
        return pred_bbox

    def _rescale_bbox(self, bbox, padded_box):
        w_scale = padded_box[2] / self.config["instance_size"]
        h_scale = padded_box[3] / self.config["instance_size"]
        bbox = bbox.copy()
        bbox[0] = round(bbox[0] * w_scale + padded_box[0])
        bbox[1] = round(bbox[1] * h_scale + padded_box[1])
        bbox[2] = max(3, round(bbox[2] * w_scale))
        bbox[3] = max(3, round(bbox[3] * h_scale))
        return np.array([int(x) for x in bbox])

    def _postprocess(self, bbox_pred, cls_pred):
        cls_score = cls_pred[0, 0, :, :]  # [16, 16]
        cls_sigmoid = 1.0 / (1.0 + np.exp(-cls_score))
        regression_map = bbox_pred[0]      # [4, 16, 16]

        if self.config.get("smooth", False):
            classification_map, penalty = self._confidence_postprocess(
                cls_sigmoid, regression_map
            )
        else:
            classification_map = cls_sigmoid
            penalty = None

        pred_location = np.stack([
            self.grid_x[0] - regression_map[0],
            self.grid_y[0] - regression_map[1],
            self.grid_x[0] + regression_map[2],
            self.grid_y[0] + regression_map[3],
        ], axis=0)  # [4, 16, 16]

        max_idx = np.argmax(classification_map)
        r_max, c_max = unravel_index(max_idx, classification_map.shape)
        output = [pred_location[i, r_max, c_max] for i in range(4)]
        pred_bbox = np.array([
            output[0], output[1],
            output[2] - output[0], output[3] - output[1]
        ])

        if self.config.get("smooth", False) and penalty is not None:
            lr = (penalty[r_max, c_max] * cls_sigmoid[r_max, c_max]
                  * self.config["lr"])
            prev_size = self.state.prev_size
            if prev_size is not None:
                size = pred_bbox[2:] * lr
                prev = np.array(prev_size, dtype=np.float64) * (1 - lr)
                pred_bbox[2] = prev[0] + lr * (size[0] + prev[0])
                pred_bbox[3] = prev[1] + lr * (size[1] + prev[1])

        return pred_bbox, cls_sigmoid[r_max, c_max]

    def _confidence_postprocess(self, cls_score, regression_map):
        prev_size = self.state.prev_size
        pred_location = np.stack([
            self.grid_x[0] - regression_map[0],
            self.grid_y[0] - regression_map[1],
            self.grid_x[0] + regression_map[2],
            self.grid_y[0] + regression_map[3],
        ], axis=0)

        w_pred = pred_location[2] - pred_location[0]
        h_pred = pred_location[3] - pred_location[1]
        s_c = limit(squared_size(w_pred, h_pred) / squared_size(prev_size[0], prev_size[1]))
        r_c = limit((prev_size[0] / prev_size[1]) / (w_pred / h_pred))

        penalty = np.exp(-(r_c * s_c - 1) * self.config["penalty_k"])
        pscore = penalty * cls_score
        pscore = (pscore * (1 - self.config["window_influence"])
                  + self.window * self.config["window_influence"])
        return pscore, penalty


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FEAR Tracker SAIL Demo")
    parser.add_argument("--bmodel", type=str,
                        default="../models/BM1684X/feartracker_fp32_1b.bmodel",
                        help="bmodel file path")
    parser.add_argument("--input", type=str, required=True,
                        help="input video file path")
    parser.add_argument("--initial_bbox", type=str, required=True,
                        help="initial bbox in x,y,w,h format (e.g. 163,53,45,174)")
    parser.add_argument("--output", type=str, default=None,
                        help="output video file path (optional)")
    parser.add_argument("--dev_id", type=int, default=0,
                        help="TPU device id (default: 0)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="confidence threshold for tracking score (optional)")
    args = parser.parse_args()

    bbox = np.array([int(x) for x in args.initial_bbox.split(",")])
    assert len(bbox) == 4, "--initial_bbox must be x,y,w,h (4 comma-separated ints)"

    print(f"Loading video: {args.input}")
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Failed to open video: {args.input}")
        return

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    print(f"Video: {len(frames)} frames, {fps:.1f} fps")

    tracker = FEARTracker(args.bmodel, args.dev_id)
    print(f"Initial bbox: {bbox.tolist()}")
    tracker.initialize(frames[0], bbox)

    tracked_bboxes = [bbox]
    total_time = 0.0
    for i, frame in enumerate(frames[1:], 1):
        t0 = time.time()
        pred_bbox = tracker.update(frame)
        dt = time.time() - t0
        total_time += dt
        tracked_bboxes.append(pred_bbox)
        if i % 50 == 0 or i == len(frames) - 1:
            print(f"  frame {i}/{len(frames)}: bbox={pred_bbox.tolist()}")

    print(f"Tracked {len(frames)} frames, {len(tracked_bboxes)} bboxes")
    avg_time = total_time / max(1, len(frames) - 1) * 1000
    print(f"Average inference time: {avg_time:.1f} ms/frame")

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        for frame, b in zip(frames, tracked_bboxes):
            vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vis = draw_bbox(vis, b)
            writer.write(vis)
        writer.release()
        print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()