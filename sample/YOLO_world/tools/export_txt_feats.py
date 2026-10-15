#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""Run the CLIP text encoder once and export txt_feats for the C++ example.

The C++ example does not embed CLIP text encoding / BPE tokenization; it loads
a precomputed [1, num_classes, 512] float32 .npy at startup instead. This
script reuses the Python example's TextEmbedder to produce that file.

Usage:
    python3 export_txt_feats.py --bmodel ../models/BM1684X/yoloworld_fp16_1b.bmodel \
        --clip_bmodel ../models/BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel \
        --class_names_file ../datasets/coco.names \
        --output ../models/BM1684X/coco80_txt_feats.npy
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../python'))

import torch  # noqa: E402
from yoloworld_opencv import TextEmbedder, YOLOworld  # noqa: E402
from utils import COCO_CLASSES  # noqa: E402


def load_class_names(path):
    names = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    return names


def main(args):
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    if not os.path.exists(args.clip_bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.clip_bmodel))

    if args.class_names == ['all']:
        class_names = list(COCO_CLASSES)[1:]  # 80 COCO classes, no background
    elif args.class_names_file:
        class_names = load_class_names(args.class_names_file)
    else:
        class_names = args.class_names

    yoloworld = YOLOworld(args)
    num_classes = yoloworld.num_classes
    print("num_classes from bmodel: {}, given class names: {}".format(
        num_classes, len(class_names)))

    # Pad with 'nan' (same as the Python example) and embed
    text_embedder = TextEmbedder(args)
    num_nan_to_add = num_classes - len(class_names)
    class_embeddings = text_embedder(class_names + ['nan'] * num_nan_to_add)

    # Zero-pad / trim to num_classes (same as prepare_embeddings)
    if class_embeddings.shape[1] != num_classes:
        class_embeddings = torch.nn.functional.pad(
            class_embeddings,
            (0, 0, 0, num_classes - class_embeddings.shape[1]),
            mode='constant', value=0)

    txt_feats = class_embeddings.cpu().numpy().astype(np.float32)
    np.save(args.output, txt_feats)
    print("saved txt_feats shape={} to {}".format(txt_feats.shape, args.output))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str,
                        default='../models/BM1684X/yoloworld_fp16_1b.bmodel',
                        help='path of yoloworld bmodel')
    parser.add_argument('--clip_bmodel', type=str,
                        default='../models/BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel',
                        help='path of clip text bmodel')
    parser.add_argument('--class_names', nargs='+', default=[],
                        help='class names list, e.g. person car dog cat')
    parser.add_argument('--class_names_file', type=str, default='',
                        help='class names file, one name per line (overrides --class_names)')
    parser.add_argument('--output', type=str,
                        default='../models/BM1684X/coco80_txt_feats.npy',
                        help='output .npy path [1,num_classes,512] float32')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    # Only read by YOLOworld(args) to construct the PostProcess; no inference
    # is run, so the values do not affect the exported embeddings.
    parser.add_argument('--conf_thresh', type=float, default=0.25)
    parser.add_argument('--nms_thresh', type=float, default=0.7)
    return parser.parse_args()


if __name__ == '__main__':
    main(argsparser())
