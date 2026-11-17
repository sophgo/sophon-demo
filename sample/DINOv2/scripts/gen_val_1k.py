#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ImageNet val zip（按类别文件夹组织）生成 sophon-demo 风格的
imagenet_val_1k 演示数据集：每个类别取 1 张，共 1000 张，平铺到 img/，
并生成 label.txt（`文件名\t类别下标`，对齐 ResNet/官方 eval 约定）。

用法:
  python3 scripts/gen_val_1k.py \
      --val_zip <path/to/imagenet_validation.zip> \
      --out_dir datasets/imagenet_val_1k
"""
import argparse
import json
import os
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_zip", required=True)
    ap.add_argument("--out_dir", default="datasets/imagenet_val_1k")
    ap.add_argument("--synset_map", default="datasets/imagenet_synset_to_idx.json")
    args = ap.parse_args()

    synset2idx = json.load(open(args.synset_map))
    img_dir = os.path.join(args.out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    zf = zipfile.ZipFile(args.val_zip)
    # 每个类别取第一张
    by_synset = {}
    for n in zf.namelist():
        if not n.lower().endswith(".jpeg"):
            continue
        s = n.split("/")[0]
        if s in synset2idx:
            by_synset.setdefault(s, n)

    lines = []
    for synset in sorted(by_synset):
        src = by_synset[synset]
        dst = os.path.join(img_dir, src.split("/")[-1])
        with open(dst, "wb") as f:
            f.write(zf.read(src))
        lines.append("{}\t{}".format(src.split("/")[-1], synset2idx[synset]))

    with open(os.path.join(args.out_dir, "label.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("images: {}, label.txt -> {}".format(len(lines), args.out_dir))


if __name__ == "__main__":
    main()