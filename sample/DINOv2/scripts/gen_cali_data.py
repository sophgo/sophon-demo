#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 DINOv2 INT8 标定数据 + model_transform 测试输入（npz）。

标定数据格式（对齐 tpu_mlir run_calibration 的 npz 约定）：
  单输入模型，每个 .npz 只含一个数组（key 任意，这里用 "input"），
  值为已过 DINOv2 官方 eval 前处理的 float32 [1,3,224,224]。

前处理严格对齐 dinov2/data/transforms.py: make_classification_eval_transform
的默认参数（= 官方公开的 ImageNet-1k linear-probe 精度口径）：
  Resize(256, BICUBIC) -> CenterCrop(224) -> ToTensor([0,1],RGB) -> Normalize

用法:
  python3 scripts/gen_cali_data.py \
      --val_zip <path/to/imagenet_validation.zip> \
      --num 200
"""
import argparse
import io
import os
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESIZE_SIZE = 256
CROP_SIZE = 224


def make_transform():
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def decode(im_bytes):
    return Image.open(io.BytesIO(im_bytes)).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_zip", required=True, help="ImageNet val zip（扁平或按类别文件夹组织均可；按类别组织时跨类别采样更均匀）")
    ap.add_argument("--num", type=int, default=200, help="标定图像数量")
    ap.add_argument("--out_dir", default=None, help="标定 npz 输出目录")
    ap.add_argument("--test_dir", default=None, help="model_transform 测试输入输出目录")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(base, "datasets", "cali_data")
    test_dir = args.test_dir or os.path.join(base, "datasets", "test_input")

    t = make_transform()
    zf = zipfile.ZipFile(args.val_zip)
    names = [n for n in zf.namelist() if n.lower().endswith(".jpeg")]
    names.sort()
    assert len(names) >= args.num, "zip 中只有 {} 张图，不足 {}".format(len(names), args.num)
    # 跨类别均匀采样 num 张
    step = len(names) // args.num
    sel = names[::step][: args.num]

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    tensors = []
    for i, n in enumerate(sel):
        x = t(decode(zf.read(n)))  # [3,224,224]
        tensors.append(x)
        arr = x.numpy()[None].astype(np.float32)  # [1,3,224,224]
        np.savez(os.path.join(out_dir, "cali_{:04d}.npz".format(i)), input=arr)
    print("标定数据: {} 个 npz -> {}".format(len(sel), out_dir))

    # model_transform 测试输入（batch 1），取第一张标定图
    b1 = tensors[0][None].numpy().astype(np.float32)
    np.savez(os.path.join(test_dir, "dinov2_1b_in_f32.npz"), input=b1)
    # 参考输出（官方 eval 前的归一化输入对应的模型输出）
    print("测试输入: dinov2_1b_in_f32.npz -> {}".format(test_dir))


if __name__ == "__main__":
    main()
