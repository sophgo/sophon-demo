#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将官方线性头 dinov2_vitb14_reg4_linear_head.pth 转成 numpy npz，
供推理脚本（无需 torch）直接加载：
  weight: [1000, 1536] float32, bias: [1000] float32

用法:
  python3 scripts/export_linear_head.py \
      --weights models/torch/dinov2_vitb14_reg4_linear_head.pth \
      --out datasets/dinov2_vitb14_reg4_linear_head.npz
"""
import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="datasets/dinov2_vitb14_reg4_linear_head.npz")
    ap.add_argument("--bin_out", default="datasets/dinov2_vitb14_reg4_linear_head.bin",
                    help="C++ 用的二进制（weight[1000*1536] float32 + bias[1000] float32）")
    args = ap.parse_args()

    sd = torch.load(args.weights, map_location="cpu")
    assert set(sd.keys()) == {"weight", "bias"}, f"意外键: {list(sd.keys())}"
    weight = sd["weight"].numpy().astype(np.float32)
    bias = sd["bias"].numpy().astype(np.float32)
    np.savez(args.out, weight=weight, bias=bias)
    print(f"weight {weight.shape}, bias {bias.shape} -> {args.out}")

    with open(args.bin_out, "wb") as f:
        f.write(np.ascontiguousarray(weight).tobytes())
        f.write(np.ascontiguousarray(bias).tobytes())
    print(f"bin (weight + bias) -> {args.bin_out}")


if __name__ == "__main__":
    main()