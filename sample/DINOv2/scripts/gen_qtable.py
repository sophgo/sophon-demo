#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 top-level MLIR 生成 INT8 部署所需的 quantize_table（qtable）。

qtable 采用「F16 白名单」写法：只列出需量化为 F16 的 op，其余 op 在
tpu-mlir 校准/部署时默认走 INT8 对称量化。DINOv2 混合精度策略：

  - top.Input                                    -> INT8（不写进 qtable，走默认）
  - top.MatMul 带 bias 的线性层（ONNX Gemm 导出，name 形如 _v_*_Gemm，
    每 block 4 个：qkv 融合投影 / attn.proj / mlp.fc1 / mlp.fc2，共 48 个）
                                                 -> INT8（同上）
  - 其余所有 op（24 个无 bias 的 QK^T/attn@V、patch 均值 MatMul、
    LayerNorm/Softmax/GELU/残差/布局等，共 388 个）-> F16（显式写出）

全部 48 个投影 Gemm 统一留 INT8（延迟最低），不再提供 tail 精度档；
三颗芯片使用同一份 qtable。

用法:
  python3 gen_qtable.py <top.mlir> <out.qtable>
"""

import argparse

from tpu_mlir.python.utils.mlir_parser import MlirParser

ap = argparse.ArgumentParser()
ap.add_argument("mlir_file")
ap.add_argument("out_file")
args = ap.parse_args()

parser = MlirParser(args.mlir_file)
int8_count = 0
f16_lines = []
int8_gemm = []          # 48 个带 bias 投影 Gemm 的 name，默认 INT8
for op in parser.ops:
    t = op.type
    name = op.name
    if t == "top.Input":
        int8_count += 1            # Input 默认 INT8，不写进 qtable
    elif t == "top.MatMul" and not (name or "").startswith("matmul"):
        int8_count += 1            # 投影头 Gemm 默认 INT8，不写进 qtable
        int8_gemm.append(name)
    else:
        f16_lines.append(f"{name} F16")

with open(args.out_file, "w") as f:
    f.write("\n".join(f16_lines) + "\n")

print(f"total ops: {len(parser.ops)}  "
      f"INT8(默认): {int8_count}  F16(白名单): {len(f16_lines)}")