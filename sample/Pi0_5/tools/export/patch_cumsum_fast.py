#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""patch_cumsum_fast.py — 快速插 Cast 修 bool CumSum: 只改 proto, 不重写 external data."""

# NOTE: this step is carried over from the project's export directory because the same
# tpu-mlir limitations were hit during the port, but it has NOT been confirmed to be
# required for the H5 (prefix 536) chain specifically. Treat it as "run it if the
# compiler rejects the graph" rather than a mandatory step.

import os as _os

# All inputs and outputs live under PI05_ROOT, the working tree that holds the
# official openpi checkout (openpi-ref/), the official checkpoint (ckpt_official/)
# and the generated artifacts. Set it before running:
#     export PI05_ROOT=/path/to/pi05-work
_PI05_ROOT = _os.environ.get("PI05_ROOT")
if not _PI05_ROOT:
    raise SystemExit("PI05_ROOT is not set; see tools/export/README.md")

import datetime
if not hasattr(datetime, 'UTC'):
    datetime.UTC = datetime.timezone.utc
import sys, onnx
from onnx import helper

D = _os.environ.get("PI05_ONNX_DIR") or f"{_PI05_ROOT}/onnx"

def patch(path, out):
    m = onnx.load(path, load_external_data=False)
    # 类型推断
    try:
        m_inf = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    except Exception as e:
        print(f"  infer warn: {e}"); m_inf = m
    types = {}
    for inp in m_inf.graph.input:
        types[inp.name] = inp.type.tensor_type.elem_type
    for ini in m_inf.graph.initializer:
        types[ini.name] = ini.data_type
    for vi in m_inf.graph.value_info:
        types[vi.name] = vi.type.tensor_type.elem_type

    inserted = 0
    new_nodes = []
    for nd in m.graph.node:
        if nd.op_type == "CumSum" and len(nd.input) >= 1:
            src = nd.input[0]
            et = types.get(src)
            if et == onnx.TensorProto.BOOL:
                cast_out = src + "_as_i32"
                cast = helper.make_node("Cast", [src], [cast_out], name=f"{nd.name}_cb", to=onnx.TensorProto.INT32)
                new_nodes.append(cast)
                nd.input[0] = cast_out
                types[cast_out] = onnx.TensorProto.INT32
                inserted += 1
        new_nodes.append(nd)
    m.graph.ClearField("node")
    m.graph.node.extend(new_nodes)
    print(f"  {path.split('/')[-1]}: {inserted} Casts")
    # 只存 proto (external_data 引用保留, 不重写 .data)
    onnx.save(m, out)
    print(f"  saved {out}")

for n in ["pi05_ddn0_6", "pi05_ddn6_12", "pi05_ddn12_18"]:
    print(f"=== {n} ===")
    patch(f"{D}/{n}.onnx", f"{D}/{n}_fx.onnx")
print("PATCH_FAST_DONE")