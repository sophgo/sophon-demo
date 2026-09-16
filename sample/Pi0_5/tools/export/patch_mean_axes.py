#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""patch_mean_axes.py — dynamo(op18) ReduceMean axes 属性物化为第二输入(op17 形式), tpu-mlir 兼容."""

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
if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc
import onnx
from onnx import helper, numpy_helper
import numpy as np

D = _os.environ.get("PI05_ONNX_DIR") or f"{_PI05_ROOT}/onnx"

def patch(nm):
    m = onnx.load(f"{D}/{nm}.onnx", load_external_data=False)
    g = m.graph
    inits = {i.name for i in g.initializer}
    new_nodes = []
    new_inits = []
    for n in g.node:
        if n.op_type == "ReduceMean" and len(n.input) == 1:
            attrs = {a.name: a for a in n.attribute}
            if "axes" in attrs:
                axes = onnx.helper.get_attribute_value(attrs["axes"])
                aname = n.name + "_axes"
                if aname not in inits:
                    new_inits.append(numpy_helper.from_array(np.asarray(axes, dtype=np.int64), aname))
                n.input.append(aname)
                n.attribute.remove(attrs["axes"])
                print(f"  {nm}: ReduceMean {n.name} axes{list(axes)} -> input {aname}")
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)
    m.ir_version = 9
    onnx.save(m, f"{D}/{nm}_meanfix.onnx", save_as_external_data=False)
    print(f"saved {nm}_meanfix.onnx")

if __name__ == "__main__":
    for nm in ["pi05_dkv0_9", "pi05_dkv9_18"]:
        patch(nm)
    print("PATCH_MEAN_DONE")
