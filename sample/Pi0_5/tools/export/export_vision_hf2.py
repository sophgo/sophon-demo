#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/usr/bin/env python3
"""Correctly export the PaliGemma (HF transformers_replace) vision tower:
load weights by stripping the `paligemma_with_expert.` container prefix, wrap in
an nn.Module so jit.trace captures parameters, export ONNX with external data."""
import os as _os

# All inputs and outputs live under PI05_ROOT, the working tree that holds the
# official openpi checkout (openpi-ref/), the official checkpoint (ckpt_official/)
# and the generated artifacts. Set it before running:
#     export PI05_ROOT=/path/to/pi05-work
_PI05_ROOT = _os.environ.get("PI05_ROOT")
if not _PI05_ROOT:
    raise SystemExit("PI05_ROOT is not set; see tools/export/README.md")

# All ONNX artifacts land in PI05_ONNX_DIR so that the compile step can consume them
# from one place. Default is ${PI05_ROOT}/onnx; point the sample's models/onnx/ at it
# (copy or symlink) before running scripts/gen_*bmodel_mlir.sh.
_ONNX_DIR = _os.environ.get("PI05_ONNX_DIR") or f"{_PI05_ROOT}/onnx"
_os.makedirs(_ONNX_DIR, exist_ok=True)

import pathlib
import sys

import numpy as np
import torch
import safetensors.torch

# The openpi checkout may be named either openpi/ or openpi-ref/.
def _add_openpi_paths():
    for _d in ("openpi-ref", "openpi"):
        _p = _os.path.join(_PI05_ROOT, _d)
        if not _os.path.isdir(_p):
            continue
        for _sub in ("src", "packages/openpi-client/src"):
            _q = _os.path.join(_p, _sub)
            if _os.path.isdir(_q):
                sys.path.insert(0, _q)
        return _p
    raise SystemExit("no openpi checkout under PI05_ROOT (expected openpi/ or openpi-ref/)")


_add_openpi_paths()
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

BASE = pathlib.Path(_PI05_ROOT)
OUT = pathlib.Path(_ONNX_DIR)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pg = _gemma.get_config("gemma_2b")
    m = PaliGemmaWithExpertModel(pg, _gemma.get_config("gemma_300m"), use_adarms=[False, True], precision="float32")
    # The LIBERO checkpoint, same one every other export script here reads. Do not swap in
    # pi05_base_pytorch: LIBERO fine-tuning moved the vision tower well away from the base
    # model (image features differ by ~29% L2), so a bmodel built from base weights would
    # silently disagree with the reference by far more than the quantization error.
    ckpt = BASE / "models/pi05_libero_official_pt/model.safetensors"
    print("weights:", ckpt)
    sd = safetensors.torch.load_file(str(ckpt))
    prefix = "paligemma_with_expert."
    stripped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    miss, unexp = m.load_state_dict(stripped, strict=False)
    print("missing:", len(miss), "unexpected:", len(unexp))
    pe = m.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight
    print("loaded pos_embed std:", float(pe.detach().float().std()))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, (1, 3, 224, 224)).astype(np.float32)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        y = m.embed_image(xt)
    y = y.float().numpy()
    print("embed_image out:", y.shape, "range", y.min(), y.max())
    np.savez(OUT / "vision_input.npz", images=x)
    np.savez(OUT / "vision_ref.npz", image_features=y)

    class W(torch.nn.Module):
        def __init__(self, pwe):
            super().__init__()
            self.pwe = pwe

        def forward(self, x):
            return self.pwe.embed_image(x)

    w = W(m)
    with torch.no_grad():
        tr = torch.jit.trace(w, xt, strict=False, check_trace=False)
    print("trace ok")
    torch.onnx.export(
        tr,
        xt,
        OUT / "pi05_siglip.onnx",
        input_names=["images"],
        output_names=["image_features"],
        opset_version=17,
        # torch >= 2.9 defaults to the torch.export-based exporter, which refuses a traced
        # ScriptModule outright ("Exporting a ScriptModule is not supported"). The legacy
        # exporter is what this graph and the downstream tpu-mlir flow are written for.
        dynamo=False,
        # The batch axis has to stay symbolic. Tracing at batch 1 otherwise bakes
        # Reshape(..., [1,256,16,72]) into the attention blocks, and model_transform.py
        # then fails shape inference the moment it is compiled for the batch the sample
        # actually uses: "Input shape:{2,256,1152}, requested shape:{1,256,16,72}".
        dynamic_axes={"images": {0: "batch"}, "image_features": {0: "batch"}},
    )
    print("onnx ->", OUT / "pi05_siglip.onnx")
    print("DONE")


if __name__ == "__main__":
    main()