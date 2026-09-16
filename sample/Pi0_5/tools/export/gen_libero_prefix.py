#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""gen_libero_prefix.py — library module for building a LIBERO prefix with the official
PaliGemma tokenizer and PI0Pytorch.embed_prefix.

Provides the two helpers gen_kvseg_pertask.py imports:

    tokenize(text, sp, maxlen=200)        -> (ids, mask)
    build_prefix(model, images, img_masks, lang_ids, lang_mask, horizon=10)
                                          -> (prefix_embs, amask, pad_mask, att2d)

Run `python3 gen_libero_prefix.py` for a standalone self-check of those contracts."""
import os as _os

# All inputs and outputs live under PI05_ROOT, the working tree that holds the
# official openpi checkout (openpi-ref/ or openpi/), the official checkpoint
# (ckpt_official/) and the generated artifacts. Set it before running:
#     export PI05_ROOT=/path/to/pi05-work
_PI05_ROOT = _os.environ.get("PI05_ROOT")
if not _PI05_ROOT:
    raise SystemExit("PI05_ROOT is not set; see tools/export/README.md")

# Sibling scripts in this directory are imported as top-level modules.
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

# The openpi checkout may be named either openpi/ or openpi-ref/.
def _add_openpi_paths():
    import os
    for name in ("openpi-ref", "openpi"):
        root = os.path.join(_PI05_ROOT, name)
        if not os.path.isdir(root):
            continue
        for sub in ("src", "packages/openpi-client/src", "third_party/libero"):
            p = os.path.join(root, sub)
            if os.path.isdir(p):
                _sys.path.insert(0, p)
        return root
    raise SystemExit("no openpi checkout under PI05_ROOT (expected openpi/ or openpi-ref/)")

_add_openpi_paths()

import pathlib, sys, math, os
import numpy as np
import torch, safetensors.torch as st
import sentencepiece



from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from export_denoise_pp import _Cfg

BASE = pathlib.Path(_PI05_ROOT)
MASK_VAL = -1e4

def load_model():
    model = PI0Pytorch(_Cfg())
    sd = st.load_file(os.environ.get("PI05_SD", str(BASE / "models/pi05_libero_pytorch/model.safetensors")))
    model.load_state_dict(sd, strict=False); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

def tokenize(text, sp, maxlen=200):
    cleaned = text.strip().replace("_", " ").replace("\n", " ")
    toks = sp.encode(cleaned, add_bos=True) + sp.encode("\n")
    ids = np.zeros(maxlen, dtype=np.int64); ids[:len(toks)] = toks
    mask = np.zeros(maxlen, dtype=np.bool_); mask[:len(toks)] = True
    return ids, mask

def build_prefix(model, images, img_masks, lang_ids, lang_mask, horizon=10):
    """images: list of [1,3,224,224] float32 tensors. Returns prefix_embs, amask, pad_mask."""
    embs, pad_masks, att_masks = model.embed_prefix(
        images, img_masks,
        torch.from_numpy(lang_ids)[None], torch.from_numpy(lang_mask)[None])
    S_p = embs.shape[1]
    att_all = torch.cat([torch.zeros(1, S_p, dtype=torch.bool),
                         torch.tensor([[1.0] + [0.0] * (horizon - 1)], dtype=torch.bool)], dim=1)
    pad = torch.cat([pad_masks, torch.ones(1, horizon, dtype=torch.bool)], dim=1)
    att2d = make_att_2d_masks(pad, att_all)
    amask = torch.where(att2d, 0.0, MASK_VAL).float()[:, None]
    pad_mask = pad_masks[0].numpy()
    return embs.detach().numpy(), amask.detach().numpy(), pad_mask, att2d.detach().numpy()

def main():
    """Standalone self-check of this module's contracts.

    Builds a prefix from a synthetic observation and a fixed prompt, then asserts the
    tensor shapes and dtypes that gen_kvseg_pertask.py relies on. Needs the PyTorch
    checkpoint (PI05_SD) and the PaliGemma tokenizer (PI05_TOKENIZER), but no other
    project artifacts.

    Returns:
        Process exit code: 0 on success, 1 on a contract violation.
    """
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--tokenizer",
                    default=_os.environ.get("PI05_TOKENIZER",
                                            str(pathlib.Path(_PI05_ROOT) /
                                                "demo_assets/paligemma_tokenizer.model")))
    ap.add_argument("--prompt", default="pick up the black bowl and place it on the plate")
    args = ap.parse_args()

    if not _os.path.exists(args.tokenizer):
        raise SystemExit(f"tokenizer not found: {args.tokenizer}; pass --tokenizer")

    model = load_model()
    sp = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
    ids, mask = tokenize(args.prompt, sp)
    print(f"prompt tokens: {int(mask.sum())}")
    assert ids.shape == (200,) and mask.shape == (200,), (ids.shape, mask.shape)

    rng = np.random.default_rng(7)
    def to_cfgx():
        return torch.from_numpy(
            rng.uniform(-1, 1, (1, 3, 224, 224)).astype(np.float32))
    images = [to_cfgx(), to_cfgx(), torch.zeros(1, 3, 224, 224)]
    img_masks = [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool),
                 torch.zeros(1, dtype=torch.bool)]

    embs, amask, pad_mask, att2d = build_prefix(model, images, img_masks, ids, mask, 10)
    print("prefix_embs:", embs.shape, "amask:", amask.shape, "pad_mask:", pad_mask.shape)
    ok = (embs.shape == (1, 968, 2048) and amask.shape == (1, 1, 978, 978)
          and pad_mask.shape == (968,) and att2d.shape == (1, 978, 978))
    print("GEN_LIBERO_PREFIX_OK" if ok else "CONTRACT_MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())