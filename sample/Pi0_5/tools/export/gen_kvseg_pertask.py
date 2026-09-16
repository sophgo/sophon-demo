#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/usr/bin/env python3
"""gen_kvseg_pertask.py — builds the per-task prefix assets for the Pi0_5 example.

For each of the ten LIBERO-Spatial tasks it runs the official prefix builder once
(PaliGemma tokenizer + PI0Pytorch.embed_prefix) and writes the tensors the device
needs, one file per task and tensor:

    tXX_{p_amask,p_cos,p_sin,f4d,s_cos,s_sin}.npy   full 968-token prefix geometry
    prompt200_tXX.npy                               200 language token embeddings

The image half of the prefix is not written out: the device recomputes it from the
actual observation with the SigLIP bmodel, so the sample observation fed here only
has to be the right shape. Only the language half (prompt200) is task dependent and
has to be baked in.

The 968-token tensors are then reduced to the 536-token set the device loads by
gen_assets_536.py, which also consumes prompt200_tXX.npy.
"""
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
import os
import pathlib
import sys
import numpy as np
import torch
import sentencepiece

BASE = _PI05_ROOT

def _add_openpi_paths():
    """Puts the openpi checkout on sys.path. The checkout may be named either
    openpi/ or openpi-ref/; try both so the scripts work with either layout."""
    import os
    for name in ("openpi-ref", "openpi"):
        root = os.path.join(_PI05_ROOT, name)
        if not os.path.isdir(root):
            continue
        for sub in ("src", "packages/openpi-client/src", "third_party/libero"):
            p = os.path.join(root, sub)
            if os.path.isdir(p):
                sys.path.insert(0, p)
        return root
    raise SystemExit("no openpi checkout under PI05_ROOT (expected openpi/ or openpi-ref/)")

_add_openpi_paths()
sys.path.insert(0, BASE)
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models_pytorch import pi0_pytorch as _pp
_pp.get_safe_dtype = lambda t, d: torch.float32
from openpi.training import config as _config
from openpi.policies import policy_config
from gen_libero_prefix import tokenize, build_prefix

# Output layout is the 968-token per-task set consumed by gen_assets_536.py, which then
# derives the 536-token assets the device loads. One file per tensor per task:
#   tXX_{p_amask,p_cos,p_sin,f4d,s_cos,s_sin}.npy
TARGET = _os.environ.get("PI05_ASSETS_IN", f"{_PI05_ROOT}/assets/dkva_npy")

def get_spatial_tasks():
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    return [bd["libero_spatial"]().get_task(i).language for i in range(10)]

def load_views(obs_dir):
    """Loads the two camera views used as the shape carrier for the prefix.

    Accepts either .npy (what this sample's dataset stores) or .png. The pixel values do
    not reach any output file -- the device recomputes the image embeddings from the real
    observation -- so any 224x224x3 RGB pair works.

    Args:
        obs_dir: directory holding agentview.{npy,png} and wrist.{npy,png}

    Returns:
        A pair of uint8 HxWx3 RGB arrays.
    """
    from PIL import Image

    def load(name):
        for ext in (".npy", ".png"):
            p = _os.path.join(obs_dir, name + ext)
            if not _os.path.exists(p):
                continue
            return np.load(p) if ext == ".npy" else np.asarray(Image.open(p).convert("RGB"))
        raise SystemExit(f"missing {name}.npy or {name}.png in {obs_dir}; pass --obs")

    return load("agentview"), load("wrist")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=str(pathlib.Path(_PI05_ROOT) / "obs/t00_init0"),
                    help="directory holding the sample observation (agentview and wrist, "
                         "224x224); used as the shape carrier the prefix is built from")
    ap.add_argument("--tokenizer", default=str(pathlib.Path(_PI05_ROOT) /
                    "demo_assets/paligemma_tokenizer.model"),
                    help="PaliGemma sentencepiece model")
    ap.add_argument("--policy-dir", default=f"{BASE}/models/pi05_libero_official_pt",
                    help="PyTorch checkpoint directory produced by the official converter")
    args = ap.parse_args()

    if not _os.path.exists(args.tokenizer):
        raise SystemExit(f"tokenizer not found: {args.tokenizer}\n"
                         "It ships with the official PaliGemma assets; pass --tokenizer "
                         "to point at your copy.")

    from openpi.policies import policy_config
    pol = policy_config.create_trained_policy(_config.get_config("pi05_libero"),
        args.policy_dir, default_prompt=None, pytorch_device="cpu")
    m = pol._model
    m.float()
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()

    sp_tok = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
    av, ws = load_views(args.obs)
    def to_cfgx(img):
        return torch.from_numpy((img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None])

    _os.makedirs(TARGET, exist_ok=True)

    AH = 10
    PL = 968
    for ti, text in enumerate(get_spatial_tasks()):
        ids, mask = tokenize(text, sp_tok)
        images = [to_cfgx(av), to_cfgx(ws), torch.zeros(1, 3, 224, 224)]
        img_masks = [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)]
        with torch.no_grad():
            embs, amask_full, pad_mask, _ = build_prefix(m, images, img_masks, ids, mask, AH)
        pad = torch.from_numpy(np.asarray(pad_mask).astype(bool))[None]  # (968,) -> (1,968)
        att = torch.zeros_like(pad)  # 官方 embed_prefix: 图像/语言 att 全 0(prefix-lm 双向)
        p_att_2d = make_att_2d_masks(pad, att)
        p_amask = m._prepare_attention_masks_4d(p_att_2d).float().clamp(min=-1e4)
        p_pids = torch.cumsum(pad, dim=1) - 1
        sp = torch.ones(1, AH, dtype=torch.bool)
        sa = torch.zeros(1, AH, dtype=torch.bool)
        sa[:, 0] = True
        p2d = pad[:, None, :].expand(1, AH, PL)
        s2d = make_att_2d_masks(sp, sa)
        full = torch.cat([p2d, s2d], dim=2)
        f4d = m._prepare_attention_masks_4d(full).float().clamp(min=-1e4)
        dummy = torch.zeros(1, 1)
        pm_rope = m.paligemma_with_expert.paligemma.model.language_model.rotary_emb
        em_rope = m.paligemma_with_expert.gemma_expert.model.rotary_emb
        p_cos, p_sin = pm_rope(dummy, p_pids)
        p_cos = p_cos.float(); p_sin = p_sin.float()
        po = torch.sum(pad, dim=-1)[:, None]
        pid = po + torch.arange(1, AH + 1, dtype=po.dtype) - 1
        s_cos, s_sin = em_rope(dummy, pid)
        s_cos = s_cos.float(); s_sin = s_sin.float()
        for nm, arr in (("p_amask", p_amask), ("f4d", f4d), ("p_cos", p_cos),
                        ("p_sin", p_sin), ("s_cos", s_cos), ("s_sin", s_sin)):
            np.save(f"{TARGET}/t{ti:02d}_{nm}.npy", arr.numpy())
        # Language half of the prefix: 200 slots after the 3 x 256 image tokens. Task
        # dependent, so it has to be baked into an asset; gen_assets_536.py trims it to
        # the live tokens and writes promptL_tXX.npy for the device.
        lang = np.asarray(embs)[0, 3 * 256:]
        np.save(f"{TARGET}/prompt200_t{ti:02d}.npy", lang)
        print(f"t{ti:02d} pad={int(pad.sum())} lang={lang.shape} -> {TARGET}/t{ti:02d}_*.npy",
              flush=True)
    print("PERTASK_ASSETS_OK")

if __name__ == "__main__":
    main()
