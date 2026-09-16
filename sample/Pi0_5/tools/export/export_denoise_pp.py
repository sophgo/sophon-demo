#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""Export the REAL pi0.5 full denoise step (18+18 dual-path layers) as per-path
rewritten ONNX: NO seq-concat shared attention, NO GQA expand op.

Fixes applied (verified on 3-layer repro, F16/BF16 cos=1.000000):
  1. per-path: prefix Q only attends shared K/V via its OWN MHA sub-graph
     (attn row-wise independent), suffix Q likewise. Semantics identical to
     openpi's seq-concat shared attention, but each path is a standard MHA.
  2. GQA weight-repeat: K/V projections are repeat-tiled to 8 heads so the
     graph has zero Expand/Tile (TPU-MLIR bugs the expand->RoPE path in F16).

Weights: real pi0.5 safetensors. Inputs: denoise_input.npz (cond [1,1024],
prefix_embs [1,264,2048], noisy_actions [1,50,32]). Full single-step forward:
action_in_proj -> action chunk -> action_out_proj(full suffix tail) -> v_t.

Usage: python export_denoise_pp.py [--tag denoise_pp] [--layers 18] [--out ...]
"""
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

import os
import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch

BASE = pathlib.Path(_PI05_ROOT)
OUT = BASE / "vlm" / "denoise_pp"
sys.path.insert(0, str(BASE / "openpi/src"))

from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks  # noqa: E402

H, HD = 8, 256
MASK_VAL = float(-1e4)  # -2.38e38 breaks BF16 exp kernel


def _gated_residual(x, y, gate):
    if gate is None:
        return x + y
    return x + y * gate

TIME_HORIZON = 50
Wp = 2048


class _Cfg:
    pi05 = True
    dtype = "float32"
    paligemma_variant = "gemma_2b"
    action_expert_variant = "gemma_300m"
    action_dim = 32
    action_horizon = 50
    max_token_len = 200
    pytorch_compile_mode = None


def rms1(x, eps=1e-6):
    xf = x.float()
    var = torch.mean(xf * xf, dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + float(eps))).to(x.dtype)


class PPBlock(nn.Module):
    """One dual-path layer: prefix (paligemma 2B) stream + expert stream share
    K/V over merged seq, computed as two independent standard MHA sub-graphs."""

    def __init__(self, pwe, gi):
        super().__init__()
        pp = f"paligemma.model.language_model.layers.{gi}"
        ep = f"gemma_expert.model.layers.{gi}"
        self.lp = pwe.paligemma.model.language_model.layers[gi]
        self.le = pwe.gemma_expert.model.layers[gi]
        self.nl_p1 = self.lp.input_layernorm
        self.nl_p2 = self.lp.post_attention_layernorm
        self.nl_e1 = self.le.input_layernorm
        self.nl_e2 = self.le.post_attention_layernorm
        ap = self.lp.self_attn
        ae = self.le.self_attn
        self.qp, self.kp, self.vp, self.op = ap.q_proj, ap.k_proj, ap.v_proj, ap.o_proj
        self.qe, self.ke, self.ve, self.oe = ae.q_proj, ae.k_proj, ae.v_proj, ae.o_proj
        # GQA: k/v have KV=1 head; tile weight to H heads (no Expand op)
        self.kp8 = self._repeat(self.kp)
        self.vp8 = self._repeat(self.vp)
        self.ke8 = self._repeat(self.ke)
        self.ve8 = self._repeat(self.ve)
        self.scale = HD ** -0.5

    def _repeat(self, l):
        w = l.weight.data  # [KV*HD, In]
        ll = nn.Linear(w.shape[1], w.shape[0] * H)
        ll.weight.data = w.repeat(H, 1)
        ll.bias = None
        return ll

    def _qkv(self, x, q, k, v):
        q = q(x).reshape(1, -1, H, HD).transpose(1, 2)
        k = k(x).reshape(1, -1, H, HD).transpose(1, 2)
        v = v(x).reshape(1, -1, H, HD).transpose(1, 2)
        return q, k, v

    def _rot(self, q, k, cos, sin):
        rq = q * cos + torch.cat([-q[..., HD // 2:], q[..., :HD // 2]], -1) * sin
        rk = k * cos + torch.cat([-k[..., HD // 2:], k[..., :HD // 2]], -1) * sin
        return rq, rk

    def forward(self, xp, xe, cond, cos, sin, amask, P, S):
        hp, g1p = self.nl_p1(xp, cond=None)
        he, g1e = self.nl_e1(xe, cond=cond)
        qp, kp, vp = self._qkv(hp, self.qp, self.kp8, self.vp8)
        qe, ke, ve = self._qkv(he, self.qe, self.ke8, self.ve8)
        qp, kp = self._rot(qp, kp, cos[:, :, :P], sin[:, :, :P])
        qe, ke = self._rot(qe, ke, cos[:, :, P:], sin[:, :, P:])
        k = torch.cat([kp, ke], 2)
        v = torch.cat([vp, ve], 2)
        # per-path independent MHA
        o_p = self._mha(qp, k, v, amask[:, :, :P, :])
        o_e = self._mha(qe, k, v, amask[:, :, P:, :])
        xp = _gated_residual(xp, self.op(o_p), g1p)
        xe = _gated_residual(xe, self.oe(o_e), g1e)
        after = xe.clone()
        pn2, pg2 = self.nl_p2(xp, cond=None)
        xp = _gated_residual(xp, self.lp.mlp(pn2), pg2)
        en2, eg2 = self.nl_e2(after, cond=cond)
        xe = _gated_residual(after, self.le.mlp(en2), eg2)
        return xp, xe

    def _mha(self, q, k, v, mmask):
        dtype = q.dtype
        # 对齐 JAX 官方语义: score matmul fp32 累积 + softmax fp32 + av fp32 累积, 输出 cast 回 dtype(bf16)
        att = (q.float() * self.scale) @ k.float().transpose(-2, -1) + mmask.float()
        m = torch.amax(att, dim=-1, keepdim=True)
        e = torch.exp(att - m)
        att = e / torch.sum(e, dim=-1, keepdim=True)
        return (att.float() @ v.float()).transpose(1, 2).reshape(1, q.shape[2], H * HD).to(dtype)


class DenoisePP(nn.Module):
    """Full single denoise step with per-path 18+18 dual layers."""

    def __init__(self, pi0, n_layers=18):
        super().__init__()
        pwe = pi0.paligemma_with_expert
        self.blocks = nn.ModuleList([PPBlock(pwe, i) for i in range(n_layers)])
        self.fin_p = pwe.paligemma.model.language_model.norm
        self.fin_e = pwe.gemma_expert.model.norm
        self.action_in_proj = pi0.action_in_proj
        self.action_out_proj = pi0.action_out_proj

    def _rope(self, x, L, pos=None):
        inv = 1.0 / (10000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
        if pos is not None:
            ang = pos[:, None] * inv[None, :]
        else:
            ang = torch.arange(L, dtype=torch.float32)[:, None] * inv[None, :]
        cos = torch.cos(ang).transpose(0, 1).reshape(1, 1, HD // 2, L).permute(0, 1, 3, 2)
        sin = torch.sin(ang).transpose(0, 1).reshape(1, 1, HD // 2, L).permute(0, 1, 3, 2)
        cos = torch.cat([cos, cos], -1).to(x.dtype)
        sin = torch.cat([sin, sin], -1).to(x.dtype)
        return cos, sin

    def forward(self, xp, xe, cond, amask, pos=None, cos_in=None, sin_in=None):
        P = xp.shape[1]
        S = xe.shape[1]
        L = P + S
        if cos_in is not None and sin_in is not None:
            cos, sin = cos_in, sin_in
        elif pos is not None:
            cos, sin = self._rope(xp, L, pos=pos)
        else:
            # JAX 语义位置(固定结构): prefix cumsum(prefix_mask)-1 + suffix sum+arange
            PIX, GAP, PRM = 512, 256, 17
            pm = [1] * PIX + [0] * GAP + [1] * PRM + [0] * (P - PIX - GAP - PRM)
            pos_arr = np.cumsum(pm) - 1
            pos_arr = np.concatenate([pos_arr, np.arange(pos_arr[-1] + 1, pos_arr[-1] + 1 + S)])
            cos, sin = self._rope(xp, L, pos=torch.from_numpy(pos_arr).float())
        for b in self.blocks:
            xp, xe = b(xp, xe, cond, cos, sin, amask, P, S)
        xp = self.fin_p(xp, cond=None)[0]
        xe = self.fin_e(xe, cond=cond)[0]
        v = self.action_out_proj(xe[:, -TIME_HORIZON:])
        return v


def build_att4d(pi0, S_p, S_s):
    """JAX 语义 amask: prefix 行看图像+prompt(无 action); action 行看图像+prompt+action.
    间隙(512..767)与尾(785..967)是 mask=False(不可见). 输出 big_neg=-2.38e38."""
    PIX, GAP, PRM = 512, 256, 17
    L = S_p + S_s
    vis = np.zeros((1, L, L), dtype=bool)
    vis[:, :S_p, :PIX] = True
    vis[:, :S_p, PIX + GAP:PIX + GAP + PRM] = True
    vis[:, S_p:, :PIX] = True
    vis[:, S_p:, PIX + GAP:PIX + GAP + PRM] = True
    vis[:, S_p:, S_p:] = True
    return pi0._prepare_attention_masks_4d(torch.from_numpy(vis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="denoise_pp")
    ap.add_argument("--layers", type=int, default=18)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    model = PI0Pytorch(_Cfg())
    SD_PATH = os.environ.get("PI05_SD", str(BASE / "models/pi05_base_pytorch/model.safetensors"))
    sd = safetensors.torch.load_file(SD_PATH)
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dn = DenoisePP(model, a.layers)
    z = np.load(str(BASE / "vlm/denoise/denoise_input.npz"))
    cond = torch.from_numpy(z["cond"].astype(np.float32))
    xp = torch.from_numpy(z["prefix_embs"].astype(np.float32))
    noise = torch.from_numpy(z["noisy_actions"].astype(np.float32))
    with torch.no_grad():
        xe = dn.action_in_proj(noise)
    amask = build_att4d(model, xp.shape[1], noise.shape[1])
    S_all = xp.shape[1] + noise.shape[1]
    # JAX prefix_mask 真实结构: 512图像+256间隙+17prompt@768+183尾 -> cumsum 逐 token 位置
    PIX, GAP, PRM = 512, 256, 17
    _pm = [1] * PIX + [0] * GAP + [1] * PRM + [0] * (xp.shape[1] - PIX - GAP - PRM)
    _pos_prefix = torch.from_numpy(np.cumsum(_pm) - 1).float()
    _pos_suffix = torch.arange(_pos_prefix[-1].item() + 1 if _pos_prefix.numel() else 0,
                               _pos_prefix[-1].item() + 1 + noise.shape[1], dtype=torch.float32)
    pid = torch.cat([_pos_prefix, _pos_suffix])

    with torch.no_grad():
        v = dn(xp, xe, cond, amask, pos=pid)
    print("v_t:", v.shape, "finite:", torch.isfinite(v).all().item(),
          "min", float(v.min()), "max", float(v.max()))

    tag = a.tag + (f"_l{a.layers}" if a.layers != 18 else "")
    out_p = OUT / a.out if a.out else OUT
    np.savez(out_p / f"{tag}_input.npz", xp=xp.numpy(), xe=xe.numpy(), cond=cond.numpy(), amask=amask.numpy())
    np.savez(out_p / f"{tag}_ref.npz", v_t=v.float().numpy())

    torch.onnx.export(dn, (xp, xe, cond, amask), out_p / f"{tag}.onnx",
                      input_names=["xp", "xe", "cond", "amask"], output_names=["v_t"],
                      opset_version=17, external_data=True, dynamo=True)
    print("onnx ->", out_p / f"{tag}.onnx")


if __name__ == "__main__":
    main()