#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/usr/bin/env python3
"""export_d_kvseg.py — D 列官方结构分段导出(KV 分离架构,基于 C 列对齐):
  bmodel A(kvseg): prefix_embs+p_amask+p_cos+p_sin -> 36 个 prefix KV (1,1,968,256)
  bmodel B/C/D(dn 段): 36 KV + suffix_in + time + f4d + s_cos + s_sin -> suffix_out
  bmodel E(final 段): 同上 + final norm + action_out_proj -> v_t
层切分: kv 全 18 层; denoise 0-6/6-12/12-18(12_18 含 final)
自包含: 不需要任何项目内部中间产物 —— 段切分固定、alive 索引由结构推导、
trace 输入随机生成(见 main() 内注释)."""
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

import datetime
if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc
import sys
import types
import numpy as np
import torch
import torch.nn as nn

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
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models_pytorch import pi0_pytorch as _pp
_pp.get_safe_dtype = lambda t, d: torch.float32
from openpi.training import config as _config

PREFIX_LEN = 536   # H5: 968 -> 536(删 436 死 token + 为存活 533 的 t04 留补位槽)
AH = 10
ADIM = 32
NL = 18


class KVSeg(nn.Module):
    """prefix KV-cache 前向(官方 paligemma 层语义), 层 [s,e), 输出本段 k/v + hidden_out.
    s>0 段输入 prefix_hidden(前段输出); s==0 输入 prefix_embs."""

    def __init__(self, m, s=0, e=NL):
        super().__init__()
        self.m = m
        self.s = s
        self.e = e

    def forward(self, p_in, p_amask, p_cos, p_sin):
        import transformers.models.gemma.modeling_gemma as mg
        m = self.m
        pm = m.paligemma_with_expert.paligemma.model.language_model
        outs = []
        x = p_in
        for li in range(self.s, self.e):
            layer = pm.layers[li]
            residual = x
            h, gate = layer.input_layernorm(x, None)
            sa = layer.self_attn
            shp_q = (int(x.shape[0]), PREFIX_LEN, 8, 256)
            shp_kv = (int(x.shape[0]), PREFIX_LEN, 1, 256)
            q = sa.q_proj(h).view(shp_q).transpose(1, 2)
            k = sa.k_proj(h).view(shp_kv).transpose(1, 2)
            v = sa.v_proj(h).view(shp_kv).transpose(1, 2)
            q, k = mg.apply_rotary_pos_emb(q, k, p_cos, p_sin)
            att, _ = mg.eager_attention_forward(sa, q, k, v, p_amask, scaling=sa.scaling)
            att = att.reshape(int(x.shape[0]), PREFIX_LEN, 2048).contiguous()
            o = sa.o_proj(att)
            o = mg._gated_residual(residual, o, gate)
            after = o
            o, gate2 = layer.post_attention_layernorm(o, None)
            o = layer.mlp(o)
            x = mg._gated_residual(after, o, gate2)
            outs.append(k)
            outs.append(v)
        if self.e < NL:
            return tuple(outs) + (x.clone(),)  # clone 防 exporter 中间张量复用 bug
        return tuple(outs)


class DnSeg(nn.Module):
    """denoise 段(官方 GemmaDecoderLayer 语义): 层 [s,e), KV=prefix cache 拼接.
    输入: suffix_in(首段 x_t (1,10,32); 否则 (1,10,1024)), time, f4d, s_cos, s_sin,
          本段层的 prefix KV(p_k{s}..p_k{e-1}/p_v 交替), final 段输出 v_t.
    forward 签名按段动态生成(显式参数名)."""

    def __init__(self, m, s, e, final):
        super().__init__()
        self.m = m
        self.s = s
        self.e = e
        self.final = final
        self.first = (s == 0)
        # 动态生成显式 forward(本段 KV 参数命名 p_k{s+i}/p_v{s+i})
        nkv = e - s
        sig = ", ".join([f"p_k{s+i}, p_v{s+i}" for i in range(nkv)])
        body = f"def _fwd(self, suffix_in, time, f4d, s_cos, s_sin, {sig}): return self._impl(suffix_in, time, f4d, s_cos, s_sin, {sig})"
        ns = {}
        exec(body, ns)
        self._fwd = types.MethodType(ns["_fwd"], self)

    def forward(self, *args):
        return self._fwd(*args)

    def _impl(self, suffix_in, time, f4d, s_cos, s_sin, *pkv_seg):
        import transformers.models.gemma.modeling_gemma as mg
        m = self.m
        bsize = 1
        pwe = m.paligemma_with_expert
        em = pwe.gemma_expert.model
        self_nkv = self.e - self.s

        if self.first:
            AH2 = suffix_in.shape[1]
            flat = suffix_in.reshape(-1, suffix_in.shape[-1])
            se = m.action_in_proj(flat).reshape(bsize, AH2, -1)
        else:
            se = suffix_in
        te = _pp.create_sinusoidal_pos_embedding(time, 1024, min_period=4e-3, max_period=4.0, device=time.device).float()
        h = torch.nn.functional.silu(m.time_mlp_in(te))
        h = m.time_mlp_out(h)
        ad = torch.nn.functional.silu(h)

        out = se
        for li in range(self.s, self.e):
            layer = em.layers[li]
            residual = out
            h, gate = layer.input_layernorm(out, ad)
            sa = layer.self_attn
            q = sa.q_proj(h).view(1, AH, 8, 256).transpose(1, 2)
            k = sa.k_proj(h).view(1, AH, 1, 256).transpose(1, 2)
            v = sa.v_proj(h).view(1, AH, 1, 256).transpose(1, 2)
            q, k = mg.apply_rotary_pos_emb(q, k, s_cos, s_sin)
            pk = pkv_seg[(li - self.s) * 2].to(k.dtype)
            pv = pkv_seg[(li - self.s) * 2 + 1].to(v.dtype)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            att, _ = mg.eager_attention_forward(sa, q, k, v, f4d, scaling=sa.scaling)
            att = att.reshape(1, AH, 2048).contiguous()
            o = sa.o_proj(att)
            o = mg._gated_residual(residual, o, gate)
            after = o
            o, gate2 = layer.post_attention_layernorm(o, ad)
            o = layer.mlp(o)
            out = mg._gated_residual(after, o, gate2)

        if self.final:
            so2, _ = em.norm(out, ad)
            return m.action_out_proj(so2[:, -AH:].to(torch.float32))
        return out


def host_assets(m, prefix, pad, att, AHn=10):
    """host 预计算外置资产(与官方同函数)."""
    with torch.no_grad():
        p_att_2d = make_att_2d_masks(pad, att)
        p_amask = m._prepare_attention_masks_4d(p_att_2d)
        p_pids = torch.cumsum(pad, dim=1) - 1
        sl = AHn
        sp = torch.ones(1, sl, dtype=torch.bool)
        sa = torch.zeros(1, sl, dtype=torch.bool)
        sa[:, 0] = True
        p2d = pad[:, None, :].expand(1, sl, PREFIX_LEN)
        s2d = make_att_2d_masks(sp, sa)
        full = torch.cat([p2d, s2d], dim=2)
        f4d = m._prepare_attention_masks_4d(full)
        # BF16 安全: mask 大负值 clamp 到 -10000(softmax 等价, 防 BF16 溢出 NaN)
        p_amask = p_amask.float().clamp(min=-1e4)
        f4d = f4d.float().clamp(min=-1e4)
        dummy = torch.zeros(1, 1)
        pm_rope = m.paligemma_with_expert.paligemma.model.language_model.rotary_emb
        em_rope = m.paligemma_with_expert.gemma_expert.model.rotary_emb
        p_cos, p_sin = pm_rope(dummy, p_pids)
        po = torch.sum(pad, dim=-1)[:, None]
        pid = po + torch.arange(1, sl + 1, dtype=po.dtype, device=po.device) - 1
        s_cos, s_sin = em_rope(dummy, pid)
        return (p_amask, f4d, p_cos.float(), p_sin.float(),
                s_cos.float(), s_sin.float())


def main():
    from openpi.policies import policy_config
    pol = policy_config.create_trained_policy(_config.get_config("pi05_libero"),
        f"{BASE}/models/pi05_libero_official_pt", default_prompt=None, pytorch_device="cpu")
    m = pol._model
    m.float()
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()

    # H5 keeps only the "alive" prefix tokens. The dead range is fully structural:
    #   alive = [0..511]   two real camera views (2 x 256 tokens)
    #         + [768..787] the 20 real language tokens
    #   dead  = [512..767] third (zero-filled) camera view, 256 tokens
    #         + [788..967] language padding, 180 tokens   -> 436 in total
    # Per-task alive counts range 528..533; the model is built for a fixed 536 and the
    # spare slots are masked out at runtime by p_amask.
    alive = np.concatenate([np.arange(0, 512), np.arange(768, 788),
                            np.arange(512, 516)]).astype(np.int64)
    assert len(alive) == PREFIX_LEN, (len(alive), PREFIX_LEN)

    # The prefix embedding and x_t below only have to carry the right shape and dtype:
    # W8BF16 quantizes weights only and ddn stays BF16, so trace input values do not
    # influence the compiled weights. Generating them keeps this script self-contained.
    rng = np.random.default_rng(7)
    prefix = torch.from_numpy(
        rng.standard_normal((1, PREFIX_LEN, 2048)).astype(np.float32))
    # 532 个 token 全部"存在", 且 prefix 注意力双向全可见 —— 与设备实际 mask 的活区完全一致
    pad = torch.ones(1, PREFIX_LEN, dtype=torch.bool)
    att = torch.zeros(1, PREFIX_LEN, dtype=torch.bool)
    print(f"[H5] prefix -> {tuple(prefix.shape)}, alive={len(alive)}")
    x_t = torch.from_numpy(rng.standard_normal((1, 10, 32)).astype(np.float32))
    time = torch.from_numpy(np.array([0.5], np.float32))

    # No C-column numerical baseline here: that reference takes a 968-token prefix and is
    # therefore not comparable at 536. Fidelity is checked on device instead.
    p_amask, f4d, p_cos, p_sin, s_cos, s_sin = host_assets(m, prefix, pad, att)

    with torch.no_grad():
        kv = KVSeg(m)(prefix, p_amask, p_cos, p_sin)
        segs = [DnSeg(m, 0, 6, False), DnSeg(m, 6, 12, False), DnSeg(m, 12, 18, True)]
        cur = x_t
        for i, sg in enumerate(segs):
            cur = sg(cur, time, f4d, s_cos, s_sin, *kv[sg.s * 2: sg.e * 2])
            print(f"seg{i}: out {tuple(cur.shape)}")
    v_d = cur.numpy()
    print("H5 v_t[:4]:", np.round(v_d[0, 0, :4], 5).tolist())
    print("H5_CHAIN_TRACED")

    if "--export" in sys.argv:
        import os
        OUTD = _ONNX_DIR
        os.makedirs(OUTD, exist_ok=True)
        with torch.no_grad():
            kv = KVSeg(m)(prefix, p_amask, p_cos, p_sin)
            kv_names = []
            for li in range(NL):
                kv_names += [f"p_k{li}", f"p_v{li}"]
            # dkv 拆两段(0-9/9-18): 段0 输出 18 KV + hidden; 段1 输入 hidden + 18 KV
            ks0 = KVSeg(m, 0, 9)
            ks1 = KVSeg(m, 9, 18)
            out0 = ks0(prefix, p_amask, p_cos, p_sin)
            hidden9 = out0[-1]
            kv0_names = [n for li in range(0, 9) for n in (f"p_k{li}", f"p_v{li}")] + ["p_hidden"]  # 交错序与输出一致
            torch.onnx.export(ks0.eval(), (prefix, p_amask, p_cos, p_sin),
                              f"{OUTD}/pi05_dkv0_9.onnx",
                              input_names=["prefix_embs", "p_amask", "p_cos", "p_sin"],
                              output_names=kv0_names, opset_version=18,
                              external_data=True, dynamo=True)
            print("exported dkv0_9.onnx -> 18kv+hidden")
            out1 = ks1(hidden9, p_amask, p_cos, p_sin)
            kv1_names = [n for li in range(9, 18) for n in (f"p_k{li}", f"p_v{li}")]  # 交错序
            torch.onnx.export(ks1.eval(), (hidden9, p_amask, p_cos, p_sin),
                              f"{OUTD}/pi05_dkv9_18.onnx",
                              input_names=["prefix_hidden", "p_amask", "p_cos", "p_sin"],
                              output_names=kv1_names, opset_version=18,
                              external_data=True, dynamo=True)
            print("exported dkv9_18.onnx -> 18kv")
            # 36 KV 全集(链式拼接供 ddn 段导出)
            kv = tuple(list(out0[:-1]) + list(out1))
            # denoise 段: 用段边界真实输入
            bounds = [x_t.numpy()]
            cur2 = x_t
            for sg in segs:
                cur2 = sg(cur2, time, f4d, s_cos, s_sin, *kv[sg.s * 2: sg.e * 2])
                bounds.append(cur2.numpy())
            dnames = [("pi05_ddn0_6", "suffix_out"), ("pi05_ddn6_12", "suffix_out"), ("pi05_ddn12_18", "v_t")]
            in_base = ["suffix_in", "time", "f4d", "s_cos", "s_sin"] + kv_names
            for i, sg in enumerate(segs):
                nm, outn = dnames[i]
                sin = torch.from_numpy(bounds[i])
                torch.onnx.export(sg, (sin, time, f4d, s_cos, s_sin, *kv[sg.s * 2: sg.e * 2]),
                                  f"{OUTD}/{nm}.onnx", input_names=in_base[:5] + kv_names[sg.s * 2: sg.e * 2],
                                  output_names=[outn], opset_version=17,
                                  external_data=True, dynamo=False)
                print(f"exported {nm}.onnx -> {outn}")
        print("EXPORT_KVSEP_DONE")


if __name__ == "__main__":
    main()
