#!/usr/bin/env python3
"""Export TAPNext++ per-step networks to ONNX for Sophon BM1688.

TAPNext++ is a recurrent (RG-LRU / linear-SSM) point tracker. Its online
inference processes one frame at a time and carries an explicit state
(`TAPNextTrackingState`) between frames. We therefore export TWO static-shape
ONNX graphs that share the same weights:

  * ``tapnext_init.onnx``  — first frame. Inputs: frame + query_points.
                             Outputs: tracks, vis, and the initial recurrent
                             state (flattened to tensors).
  * ``tapnext_step.onnx``  — subsequent frames. Inputs: frame + the flattened
                             state. Outputs: tracks, vis, updated state.

The host (Python / C++ on BM1688) runs the recurrence loop: init graph for
frame 0, step graph for frames 1..N, feeding the state tensors back each step.

Shapes (POC, 256x256, Q=1):
  frame         : [1, 3, 256, 256]  float32, NCHW, range [-1, 1]
  query_points  : [1, Q, 3]         float32, layout [t, y, x] in model pixels
  step          : [1]               float32 (frame counter; INPUT only)
  tracks        : [1, 1, Q, 2]      float32, [y, x] in model pixels
  vis_logits    : [1, 1, Q, 1]      float32
  rg_lru_{i}    : [1025, 768]       float32  (i = 0..11, 12 blocks)
  conv1d_{i}    : [1025, 3, 768]    float32

  init graph:  2 inputs  (frame, query_points)
               26 outputs (tracks, vis_logits, new_rg_lru_0, new_conv1d_0, ...)
  step graph: 27 inputs  (frame, step, query_points, rg_lru_0, conv1d_0, ...)
               26 outputs (tracks, vis_logits, new_rg_lru_0, new_conv1d_0, ...)

``step`` and ``query_points`` are step-graph INPUTS only (the model uses step to
time-shift query points, and query_points to build the query embedding). They
are NOT echoed as outputs: step is a frame counter the host increments, and
query_points is constant across the video the host already holds.

Why torch 1.13 + MHA monkey-patch: the ViT blocks use torchvision's
``EncoderBlock`` which wraps ``nn.MultiheadAttention``. On torch >= 2.0 MHA
routes through ``aten::scaled_dot_product_attention`` (no ONNX symbolic). On
torch 1.13 MHA has a fused fast path -> ``aten::_native_multi_head_attention``
which ALSO has no ONNX symbolic at any opset, and the EncoderBlock hits every
fast-path gate in eval mode. We therefore monkey-patch
``nn.MultiheadAttention.forward`` onto its explicit MatMul/Softmax/MatMul path
(see ``_mha_forward_exportable`` below) -- same math/weights, exportable ops.

Run (inside the torch-1.13 export venv):
  python tools/export_onnx.py \
      --ckpt models/tapnextpp_ckpt.pt \
      --out-dir models/onnx \
      --model-size 256 --num-queries 1 --validate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as _F

# Vendor the minimal pure-torch tapnext source so the export is self-contained.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tapnext_src"))

from tapnet.tapnext import tapnext_torch  # noqa: E402
from tapnet.tapnext.tapnext_lru_modules import RecurrentBlockCache  # noqa: E402


# ---------------------------------------------------------------------------
# Force nn.MultiheadAttention onto its explicit (exportable) path.
#
# torch 1.13's MultiheadAttention.forward has a fused fast path that calls
# aten::_native_multi_head_attention whenever: batched + self-attention + eval
# + batch_first + dropout==0 + no attn_mask + even num_heads. The torchvision
# EncoderBlock used by TAPNext satisfies ALL of these, so it always hits the
# fused kernel -- which has NO ONNX symbolic at any opset (the export dies with
# "Exporting the operator 'aten::_native_multi_head_attention' ... is not
# supported"). need_weights is NOT part of the fast-path gate, so flipping it
# does not help.
#
# Fix: replace forward with the exact slow-path body from torch 1.13 (the code
# after the `if not why_not_fast_path:` block), which routes through
# F.multi_head_attention_forward -> MatMul/Softmax/MatMul. Identical math and
# weights (in_proj_weight/in_proj_bias/out_proj); only the dispatch changes.
# ---------------------------------------------------------------------------
def _mha_forward_exportable(
    self, query, key, value, key_padding_mask=None, need_weights=False,
    attn_mask=None, average_attn_weights=True,
):
    is_batched = query.dim() == 3
    if self.batch_first and is_batched:
        if key is value:
            if query is key:
                query = key = value = query.transpose(1, 0)
            else:
                query, key = [x.transpose(1, 0) for x in (query, key)]
                value = key
        else:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
    attn_output, attn_output_weights = _F.multi_head_attention_forward(
        query, key, value, self.embed_dim, self.num_heads,
        self.in_proj_weight, self.in_proj_bias,
        self.bias_k, self.bias_v, self.add_zero_attn,
        self.dropout, self.out_proj.weight, self.out_proj.bias,
        training=self.training,
        key_padding_mask=key_padding_mask, need_weights=need_weights,
        attn_mask=attn_mask, average_attn_weights=average_attn_weights,
    )
    if self.batch_first and is_batched:
        return attn_output.transpose(1, 0), attn_output_weights
    return attn_output, attn_output_weights


nn.MultiheadAttention.forward = _mha_forward_exportable

N_BLOCKS = 12  # TAPNext default depth


# ---------------------------------------------------------------------------
# State flattening  (TAPNextTrackingState <-> flat tuple of tensors)
# ---------------------------------------------------------------------------
def cache_input_names() -> list[str]:
    names = []
    for i in range(N_BLOCKS):
        names.append(f"rg_lru_{i}")
        names.append(f"conv1d_{i}")
    return names


def state_in_names() -> list[str]:
    """Flattened state as *inputs* to the step graph: step, query_points, caches."""
    return ["step", "query_points"] + cache_input_names()


def state_out_names() -> list[str]:
    """Cache tensors as *outputs*: new_* prefix so they never collide with the
    step-graph input names (torch.onnx.export otherwise renames inputs to
    ``rg_lru_0.1`` etc. on a name clash).

    Note: ``step`` and ``query_points`` are NOT echoed back. ``step`` is a frame
    counter (host increments it) and ``query_points`` is constant across the
    video (host keeps the init value). Echoing them produced pass-through
    Cast/Constant outputs that the ONNX->MLIR converter could not resolve
    (``operand new_query_points not found``); dropping them also shrinks I/O."""
    return [f"new_{n}" for n in cache_input_names()]


def flatten_caches(state) -> tuple:
    """TAPNextTrackingState -> (rg_lru_0, conv1d_0, rg_lru_1, conv1d_1, ...).
    Only the 24 recurrent cache tensors -- the graph outputs."""
    out = []
    for h in state.hidden_state:
        out.append(h.rg_lru_state.float())
        out.append(h.conv1d_state.float())
    return tuple(out)


def flatten_state(state) -> tuple:
    """TAPNextTrackingState -> (step, query_points, rg_lru_0, conv1d_0, ...).
    Full state as *inputs* to the step graph (step + query_points are used
    inside the model: step time-shifts query points, query_points feeds the
    query embedding)."""
    step_t = torch.as_tensor(state.step).reshape(1).float()
    out = [step_t, state.query_points.float()]
    for h in state.hidden_state:
        out.append(h.rg_lru_state.float())
        out.append(h.conv1d_state.float())
    return tuple(out)


def unflatten_state(step, query_points, caches: tuple):
    """Reconstruct TAPNextTrackingState from flat tensors."""
    hidden = []
    for i in range(N_BLOCKS):
        hidden.append(
            RecurrentBlockCache(
                rg_lru_state=caches[2 * i],
                conv1d_state=caches[2 * i + 1],
            )
        )
    return tapnext_torch.TAPNextTrackingState(
        step=step, query_points=query_points, hidden_state=hidden
    )


# ---------------------------------------------------------------------------
# Export wrappers  (NCHW frame input, flat state I/O)
# ---------------------------------------------------------------------------
class InitWrapper(nn.Module):
    """First-frame graph: frame + query_points -> tracks, vis, state."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frame, query_points):
        # frame: [1,3,H,W] NCHW -> video: [1,1,H,W,3] NHWC
        video = frame.permute(0, 2, 3, 1).unsqueeze(0)
        tracks, _track_logits, vis_logits, state = self.model(
            video=video, query_points=query_points, state=None
        )
        return (tracks, vis_logits) + flatten_caches(state)


class StepWrapper(nn.Module):
    """Per-frame graph: frame + state -> tracks, vis, new state."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frame, step, query_points, *caches):
        video = frame.permute(0, 2, 3, 1).unsqueeze(0)
        state = unflatten_state(step, query_points, caches)
        tracks, _track_logits, vis_logits, new_state = self.model(
            video=video, state=state
        )
        return (tracks, vis_logits) + flatten_caches(new_state)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, model_size: int) -> tapnext_torch.TAPNext:
    inner = tapnext_torch.TAPNext(image_size=(model_size, model_size))
    # weights_only=False: the official Google DeepMind checkpoint (2.4 GB) contains
    # pickle objects that torch 1.13's restrictive weights_only unpickler rejects
    # ("Unsupported operand 71"). Trusted source -> safe to full-unpickle.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.removeprefix("tapnext."): v for k, v in state_dict.items()}
    missing, unexpected = inner.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:3]} ...")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
    inner.eval()
    return inner


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_graph(model, wrapper, args_tuple, input_names, output_names, out_path, opset=16):
    print(f"[export] {out_path}  inputs={len(input_names)}  outputs={len(output_names)}  opset={opset}")
    torch.onnx.export(
        wrapper,
        args_tuple,
        str(out_path),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        dynamic_axes=None,  # static shapes (POC)
    )
    # simplify
    import onnx  # imported here so the onnxsim branch below can call onnx.save
    try:
        import onnxsim
        onnx_model, ok = onnxsim.simplify(str(out_path), overwrite_input_shapes=None)
        if ok:
            onnx.save(onnx_model, str(out_path))
            print(f"[export] simplified OK")
        else:
            print(f"[export] onnxsim returned not-ok, keeping unsimplified")
    except Exception as e:
        print(f"[export] onnxsim skipped: {e}")
    m = onnx.load(str(out_path))
    print(f"[export] {m.ir_version=} opset={[o.version for o in m.opset_import]}")
    return out_path


# ---------------------------------------------------------------------------
# Validation  (ONNX Runtime vs PyTorch reference)
# ---------------------------------------------------------------------------
def _cmp(name, a, b, atol):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"  [FAIL] {name}: shape {a.shape} vs {b.shape}")
        return False
    diff = np.abs(a - b)
    ok = bool(np.all(diff <= atol))
    tag = "ok" if ok else "FAIL"
    print(f"  [{tag}] {name}: max|d|={diff.max():.3e}  shape={a.shape}")
    return ok


def validate(ckpt_path, model_size, n_queries, onnx_dir):
    import onnxruntime as ort

    model = load_model(ckpt_path, model_size)
    H = model_size
    Q = n_queries
    # dummy frame in [-1,1], NCHW
    frame0 = (torch.rand(1, 3, H, H) * 2 - 1).float()
    frame1 = (torch.rand(1, 3, H, H) * 2 - 1).float()
    # query points [1, Q, 3] = [t=0, y, x] in model pixels (center-ish)
    qp = torch.zeros(1, Q, 3).float()
    qp[..., 0] = 0.0
    qp[..., 1] = float(H // 2)
    qp[..., 2] = float(H // 2)

    # --- PyTorch reference ---
    with torch.no_grad():
        v0 = frame0.permute(0, 2, 3, 1).unsqueeze(0)
        ref_tracks0, _, ref_vis0, ref_state0 = model(video=v0, query_points=qp, state=None)
        caches0 = flatten_caches(ref_state0)  # 24 cache tensors (graph outputs)
        in0 = flatten_state(ref_state0)       # (step, query_points, *caches) -- step-graph inputs
        v1 = frame1.permute(0, 2, 3, 1).unsqueeze(0)
        ref_tracks1, _, ref_vis1, ref_state1 = model(video=v1, state=ref_state0)
        caches1 = flatten_caches(ref_state1)

    out_names = ["tracks", "vis_logits"] + state_out_names()  # 2 + 24 = 26

    # --- validate init graph ---
    print("[validate] init graph")
    sess = ort.InferenceSession(str(onnx_dir / "tapnext_init.onnx"))
    feeds = {"frame": frame0.numpy(), "query_points": qp.numpy()}
    ort_out = sess.run(None, feeds)
    ok_all = True
    ok_all &= _cmp("tracks", ort_out[0], ref_tracks0.numpy(), 1e-3)
    ok_all &= _cmp("vis_logits", ort_out[1], ref_vis0.numpy(), 1e-3)
    for j, nm in enumerate(out_names[2:]):
        ok_all &= _cmp(nm, ort_out[2 + j], caches0[j].numpy(), 1e-3)

    # --- validate step graph ---
    print("[validate] step graph")
    sess2 = ort.InferenceSession(str(onnx_dir / "tapnext_step.onnx"))
    feeds2 = {
        "frame": frame1.numpy(),
        "step": in0[0].numpy(),
        "query_points": in0[1].numpy(),
    }
    for i, nm in enumerate(cache_input_names()):
        feeds2[nm] = in0[2 + i].numpy()
    ort_out2 = sess2.run(None, feeds2)
    ok_all &= _cmp("tracks", ort_out2[0], ref_tracks1.numpy(), 1e-3)
    ok_all &= _cmp("vis_logits", ort_out2[1], ref_vis1.numpy(), 1e-3)
    for j, nm in enumerate(out_names[2:]):
        ok_all &= _cmp(nm, ort_out2[2 + j], caches1[j].numpy(), 1e-3)

    print(f"[validate] {'ALL PASS' if ok_all else 'MISMATCH'}")
    return ok_all


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/tapnextpp_ckpt.pt")
    ap.add_argument("--out-dir", default="models/onnx")
    ap.add_argument("--model-size", type=int, default=256)
    ap.add_argument("--num-queries", type=int, default=1)
    ap.add_argument("--opset", type=int, default=16)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, args.model_size)
    H = args.model_size
    Q = args.num_queries

    # dummy inputs
    frame = (torch.rand(1, 3, H, H) * 2 - 1).float()
    qp = torch.zeros(1, Q, 3).float()
    qp[..., 1] = float(H // 2)
    qp[..., 2] = float(H // 2)

    # --- init graph ---
    init_wrap = InitWrapper(model)
    init_wrap.eval()
    with torch.no_grad():
        # run once to get real state shapes for the step-graph dummy inputs
        v = frame.permute(0, 2, 3, 1).unsqueeze(0)
        _, _, _, ref_state = model(video=v, query_points=qp, state=None)
        flat = flatten_state(ref_state)
    init_in = ["frame", "query_points"]
    init_out = ["tracks", "vis_logits"] + state_out_names()
    with torch.no_grad():
        export_graph(model, init_wrap, (frame, qp), init_in, init_out,
                     out_dir / "tapnext_init.onnx", opset=args.opset)

    # --- step graph ---
    step_wrap = StepWrapper(model)
    step_wrap.eval()
    frame2 = (torch.rand(1, 3, H, H) * 2 - 1).float()
    step_in = ["frame"] + state_in_names()
    step_out = init_out
    step_args = (frame2, flat[0], flat[1], *flat[2:])
    with torch.no_grad():
        export_graph(model, step_wrap, step_args, step_in, step_out,
                     out_dir / "tapnext_step.onnx", opset=args.opset)

    print(f"\n[done] ONNX in {out_dir}/")
    if args.validate:
        ok = validate(args.ckpt, args.model_size, args.num_queries, out_dir)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
