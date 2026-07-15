#!/usr/bin/env python3
# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
"""
SAM3 source-side reference runner for the consistency harness.

Loads the official sam3 PyTorch model at 504 resolution (FP32, CPU) and runs
the full ``forward_grounding`` path through ``Sam3Processor``, capturing every
intermediate stage via forward hooks.  The dumped tensors are diffed against
the TPU pipeline's corresponding stages by ``consistency_harness.py``.

Stage capture points (names match the harness):
  preprocess_in : input tensor to the ViT trunk (post processor preprocess)
  vit_out       : ViT trunk output (pre-neck FPN)
  neck_fpn      : backbone_out["backbone_fpn"]  (post-scalp FPN list)
  text_feats    : backbone_out["language_features"] / ["language_mask"]
  gnd_enc       : transformer.encoder output (memory)
  gnd_dec       : transformer.decoder output (hs, reference_boxes, presence)
  scoring       : dot_prod_scoring output
  mask_head     : segmentation_head output
  e2d           : forward_grounding pred_boxes / pred_logits / presence

Usage:
  python sam3_source_ref.py --image ../datasets/truck.jpg --prompt "a truck"
  python sam3_source_ref.py --dump ref.npz
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("Sam3SourceRef")

# sam3 source repo root (~/work/git_commits/developer/sam3).  The package is
# usually pip-installed editable, but add the repo root defensively so imports
# resolve even without an install.
_SAM3_REPO = os.path.expanduser("~/work/git_commits/developer/sam3")
if _SAM3_REPO not in sys.path:
    sys.path.insert(0, _SAM3_REPO)

# sophon-demo sample dir (for reusing patches from sam3_infer.py)
_SAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PY_DIR = os.path.join(_SAMPLE_DIR, "python")
if _PY_DIR not in sys.path:
    sys.path.insert(0, _PY_DIR)


# ============================================================
# CPU / checkpoint patches (extends sam3_infer._patch_sam3_model)
# ============================================================

def _patch_sam3_for_cpu_ref():
    """Patch sam3 for CPU reference + disable activation checkpointing.

    Reuses sam3_infer._patch_sam3_model (Mlp, PositionEmbeddingSine, pin_memory)
    then force-disables activation checkpointing so forward hooks fire exactly
    once on the real (non-recomputed) path.
    """
    from sam3_infer import _patch_sam3_model  # noqa: F401 (idempotent)
    _patch_sam3_model()

    import torch
    import sam3.model.model_misc as mm

    # Disable whole-backbone activation checkpointing at runtime.
    def _disable_act_ckpt(model):
        if hasattr(model, "use_act_checkpoint"):
            model.use_act_checkpoint = False
        bb = getattr(model, "backbone", None)
        if bb is not None:
            for flag in ("act_ckpt_whole_vision_backbone",
                         "act_ckpt_whole_language_backbone"):
                if hasattr(bb, flag):
                    setattr(bb, flag, False)
            vb = getattr(bb, "vision_backbone", None)
            if vb is not None and hasattr(vb, "use_act_checkpoint"):
                vb.use_act_checkpoint = False
        return model

    # Patch the activation-ckpt wrapper to identity as a belt-and-suspenders
    # measure (some submodules may hold their own use_act_checkpoint flags).
    _orig_wrapper = getattr(mm, "activation_ckpt_wrapper", None)

    def _identity_wrapper(fn, use_reentrant=True):
        return fn

    if _orig_wrapper is not None:
        mm.activation_ckpt_wrapper = _identity_wrapper
        # Also patch the name imported into vl_combiner's namespace.
        try:
            import sam3.model.vl_combiner as vlc
            if getattr(vlc, "activation_ckpt_wrapper", None) is _orig_wrapper:
                vlc.activation_ckpt_wrapper = _identity_wrapper
        except Exception:
            pass

    return _disable_act_ckpt


# ============================================================
# RoPE freqs_cis recompute for non-native resolution (mirrors
# export_onnx.patch_freqs_cis, which is how the 504 ONNX was
# exported from the 1008-trained source model)
# ============================================================

def _patch_rope_for_resolution(model, new_grid_size):
    """Recompute RoPE freqs_cis for global attention blocks to ``new_grid_size``.

    Source model is trained at 1008 (grid 72); running at 504 (grid 36) without
    this patch trips ``reshape_for_broadcast``'s shape assertion.  Window
    attention blocks (input_size = window_size 24) are left unchanged — window
    partition pads the 36x36 grid to 48x48 and reuses the 24x24 freqs_cis.
    """
    import torch.nn as nn
    from sam3.model.vitdet import Attention

    trunk = model.backbone.vision_backbone.trunk
    count = 0
    for _name, module in trunk.named_modules():
        if not isinstance(module, Attention):
            continue
        if module.input_size is None or module.input_size[0] <= 30:
            continue  # window attention — unchanged
        module.input_size = (new_grid_size, new_grid_size)
        module.rope_pt_size = (new_grid_size, new_grid_size)
        if getattr(module, "rel_pos_h", None) is not None:
            module.rel_pos_h = nn.Parameter(
                torch.zeros(2 * new_grid_size - 1, module.head_dim))
        if getattr(module, "rel_pos_w", None) is not None:
            module.rel_pos_w = nn.Parameter(
                torch.zeros(2 * new_grid_size - 1, module.head_dim))
        module._setup_rope_freqs()
        count += 1
    logger.info("Patched global attn RoPE → grid %d (%d modules)",
                new_grid_size, count)


# ============================================================
# Hook capture
# ============================================================

def _to_numpy(x):
    """Recursively convert torch tensors / tuples / lists / dicts to numpy."""
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return type(x)(_to_numpy(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    return x


class _HookCollector:
    """Register forward (and pre) hooks on named submodules; store outputs."""

    def __init__(self):
        self.captures = {}          # stage -> list of captured outputs
        self._handles = []

    def _make_fwd(self, stage):
        def hook(module, inp, out):
            self.captures.setdefault(stage, []).append(_to_numpy(out))
        return hook

    def _make_pre(self, stage):
        def hook(module, inp):
            self.captures.setdefault(stage, []).append(_to_numpy(inp))
        return hook

    def attach(self, model):
        bb = model.backbone
        # 1 preprocess_in: input to the ViT trunk (preprocessed image)
        # 2 vit_out: ViT trunk output (pre-neck FPN) — analog of TPU ViT bmodel
        # NB: vision_backbone.forward is called directly (vl_combiner.py:88),
        # bypassing __call__, so hooks on vision_backbone never fire; we hook
        # its .trunk submodule instead (neck calls trunk via __call__, necks.py:110).
        trunk = bb.vision_backbone.trunk
        self._handles.append(
            trunk.register_forward_pre_hook(self._make_pre("preprocess_in")))
        self._handles.append(
            trunk.register_forward_hook(self._make_fwd("vit_out")))
        # 5 grounding encoder
        self._handles.append(
            model.transformer.encoder.register_forward_hook(self._make_fwd("gnd_enc")))
        # 6 grounding decoder
        self._handles.append(
            model.transformer.decoder.register_forward_hook(self._make_fwd("gnd_dec")))
        # 7 scoring
        if getattr(model, "dot_prod_scoring", None) is not None:
            self._handles.append(
                model.dot_prod_scoring.register_forward_hook(self._make_fwd("scoring")))
        # 8 mask head
        if getattr(model, "segmentation_head", None) is not None:
            self._handles.append(
                model.segmentation_head.register_forward_hook(self._make_fwd("mask_head")))
        return self

    def detach(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


# ============================================================
# Sam3SourceRef
# ============================================================

class Sam3SourceRef:
    """Run the official sam3 model at 504 and capture per-stage tensors."""

    def __init__(self, ckpt_path=None, bpe_path=None, resolution=504,
                 precision="f32"):
        from sam3.model_builder import build_sam3_image_model

        disable_ckpt = _patch_sam3_for_cpu_ref()

        if ckpt_path is None:
            ckpt_path = os.path.join(_SAMPLE_DIR, "models", "sam3.pt")
        if bpe_path is None:
            import sam3
            bpe_path = os.path.join(os.path.dirname(sam3.__file__),
                                    "assets", "bpe_simple_vocab_16e6.txt.gz")

        logger.info("Loading sam3 source model (resolution=%d, %s)...",
                    resolution, precision)
        t0 = time.time()
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=ckpt_path,
            device="cpu",
            load_from_HF=False,
            enable_segmentation=True,   # capture mask_head; reusable for phase 2
            enable_inst_interactivity=False,
        )
        model = model.float().eval()
        disable_ckpt(model)
        _patch_rope_for_resolution(model, resolution // 14)
        self.model = model
        self.resolution = resolution
        self.precision = precision

        # Sam3Processor drives preprocess + forward_grounding.
        from sam3.model.sam3_image_processor import Sam3Processor
        self.processor = Sam3Processor(
            model, resolution=resolution, device="cpu",
            confidence_threshold=0.0)   # keep all for raw comparison
        logger.info("  Model loaded in %.1fs", time.time() - t0)

        self.hooks = _HookCollector().attach(model)

    def _to_pil(self, image_bgr):
        from PIL import Image
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def run(self, image_bgr, prompt):
        """Run set_image + set_text_prompt; return dict of stage numpy tensors.

        Stages 3 (neck_fpn) and 4 (text_feats) come from the processor state;
        the rest come from hooks.  Raw forward_grounding outputs are captured
        by temporarily wrapping model.forward_grounding — the processor's
        _forward_grounding filters/scales to pixel xyxy, so its return is not
        the raw [B,Q,...] tensors we need for comparison.
        """
        self.hooks.captures.clear()
        pil = self._to_pil(image_bgr)

        t0 = time.time()
        state = self.processor.set_image(pil)
        backbone_out = state["backbone_out"]
        # 3 neck FPN — capture to numpy now; forward_grounding pops backbone_fpn.
        stage3 = {
            "backbone_fpn": _to_numpy(backbone_out["backbone_fpn"]),
            "vision_pos_enc": _to_numpy(backbone_out["vision_pos_enc"]),
        }

        # Capture raw forward_grounding return (pre processor filtering).
        raw = {}
        _orig_fg = self.model.forward_grounding

        def _capturing_fg(*args, **kwargs):
            out = _orig_fg(*args, **kwargs)
            raw.update(out)
            return out

        self.model.forward_grounding = _capturing_fg
        try:
            self.processor.set_text_prompt(prompt, state)
        finally:
            self.model.forward_grounding = _orig_fg

        # 4 text features (seq-first [T,B,C]) + mask (survive forward_grounding)
        stage4 = {
            "language_features": _to_numpy(backbone_out["language_features"]),
            "language_mask": _to_numpy(backbone_out["language_mask"]),
        }
        logger.info("  forward_grounding done in %.2fs", time.time() - t0)

        e2d = {
            "pred_boxes": _to_numpy(raw["pred_boxes"]),
            "pred_logits": _to_numpy(raw["pred_logits"]),
            "presence_logit_dec": _to_numpy(raw["presence_logit_dec"]),
        }
        if "pred_masks" in raw:
            e2d["pred_masks"] = _to_numpy(raw["pred_masks"])

        cap = self.hooks.captures
        return {
            "preprocess_in": cap.get("preprocess_in", [None])[0],
            "vit_out": cap.get("vit_out", [None])[0],
            "neck_fpn": stage3,
            "text_feats": stage4,
            "gnd_enc": cap.get("gnd_enc", [None])[0],
            "gnd_dec": cap.get("gnd_dec", [None])[0],
            "scoring": cap.get("scoring", [None])[0],
            "mask_head": cap.get("mask_head", [None])[0],
            "e2d": e2d,
            "fire_counts": {k: len(v) for k, v in cap.items()},
        }

    def run_tensor(self, x_np, prompt, orig_h=None, orig_w=None):
        """Run the source model on a PRE-PREPROCESSED image tensor (shared with
        the TPU pipeline) to isolate model-path divergence from preprocessing.

        x_np: (1,3,H,W) float32 normalized image (identical to TPU prepare_input).
        Bypasses Sam3Processor's resize/normalize; calls forward_image(x) +
        forward_text + forward_grounding directly.
        """
        import torch
        self.hooks.captures.clear()
        x = torch.from_numpy(np.ascontiguousarray(x_np)).float()

        t0 = time.time()
        backbone_out = self.model.backbone.forward_image(x)
        stage3 = {
            "backbone_fpn": _to_numpy(backbone_out["backbone_fpn"]),
            "vision_pos_enc": _to_numpy(backbone_out["vision_pos_enc"]),
        }

        text_outputs = self.model.backbone.forward_text([prompt], device="cpu")
        backbone_out.update(text_outputs)
        stage4 = {
            "language_features": _to_numpy(backbone_out["language_features"]),
            "language_mask": _to_numpy(backbone_out["language_mask"]),
        }

        raw = {}
        _orig_fg = self.model.forward_grounding

        def _capturing_fg(*args, **kwargs):
            out = _orig_fg(*args, **kwargs)
            raw.update(out)
            return out

        self.model.forward_grounding = _capturing_fg
        try:
            self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=self.processor.find_stage,
                geometric_prompt=self.model._get_dummy_prompt(),
                find_target=None)
        finally:
            self.model.forward_grounding = _orig_fg
        logger.info("  forward_grounding (shared tensor) done in %.2fs",
                    time.time() - t0)

        e2d = {
            "pred_boxes": _to_numpy(raw["pred_boxes"]),
            "pred_logits": _to_numpy(raw["pred_logits"]),
            "presence_logit_dec": _to_numpy(raw["presence_logit_dec"]),
        }
        if "pred_masks" in raw:
            e2d["pred_masks"] = _to_numpy(raw["pred_masks"])

        cap = self.hooks.captures
        return {
            "preprocess_in": _to_numpy(x),
            "vit_out": cap.get("vit_out", [None])[0],
            "neck_fpn": stage3,
            "text_feats": stage4,
            "gnd_enc": cap.get("gnd_enc", [None])[0],
            "gnd_dec": cap.get("gnd_dec", [None])[0],
            "scoring": cap.get("scoring", [None])[0],
            "mask_head": cap.get("mask_head", [None])[0],
            "e2d": e2d,
            "fire_counts": {k: len(v) for k, v in cap.items()},
        }

    def close(self):
        self.hooks.detach()


# ============================================================
# Diagnostics: print captured shapes (v1 validation)
# ============================================================

def _describe(name, obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict")
        for k, v in obj.items():
            _describe(k, v, indent + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj[:4]):
            _describe(f"[{i}]", v, indent + 1)
        if len(obj) > 4:
            print(f"{pad}  ... ({len(obj) - 4} more)")
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{pad}{name}: {type(obj).__name__} val={obj!r}")


def main():
    ap = argparse.ArgumentParser(description="sam3 source reference runner")
    ap.add_argument("--image", default=os.path.join(_SAMPLE_DIR, "datasets", "truck.jpg"))
    ap.add_argument("--prompt", default="a truck")
    ap.add_argument("--ckpt_path", default=None)
    ap.add_argument("--bpe_path", default=None)
    ap.add_argument("--resolution", type=int, default=504)
    ap.add_argument("--precision", default="f32", choices=["f32", "f16"])
    ap.add_argument("--dump", default=None, help="dump stage tensors to .npz")
    args = ap.parse_args()

    import cv2
    image = cv2.imread(args.image)
    if image is None:
        logger.error("Cannot read image: %s", args.image)
        sys.exit(1)
    logger.info("image %s shape=%s prompt=%r", args.image, image.shape, args.prompt)

    ref = Sam3SourceRef(ckpt_path=args.ckpt_path, bpe_path=args.bpe_path,
                        resolution=args.resolution, precision=args.precision)
    try:
        result = ref.run(image, args.prompt)
    finally:
        ref.close()

    print("\n=== captured stages ===")
    for stage in ("preprocess_in", "vit_out", "neck_fpn", "text_feats",
                  "gnd_enc", "gnd_dec", "scoring", "mask_head", "e2d"):
        _describe(stage, result.get(stage))

    print("\n=== hook fire counts (should be 1 each in eval) ===")
    print(result["fire_counts"])

    # End-to-end best detection
    e2d = result["e2d"]
    scores = 1.0 / (1.0 + np.exp(-e2d["pred_logits"].squeeze(-1)))  # [B,Q]
    best = int(scores[0].argmax())
    box = e2d["pred_boxes"][0, best]
    print(f"\n=== source E2D best: score={scores[0, best]:.4f} "
          f"box(cx,cy,w,h)={box.tolist()}")

    if args.dump:
        import pickle
        flat = {k: v for k, v in result.items() if k != "fire_counts"}
        with open(args.dump, "wb") as f:
            pickle.dump(flat, f)
        logger.info("dumped stages to %s", args.dump)


if __name__ == "__main__":
    main()
