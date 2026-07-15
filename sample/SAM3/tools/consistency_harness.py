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
SAM3 TPU-vs-source consistency harness.

Runs the official sam3 PyTorch model (via sam3_source_ref.Sam3SourceRef) and
the TPU pipeline (Sam3TpuRef below) on the SAME image+prompt, then diffs every
intermediate stage to localize where the TPU pipeline diverges from source.

Stage alignment (source key -> TPU key, both produced by the two refs):

  preprocess_in : normalized image (1,3,504,504)
  vit_out       : ViT output, pre-neck  (1,1024,36,36)
  neck_fpn      : {backbone_fpn[3], vision_pos_enc[3]}  post-scalp
  text_feats    : {language_features (T,B,C), language_mask (B,T)}
  gnd_enc       : {memory (L,B,D), memory_pos, [memory_text]}
  gnd_dec       : (hs (L,Q,B,C), ref_boxes (L,Q,B,4))  -- TPU DROPS presence
  scoring       : dot_prod_scoring out (L,B,Q,1)  -- layout-sensitive
  e2d           : {pred_boxes (B,Q,4), pred_logits (B,Q,1)}  -- TPU DROPS presence

Per-stage metrics: max_abs_err, rel_err = max_abs / max(|source|).
End-to-end: top-1 IoU, score diff, box (cxcywh) diff.

Usage:
  python consistency_harness.py --image ../datasets/truck.jpg --prompt "a truck"
  python consistency_harness.py --precision f32   # FP32 bmodels (isolate FP16 drift)
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("Sam3Harness")

_SAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PY_DIR = os.path.join(_SAMPLE_DIR, "python")
if _PY_DIR not in sys.path:
    sys.path.insert(0, _PY_DIR)
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)


# ============================================================
# shape / diff helpers
# ============================================================

def _shape(x):
    if isinstance(x, dict):
        return {k: _shape(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_shape(v) for v in x)
    if isinstance(x, np.ndarray):
        return x.shape
    return type(x).__name__


def _is_leaf_arr(x):
    return isinstance(x, np.ndarray)


def _unwrap_single(x):
    """Unwrap a single-element list/tuple to its contents (hooks capture inputs
    as 1-tuples; ViT trunk returns a 1-list).  Multi-element seqs are unchanged."""
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return _unwrap_single(x[0])
    return x


def _max_abs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None  # caller reports shape mismatch
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def _rel_err(a, b, max_abs):
    if max_abs is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    denom = float(max(np.max(np.abs(a)), 1e-8))
    return max_abs / denom


class DiffReport:
    """Accumulate per-leaf diffs; print a table at the end."""

    def __init__(self):
        self.rows = []  # (path, src_shape, tpu_shape, max_abs, rel_err, note)

    def add(self, path, src, tpu):
        src = _unwrap_single(src)
        tpu = _unwrap_single(tpu)
        src_shape = _shape(src)
        tpu_shape = _shape(tpu)
        if isinstance(src, dict) or isinstance(tpu, dict):
            if not (isinstance(src, dict) and isinstance(tpu, dict)):
                self.rows.append((path, src_shape, tpu_shape, None, None,
                                  "type mismatch (dict vs non-dict)"))
                return
            keys = list(src.keys())
            for k in tpu.keys():
                if k not in src:
                    keys.append(k)
            for k in keys:
                if k not in src:
                    self.rows.append((f"{path}.{k}", None, _shape(tpu[k]),
                                      None, None, "missing on SOURCE"))
                elif k not in tpu:
                    self.rows.append((f"{path}.{k}", _shape(src[k]), None,
                                      None, None, "missing on TPU"))
                else:
                    self.add(f"{path}.{k}", src[k], tpu[k])
            return
        if isinstance(src, (list, tuple)) or isinstance(tpu, (list, tuple)):
            if not (isinstance(src, (list, tuple)) and isinstance(tpu, (list, tuple))):
                self.rows.append((path, src_shape, tpu_shape, None, None,
                                  "type mismatch (seq vs non-seq)"))
                return
            n = max(len(src), len(tpu))
            for i in range(n):
                if i >= len(src):
                    self.rows.append((f"{path}[{i}]", None, _shape(tpu[i]),
                                      None, None, "missing on SOURCE"))
                elif i >= len(tpu):
                    self.rows.append((f"{path}[{i}]", _shape(src[i]), None,
                                      None, None, "missing on TPU"))
                else:
                    self.add(f"{path}[{i}]", src[i], tpu[i])
            return
        # leaf
        if not (_is_leaf_arr(src) and _is_leaf_arr(tpu)):
            self.rows.append((path, src_shape, tpu_shape, None, None,
                              f"non-array ({src!r} vs {tpu!r})"))
            return
        ma = _max_abs(src, tpu)
        if ma is None:
            self.rows.append((path, src_shape, tpu_shape, None, None,
                              "SHAPE MISMATCH"))
        else:
            self.rows.append((path, src_shape, tpu_shape, ma,
                              _rel_err(src, tpu, ma), ""))

    def print(self, abs_thresh=1e-3, rel_thresh=1e-2):
        print("\n=== per-stage diff (source vs TPU) ===")
        hdr = f"{'path':<34} {'src_shape':<20} {'tpu_shape':<20} {'max_abs':>12} {'rel_err':>10}  note"
        print(hdr)
        print("-" * len(hdr))
        for path, ss, ts, ma, re, note in self.rows:
            ma_s = f"{ma:.3e}" if ma is not None else "-"
            re_s = f"{re:.3e}" if re is not None else "-"
            ss_s = str(ss) if ss is not None else "-"
            ts_s = str(ts) if ts is not None else "-"
            flag = ""
            if ma is not None and (ma > abs_thresh or (re or 0) > rel_thresh):
                flag = "  << DIVERGES"
            print(f"{path:<34} {ss_s:<20} {ts_s:<20} {ma_s:>12} {re_s:>10}  {note}{flag}")

    def first_divergence(self, abs_thresh=1e-3, rel_thresh=1e-2):
        for path, ss, ts, ma, re, note in self.rows:
            if ma is None:
                continue
            if ma > abs_thresh or (re or 0) > rel_thresh:
                return path, ma, re
        return None


# ============================================================
# Sam3TpuRef — drive the TPU pipeline, capture each stage
# ============================================================

class Sam3TpuRef:
    """Run the TPU pipeline step-by-step, mirroring SAM3Pipeline.set_image +
    predict, but capture every bmodel-boundary tensor for diffing."""

    def __init__(self, model_dir, precision="f16", dev_id=0, resolution=504,
                 bpe_path=None, ckpt_path=None, mode="bmodel"):
        from sam3_infer import SAM3Pipeline
        logger.info("Loading %s pipeline (precision=%s, model_dir=%s)...",
                    mode.upper(), precision, model_dir)
        t0 = time.time()
        self.pipe = SAM3Pipeline(
            model_dir, precision, dev_id, resolution,
            bpe_path=bpe_path, ckpt_path=ckpt_path, mode=mode)
        self.precision = precision
        self.mode = mode
        logger.info("  %s pipeline loaded in %.1fs",
                    mode.upper(), time.time() - t0)

    def run(self, image_bgr, prompt, x=None):
        enc = self.pipe.image_encoder
        eng = self.pipe.engines
        g = self.pipe.grounding
        pp = g.post_processor

        # 1 preprocess_in (shared tensor x if provided, else prepare here)
        if x is None:
            preprocess_in = enc.prepare_input(image_bgr)        # (1,3,504,504)
        else:
            preprocess_in = np.ascontiguousarray(x)

        # 2 vit_out (ViT 5-part output, pre-neck)
        vit_out = enc.vit_encoder(preprocess_in)

        # 3 neck_fpn (4 levels -> scalp -> 3)
        neck_out = eng.neck_engine.process(
            eng.neck_graph_name,
            {eng.neck_input_name: np.ascontiguousarray(vit_out)})
        on = eng.neck_output_names
        fpn_feats = [neck_out[on[i]] for i in range(4)]
        fpn_pos = [enc.pos_encoder(f).astype(np.float32) for f in fpn_feats]
        if enc.scalp > 0:
            fpn_feats = fpn_feats[:-enc.scalp]
            fpn_pos = fpn_pos[:-enc.scalp]
        neck_fpn = {"backbone_fpn": fpn_feats, "vision_pos_enc": fpn_pos}

        # 4 text_feats
        text_features, text_mask, _inputs = self.pipe.text_encoder([prompt])
        text_feats = {"language_features": text_features,
                      "language_mask": text_mask}

        # 5 gnd_enc
        enc_inputs, src = g.prepare_encoder_inputs(
            fpn_feats, fpn_pos, text_features, text_mask)
        enc_gname = eng.gr_encoder.get_graph_names()[0]
        enc_output = eng.gr_encoder.process(enc_gname, enc_inputs)
        memory = enc_output[eng.gr_enc_output_names[0]]
        memory_pos = enc_output[eng.gr_enc_output_names[1]]
        # TPU encoder exports memory + its pos encoding; source names the
        # encoder-input pos "pos_embed" — align keys so the harness diffs them.
        gnd_enc = {"memory": memory, "pos_embed": memory_pos}
        for nm in eng.gr_enc_output_names[2:]:
            gnd_enc[nm] = enc_output[nm]   # e.g. memory_text if exported

        # 6 gnd_dec (TPU decoder exports hs + ref_boxes only — NO presence)
        # Decoder bmodel expects batch-first [B,T,C]; text_features is [T,B,C].
        prompt_arr = np.ascontiguousarray(
            text_features.transpose(1, 0, 2).astype(np.float32))  # [B,T,C]
        # Scoring (dot_prod_scoring/mean_pool_text) expects seq-first [T,B,C];
        # passing the transposed prompt here leaves scores as [L,T,Q,1] instead
        # of [L,B,Q,1], mis-ranking the top-1 box.
        prompt_tbc = np.ascontiguousarray(text_features.astype(np.float32))  # [T,B,C]
        prompt_mask_arr = np.ascontiguousarray(text_mask.astype(np.float32))
        dec_inputs = g.prepare_decoder_inputs(
            memory, memory_pos, prompt_arr, prompt_mask_arr, src)
        dec_gname = eng.gr_decoder.get_graph_names()[0]
        dec_output = eng.gr_decoder.process(dec_gname, dec_inputs)
        hs = dec_output[eng.gr_dec_output_names[0]]
        reference_boxes = dec_output[eng.gr_dec_output_names[1]]
        # Handle presence_feats concatenated in channel dim (260=4+256)
        if reference_boxes.shape[-1] > 4:
            reference_boxes = reference_boxes[..., :4]
        gnd_dec = (hs, reference_boxes)

        # 7 scoring (replicate process()'s scoring call for the diff)
        #    process() transposes hs [L,Q,B,C] -> [L,B,Q,C] before scoring.
        hs_lbc = np.ascontiguousarray(hs).transpose(0, 2, 1, 3)
        scoring = pp.dot_prod_scoring(hs_lbc, prompt_tbc, prompt_mask_arr)

        # 8 e2d (full postproc: scoring + bbox_embed; NO presence on TPU)
        results = g.process_output(hs, reference_boxes, prompt_tbc, prompt_mask_arr)
        e2d = {"pred_boxes": results["pred_boxes"],
               "pred_logits": results["pred_logits"]}

        # 9 mask (CPU NumpyMaskDecoder — same path as g.__call__). Mask feats
        # use the scalp'd fpn_feats (s05 dropped) the live pipeline passes; if
        # the decoder expects s05 as the highest-res level this surfaces as a
        # shape/quality bug in mask_iou. Guarded so a mask failure doesn't kill
        # the staged diff.
        obj_queries = np.ascontiguousarray(hs[-1]).transpose(1, 0, 2)  # [B,Q,C]
        prompt_mask_bool = text_mask.astype(bool)  # [B,T]
        try:
            mask_out = g.mask_decoder.predict_masks(
                fpn_feats, memory, obj_queries, prompt_tbc, prompt_mask_bool)
            e2d["pred_masks"] = mask_out["pred_masks"]
        except Exception as exc:
            logger.warning("mask stage failed: %s", exc, exc_info=True)

        return {
            "preprocess_in": preprocess_in,
            "vit_out": vit_out,
            "neck_fpn": neck_fpn,
            "text_feats": text_feats,
            "gnd_enc": gnd_enc,
            "gnd_dec": gnd_dec,
            "scoring": scoring,
            "e2d": e2d,
        }

    def close(self):
        try:
            self.pipe.close()
        except Exception:
            pass


# ============================================================
# End-to-end metrics
# ============================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _cxcywh_to_xyxy(b):
    xyxy = np.zeros_like(b)
    xyxy[..., 0] = b[..., 0] - b[..., 2] / 2
    xyxy[..., 1] = b[..., 1] - b[..., 3] / 2
    xyxy[..., 2] = b[..., 0] + b[..., 2] / 2
    xyxy[..., 3] = b[..., 1] + b[..., 3] / 2
    return xyxy


def _iou(a, b):
    """IoU of two xyxy boxes (scalars)."""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
        max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _resize_nearest(mask_2d, out_h, out_w):
    """Nearest-neighbor resize of a 2D array to (out_h, out_w)."""
    h, w = mask_2d.shape
    if h == out_h and w == out_w:
        return mask_2d
    sy, sx = h / out_h, w / out_w
    yi = np.clip((np.arange(out_h) / sy).astype(np.int32), 0, h - 1)
    xi = np.clip((np.arange(out_w) / sx).astype(np.int32), 0, w - 1)
    return mask_2d[yi[:, None], xi[None, :]]


def _mask_iou(src_logit, tpu_logit, src_q, tpu_q):
    """IoU of two logit masks (threshold >0), per query, resolution-aligned.

    src/tpu_logit: [Q, H, W] (batch 0 stripped by caller). Upsamples the smaller
    mask to the larger (nearest) before IoU. Returns None if an index is OOB.
    """
    src_logit = np.asarray(src_logit)
    tpu_logit = np.asarray(tpu_logit)
    if src_q >= src_logit.shape[0] or tpu_q >= tpu_logit.shape[0]:
        return None
    s = src_logit[src_q] > 0
    t = tpu_logit[tpu_q] > 0
    sh, sw = s.shape
    th, tw = t.shape
    if (sh, sw) != (th, tw):
        if sh * sw <= th * tw:
            s = _resize_nearest(s, th, tw)
        else:
            t = _resize_nearest(t, sh, sw)
    inter = int(np.logical_and(s, t).sum())
    union = int(np.logical_or(s, t).sum())
    return inter / union if union > 0 else 0.0


def e2e_compare(src_e2d, tpu_e2d):
    """Top-1 IoU + score/box diff.  Source score uses presence; TPU does not."""
    src_logits = np.asarray(src_e2d["pred_logits"]).squeeze(-1)  # [B,Q]
    src_scores = _sigmoid(src_logits)
    if "presence_logit_dec" in src_e2d:
        pres = _sigmoid(np.asarray(src_e2d["presence_logit_dec"]))
        src_scores = src_scores * pres   # source applies presence
    src_best = int(src_scores[0].argmax())
    src_box = np.asarray(src_e2d["pred_boxes"])[0, src_best]
    src_score = float(src_scores[0, src_best])

    tpu_logits = np.asarray(tpu_e2d["pred_logits"]).squeeze(-1)  # [B,Q] or [T,Q]
    tpu_scores = _sigmoid(tpu_logits)
    # TPU may carry an extra T axis from the text-layout bug; reduce to [?,Q]
    # by max over the leading axis so we still get a per-query score.
    tpu_box_raw = np.asarray(tpu_e2d["pred_boxes"])
    if tpu_scores.ndim == 2 and tpu_scores.shape[0] != src_scores.shape[0]:
        tpu_scores = tpu_scores.max(axis=0)          # [Q]
        tpu_scores = tpu_scores[np.newaxis, :]       # [1,Q]
    tpu_best = int(tpu_scores[0].argmax())
    tpu_box = tpu_box_raw[0, tpu_best] if tpu_box_raw.ndim == 3 \
        else tpu_box_raw[tpu_best]
    tpu_score = float(tpu_scores[0, tpu_best])

    src_xyxy = _cxcywh_to_xyxy(src_box)
    tpu_xyxy = _cxcywh_to_xyxy(tpu_box)
    iou = _iou(src_xyxy, tpu_xyxy)
    box_diff = float(np.max(np.abs(src_box - tpu_box)))
    score_diff = abs(src_score - tpu_score)

    # Mask IoU (logit masks, threshold >0). best-vs-best reflects end-to-end
    # quality (affected by top-1 ranking drift); same-query (src_best vs
    # tpu[src_best]) isolates mask-path correctness from ranking drift.
    mask_iou = None
    mask_iou_sameq = None
    src_m = src_e2d.get("pred_masks")
    tpu_m = tpu_e2d.get("pred_masks")
    if src_m is not None and tpu_m is not None:
        mask_iou = _mask_iou(src_m[0], tpu_m[0], src_best, tpu_best)
        mask_iou_sameq = _mask_iou(src_m[0], tpu_m[0], src_best, src_best)

    return {
        "src_best": src_best, "tpu_best": tpu_best,
        "src_score": src_score, "tpu_score": tpu_score,
        "score_diff": score_diff, "box_diff_cxcywh": box_diff,
        "iou": iou, "mask_iou": mask_iou, "mask_iou_sameq": mask_iou_sameq,
    }


def _stats(a, b):
    """Return (src_range, tpu_range, cosine_sim, src_mean, tpu_mean) for flat arrays."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    cos = float(np.dot(a, b) / (na * nb))
    return (float(a.min()), float(a.max()), float(a.mean()),
            float(b.min()), float(b.max()), float(b.mean()), cos)


def _print_stats(label, a, b):
    if a is None or b is None:
        print(f"  {label}: (one side None)")
        return
    try:
        smin, smax, smean, tmin, tmax, tmean, cos = _stats(a, b)
    except Exception as e:
        print(f"  {label}: stats error {e}")
        return
    print(f"  {label:<26} src[{smin:+.3f},{smax:+.3f}] mean={smean:+.3f}  "
          f"tpu[{tmin:+.3f},{tmax:+.3f}] mean={tmean:+.3f}  cos={cos:.4f}")


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="SAM3 TPU-vs-source consistency harness")
    ap.add_argument("--image", default=os.path.join(_SAMPLE_DIR, "datasets", "truck.jpg"))
    ap.add_argument("--prompt", default="a truck")
    ap.add_argument("--model_dir", default=os.path.join(_SAMPLE_DIR, "models", "BM1684X_504"))
    ap.add_argument("--precision", default="f16", choices=["f16", "f32", "bf16", "int8", "f16hp"])
    ap.add_argument("--mode", default="bmodel", choices=["bmodel", "onnx"],
                    help="backend for the non-source ref: bmodel=TPU sail "
                         "(needs device), onnx=onnxruntime CPU (no device). "
                         "onnx vs source isolates ONNX-quantitative baseline "
                         "from TPU codegen drift.")
    ap.add_argument("--resolution", type=int, default=504)
    ap.add_argument("--ckpt_path", default=None)
    ap.add_argument("--bpe_path", default=None)
    ap.add_argument("--abs_thresh", type=float, default=1e-3)
    ap.add_argument("--rel_thresh", type=float, default=1e-2)
    ap.add_argument("--shared_input", action="store_true",
                    help="preprocess once (TPU cv2) and feed the SAME tensor to "
                         "both source and TPU — isolates model-path divergence "
                         "from preprocessing")
    args = ap.parse_args()

    import cv2
    image = cv2.imread(args.image)
    if image is None:
        logger.error("Cannot read image: %s", args.image)
        sys.exit(1)
    logger.info("image %s shape=%s prompt=%r precision=%s",
                args.image, image.shape, args.prompt, args.precision)

    # Resolve precision-specific model_dir
    model_dir = args.model_dir
    if args.precision == "f32":
        f32_dir = os.path.join(_SAMPLE_DIR, "models", "BM1684X_504_f32")
        if os.path.isdir(f32_dir):
            model_dir = f32_dir

    # --- source ref ---
    from sam3_source_ref import Sam3SourceRef
    src = Sam3SourceRef(ckpt_path=args.ckpt_path, bpe_path=args.bpe_path,
                        resolution=args.resolution, precision="f32")

    # Shared preprocessed tensor (TPU cv2 path) for isolation mode.
    shared_x = None
    if args.shared_input:
        # Build a throwaway image_encoder just for prepare_input? Reuse TPU's.
        # Cheaper: replicate prepare_input inline (cv2).
        import cv2 as _cv2
        _img = _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB)
        _img = _cv2.resize(_img, (args.resolution, args.resolution),
                           interpolation=_cv2.INTER_LINEAR)
        shared_x = (_img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        logger.info("shared_input: feeding identical (1,3,%d,%d) tensor to both",
                    args.resolution, args.resolution)

    try:
        t0 = time.time()
        if args.shared_input:
            src_result = src.run_tensor(shared_x, args.prompt,
                                        orig_h=image.shape[0], orig_w=image.shape[1])
        else:
            src_result = src.run(image, args.prompt)
        logger.info("source ref run: %.2fs", time.time() - t0)
    finally:
        src.close()

    # --- TPU ref ---
    tpu = Sam3TpuRef(model_dir, precision=args.precision, dev_id=0,
                     resolution=args.resolution,
                     bpe_path=args.bpe_path, ckpt_path=args.ckpt_path,
                     mode=args.mode)
    try:
        t0 = time.time()
        tpu_result = tpu.run(image, args.prompt, x=shared_x)
        logger.info("%s ref run: %.2fs", args.mode.upper(), time.time() - t0)
    finally:
        tpu.close()

    # --- shapes (sanity) ---
    print("\n=== shapes (source) ===")
    for k in ("preprocess_in", "vit_out", "neck_fpn", "text_feats",
              "gnd_enc", "gnd_dec", "scoring", "e2d"):
        print(f"  src {k}: {_shape(src_result.get(k))}")
    print("\n=== shapes (TPU) ===")
    for k in ("preprocess_in", "vit_out", "neck_fpn", "text_feats",
              "gnd_enc", "gnd_dec", "scoring", "e2d"):
        print(f"  tpu {k}: {_shape(tpu_result.get(k))}")

    # --- stats: ranges + cosine similarity (is TPU dead / exploded / different?) ---
    print("\n=== stats (src vs tpu ranges + cosine) ===")
    _print_stats("vit_out", src_result.get("vit_out"), tpu_result.get("vit_out"))
    _sneck = src_result.get("neck_fpn", {})
    _tneck = tpu_result.get("neck_fpn", {})
    if isinstance(_sneck, dict) and isinstance(_tneck, dict):
        _sbf = _sneck.get("backbone_fpn")
        _tbf = _tneck.get("backbone_fpn")
        if isinstance(_sbf, list) and isinstance(_tbf, list) and len(_sbf) == len(_tbf):
            for i in range(len(_sbf)):
                _print_stats(f"neck_fpn.backbone_fpn[{i}]", _sbf[i], _tbf[i])
    _stxt = src_result.get("text_feats", {})
    _ttxt = tpu_result.get("text_feats", {})
    if isinstance(_stxt, dict) and isinstance(_ttxt, dict):
        _print_stats("text_feats.language_features",
                     _stxt.get("language_features"), _ttxt.get("language_features"))
    _senc = src_result.get("gnd_enc", {})
    _tenc = tpu_result.get("gnd_enc", {})
    if isinstance(_senc, dict) and isinstance(_tenc, dict):
        _print_stats("gnd_enc.memory", _senc.get("memory"), _tenc.get("memory"))
    _sdec = src_result.get("gnd_dec")
    _tdec = tpu_result.get("gnd_dec")
    if isinstance(_sdec, (list, tuple)) and isinstance(_tdec, (list, tuple)) \
            and len(_sdec) and len(_tdec):
        _print_stats("gnd_dec[0] (hs)", _sdec[0], _tdec[0])
        # per-layer gnd_dec hs 诊断 (2026-07-09)
        if isinstance(_sdec, (list, tuple)) and isinstance(_tdec, (list, tuple)) \
                and len(_sdec) and len(_tdec) \
                and hasattr(_sdec[0], 'shape') and len(_sdec[0].shape) >= 1:
            _nL = _sdec[0].shape[0]
            for _li in range(_nL):
                _print_stats(f"gnd_dec[0].hs[L{_li}]", _sdec[0][_li], _tdec[0][_li])
    _print_stats("e2d.pred_boxes",
                 src_result.get("e2d", {}).get("pred_boxes"),
                 tpu_result.get("e2d", {}).get("pred_boxes"))

    # --- per-stage diff ---
    rep = DiffReport()
    for k in ("preprocess_in", "vit_out", "neck_fpn", "text_feats",
              "gnd_enc", "gnd_dec", "scoring", "e2d"):
        rep.add(k, src_result.get(k), tpu_result.get(k))
    rep.print(args.abs_thresh, args.rel_thresh)

    div = rep.first_divergence(args.abs_thresh, args.rel_thresh)
    if div:
        print(f"\n>>> first divergence at: {div[0]}  (max_abs={div[1]:.3e}, "
              f"rel_err={div[2]:.3e})")
    else:
        print("\n>>> all stages within tolerance")

    # --- end-to-end ---
    print("\n=== end-to-end (top-1 detection) ===")
    m = e2e_compare(src_result["e2d"], tpu_result["e2d"])
    print(f"  source : best={m['src_best']:>3} score={m['src_score']:.4f}")
    print(f"  tpu    : best={m['tpu_best']:>3} score={m['tpu_score']:.4f}")
    print(f"  IoU={m['iou']:.4f}  score_diff={m['score_diff']:.4f}  "
          f"box_diff(cxcywh)={m['box_diff_cxcywh']:.4f}")
    miou = m.get("mask_iou")
    miou_sq = m.get("mask_iou_sameq")
    if miou is None:
        print("  mask_IoU=n/a")
    else:
        print(f"  mask_IoU(best-vs-best)={miou:.4f}  mask_IoU(same-query)={miou_sq:.4f}"
              f"  [src_best={m['src_best']} tpu_best={m['tpu_best']}]")


if __name__ == "__main__":
    main()
