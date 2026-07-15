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
SAM3 Full Inference Pipeline
=============================
End-to-end SAM3 inference with hybrid TPU/CPU execution.

  Image → ViT (TPU, 5 parts) → Neck (TPU) → FPN features + Position encodings
  Text  → TextEncoder (TPU) → Text features (850x speedup vs CPU)
  FPN + Text → Grounding (TPU: encoder + decoder), CPU post-process → Boxes + Scores

Architecture follows SAM2 pattern:
  - Engines loaded once in __init__, reused across calls
  - Separate classes per component with clear responsibilities
  - Timing tracked per stage as instance attributes

Usage:
  python sam3_infer.py --image cat.jpg --prompt "a cat"
  python sam3_infer.py --model_dir models/BM1684X_504 --precision f16
"""

import sys
import os
import time
import argparse
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("SAM3")

# ============================================================
# CPU model patches (for SAM3 compatibility on CPU)
# ============================================================

PATCHES_APPLIED = False


def _patch_sam3_model():
    """Patch SAM3 ops for ONNX/CPU compatibility. Idempotent."""
    global PATCHES_APPLIED
    if PATCHES_APPLIED:
        return
    PATCHES_APPLIED = True

    import torch

    # We need the developer repo root on sys.path for sam3 imports
    _repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from sam3.model.vitdet import Mlp

    def patched_mlp_forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop1(x)
        x = self.norm(x); x = self.fc2(x); x = self.drop2(x)
        return x

    Mlp.forward = patched_mlp_forward

    # Patch PositionEmbeddingSine to avoid CUDA precomputation on CPU
    from sam3.model import position_encoding
    _orig_init = position_encoding.PositionEmbeddingSine.__init__

    def _patched_posenc_init(self, *args, **kwargs):
        kwargs["precompute_resolution"] = None
        return _orig_init(self, *args, **kwargs)

    position_encoding.PositionEmbeddingSine.__init__ = _patched_posenc_init

    # Patch pin_memory for CPU safety
    _pin_orig = torch.Tensor.pin_memory

    def cpu_safe_pin(self, device=None):
        if not self.is_cuda:
            return self
        return _pin_orig(self, device)

    torch.Tensor.pin_memory = cpu_safe_pin


def _load_sam3_model(ckpt_path, bpe_path, enable_segmentation=False):
    """Load SAM3 model for CPU post-processing.

    enable_segmentation=False by default (post-process scoring/bbox weights
    don't need the seg head). The numpy mask decoder extraction passes True
    so model.segmentation_head is populated.
    """
    from sam3.model_builder import build_sam3_image_model

    logger.info("Loading SAM3 model (CPU)...")
    t0 = time.time()
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=ckpt_path,
        device='cpu',
        load_from_HF=False,
        enable_segmentation=enable_segmentation,
        enable_inst_interactivity=False,
    )
    model = model.float().eval()
    logger.info("  Model loaded in %.1fs", time.time() - t0)
    return model


# ============================================================
# Position Encoding (numpy, for Neck / Grounding)
# ============================================================

class PositionEmbeddingSine:
    """Sinusoidal position encoding matching SAM3's PositionEmbeddingSine."""

    def __init__(self, num_pos_feats=256, temperature=10000,
                 normalize=True, scale=None):
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * np.pi
        self._cache = {}

    def __call__(self, x):
        H, W = x.shape[2], x.shape[3]
        cache_key = (H, W)
        if cache_key in self._cache:
            pos = self._cache[cache_key]
            return np.repeat(pos[np.newaxis], x.shape[0], axis=0)

        y_embed = np.arange(1, H + 1, dtype=np.float32).reshape(1, -1, 1)
        y_embed = np.tile(y_embed, (1, 1, W))
        x_embed = np.arange(1, W + 1, dtype=np.float32).reshape(1, 1, -1)
        x_embed = np.tile(x_embed, (1, H, 1))

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = np.arange(self.num_pos_feats, dtype=np.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, np.newaxis] / dim_t[np.newaxis, np.newaxis, np.newaxis, :]
        pos_y = y_embed[:, :, :, np.newaxis] / dim_t[np.newaxis, np.newaxis, np.newaxis, :]

        pos_x = np.stack([np.sin(pos_x[:, :, :, 0::2]),
                          np.cos(pos_x[:, :, :, 1::2])], axis=4)
        pos_x = pos_x.reshape(pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], -1)
        pos_y = np.stack([np.sin(pos_y[:, :, :, 0::2]),
                          np.cos(pos_y[:, :, :, 1::2])], axis=4)
        pos_y = pos_y.reshape(pos_y.shape[0], pos_y.shape[1], pos_y.shape[2], -1)

        pos = np.concatenate([pos_y, pos_x], axis=3).transpose(0, 3, 1, 2)
        self._cache[cache_key] = pos[0]
        return np.repeat(pos, x.shape[0], axis=0)


# ============================================================
# OnnxEngine — onnxruntime shim with sail.Engine-compatible API
# (process(graph, {name: arr}) -> {name: arr}; name-keyed like SYSIO)
# ============================================================

_ONNX_DTYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(bool)": np.bool_,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
    "tensor(float16)": np.float16,
}


class OnnxEngine:
    """Wraps onnxruntime InferenceSession to mimic sail.Engine's SYSIO API.

    sail call sites do engine.process(graph_name, {in_name: arr}) and index
    outputs by name (via get_output_names). This shim returns a name-keyed
    dict too, so ViT/neck/text/gr_enc/gr_dec call sites work unchanged.

    input_aliases/output_aliases: when given, expose logical names (the keys
    the calling code builds) instead of raw ONNX tensor names; the internal
    run() still feeds raw ONNX input names. Needed for gr_enc/gr_dec where
    prepare_*_inputs hardcodes logical names ("src", "memory", ...). Also
    coerces each input to the ONNX-expected dtype (sail accepts float32 for
    bool/int inputs; onnxruntime is strict).
    """

    def __init__(self, onnx_path, sess_options=None,
                 input_aliases=None, output_aliases=None):
        import onnxruntime as ort
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
            sess_options.intra_op_num_threads = max(
                1, (os.cpu_count() or 4) // 2)
        self.sess = ort.InferenceSession(
            onnx_path, sess_options, providers=["CPUExecutionProvider"])
        self._inputs = self.sess.get_inputs()
        self._outputs = self.sess.get_outputs()
        self._in_onnx_names = [i.name for i in self._inputs]
        self._out_onnx_names = [o.name for o in self._outputs]
        self._in_dtypes = [
            _ONNX_DTYPE_MAP.get(i.type, np.float32) for i in self._inputs]
        self.in_names = (list(input_aliases) if input_aliases
                         else list(self._in_onnx_names))
        self.out_names = (list(output_aliases) if output_aliases
                          else list(self._out_onnx_names))
        self._graph = "onnx"

    def get_graph_names(self):
        return [self._graph]

    def get_input_names(self, gname):
        return list(self.in_names)

    def get_output_names(self, gname):
        return list(self.out_names)

    def process(self, gname, feed):
        arrs = {}
        for inp, alias, dt in zip(self._inputs, self.in_names, self._in_dtypes):
            arr = feed[alias]
            if arr.dtype != dt:
                arr = arr.astype(dt)
            arrs[inp.name] = np.ascontiguousarray(arr)
        outs = self.sess.run(self._out_onnx_names, arrs)
        return dict(zip(self.out_names, outs))


# ============================================================
# SAM3 Engines — loads all bmodels once (SAM2 ModelLoader pattern)
# ============================================================

class SAM3Engines:
    """
    Load and hold all TPU bmodel engines (SAM2 pattern: load once, reuse).

    Engines:
      - vit_engines[0..4]: ViT backbone (5 parts)
      - neck_engine:       Neck FPN
      - text_encoder:      Text encoder (VETextEncoder)
      - grounding_encoder: Grounding TransformerEncoderFusion
      - grounding_decoder: Grounding TransformerDecoder
    """

    def __init__(self, model_dir, precision="f16", dev_id=0, mode="bmodel"):
        self.dev_id = dev_id
        self.model_dir = os.path.abspath(model_dir)
        self.precision = precision
        self.mode = mode

        if mode == "onnx":
            self._init_onnx()
            return

        import sophon.sail as sail

        # --- ViT engines (5 parts) ---
        self.vit_engines = []
        self.vit_graph_names = []
        self.vit_input_names = []
        self.vit_output_names = []
        for part in range(5):
            path = os.path.join(model_dir, "vit",
                                f"sam3_vit_part{part}_{precision}_1b.bmodel")
            if not os.path.exists(path):
                raise FileNotFoundError(f"ViT part {part} bmodel not found: {path}")
            engine = sail.Engine(path, dev_id, sail.IOMode.SYSIO)
            gname = engine.get_graph_names()[0]
            self.vit_engines.append(engine)
            self.vit_graph_names.append(gname)
            self.vit_input_names.append(engine.get_input_names(gname)[0])
            self.vit_output_names.append(engine.get_output_names(gname)[0])
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info("ViT part %d: %.0fMB loaded", part, size_mb)

        # --- Neck engine ---
        neck_path = os.path.join(model_dir, "neck",
                                 f"sam3_neck_{precision}_1b.bmodel")
        if not os.path.exists(neck_path):
            raise FileNotFoundError(f"Neck bmodel not found: {neck_path}")
        self.neck_engine = sail.Engine(neck_path, dev_id, sail.IOMode.SYSIO)
        self.neck_graph_name = self.neck_engine.get_graph_names()[0]
        self.neck_input_name = self.neck_engine.get_input_names(
            self.neck_graph_name)[0]
        self.neck_output_names = self.neck_engine.get_output_names(
            self.neck_graph_name)
        logger.info("Neck: %.0fMB loaded",
                    os.path.getsize(neck_path) / (1024 * 1024))

        # --- Text encoder engine ---
        te_path, _ = self._gnd_bmodel_path("sam3_text_encoder")
        self.text_encoder = None
        if os.path.exists(te_path):
            self.text_encoder = sail.Engine(te_path, dev_id, sail.IOMode.SYSIO)
            te_gname = self.text_encoder.get_graph_names()[0]
            self.te_input_name = self.text_encoder.get_input_names(te_gname)[0]
            self.te_output_name = self.text_encoder.get_output_names(te_gname)[0]
            logger.info("Text encoder: %.0fMB loaded",
                        os.path.getsize(te_path) / (1024 * 1024))
        else:
            logger.info("Text encoder bmodel not found, will use CPU fallback")

        # --- Grounding encoder engine ---
        ge_path, _ = self._gnd_bmodel_path("sam3_grounding_encoder")
        if not os.path.exists(ge_path):
            raise FileNotFoundError(f"Grounding encoder bmodel not found: {ge_path}")
        self.gr_encoder = sail.Engine(ge_path, dev_id, sail.IOMode.SYSIO)
        ge_gname = self.gr_encoder.get_graph_names()[0]
        self.gr_enc_input_names = self.gr_encoder.get_input_names(ge_gname)
        self.gr_enc_output_names = self.gr_encoder.get_output_names(ge_gname)
        logger.info("Grounding encoder: %.0fMB loaded",
                    os.path.getsize(ge_path) / (1024 * 1024))

        # --- Grounding decoder engine ---
        gd_path, _ = self._gnd_bmodel_path("sam3_grounding_decoder")
        if not os.path.exists(gd_path):
            raise FileNotFoundError(f"Grounding decoder bmodel not found: {gd_path}")
        self.gr_decoder = sail.Engine(gd_path, dev_id, sail.IOMode.SYSIO)
        gd_gname = self.gr_decoder.get_graph_names()[0]
        self.gr_dec_input_names = self.gr_decoder.get_input_names(gd_gname)
        self.gr_dec_output_names = self.gr_decoder.get_output_names(gd_gname)
        logger.info("Grounding decoder: %.0fMB loaded",
                    os.path.getsize(gd_path) / (1024 * 1024))

    def _init_onnx(self):
        """ONNX backend: load onnxruntime sessions as sail.Engine-compatible
        shims. Uses onnx_504_static/ (same source ONNX as deployed bmodels)
        + onnx_grounding_504/ so onnx vs bmodel differ only by TPU codegen.
        text_encoder stays None -> CPU torch fallback (same as source ref)."""
        import onnxruntime as ort
        onnx_root = os.path.dirname(self.model_dir)  # .../models
        vit_dir = os.path.join(onnx_root, "onnx_504_static")
        gnd_dir = os.path.join(onnx_root, "onnx_grounding_504")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)

        def _load(path, input_aliases=None):
            eng = OnnxEngine(path, so, input_aliases=input_aliases)
            g = eng.get_graph_names()[0]
            return eng, g, eng.get_input_names(g), eng.get_output_names(g)

        # ViT 5 parts
        self.vit_engines, self.vit_graph_names = [], []
        self.vit_input_names, self.vit_output_names = [], []
        for part in range(5):
            path = os.path.join(vit_dir, f"sam3_vit_part{part}.onnx")
            if not os.path.exists(path):
                raise FileNotFoundError(f"ViT part {part} onnx not found: {path}")
            eng, g, ins, outs = _load(path)
            self.vit_engines.append(eng)
            self.vit_graph_names.append(g)
            self.vit_input_names.append(ins[0])
            self.vit_output_names.append(outs[0])
            logger.info("ViT part %d onnx loaded: %s", part, os.path.basename(path))

        # Neck
        neck_path = os.path.join(vit_dir, "sam3_neck_combined.onnx")
        if not os.path.exists(neck_path):
            raise FileNotFoundError(f"Neck onnx not found: {neck_path}")
        self.neck_engine, self.neck_graph_name, neck_ins, neck_outs = _load(neck_path)
        self.neck_input_name = neck_ins[0]
        self.neck_output_names = neck_outs
        logger.info("Neck onnx loaded")

        # Text encoder: None -> CPU torch fallback (keeps text path identical
        # to bmodel mode when no text bmodel is deployed; same source as ref)
        self.text_encoder = None
        logger.info("Text encoder: onnx mode uses CPU torch fallback")

        # Grounding encoder (input aliases match prepare_encoder_inputs keys)
        ge_path = os.path.join(gnd_dir, "sam3_grounding_encoder.onnx")
        if not os.path.exists(ge_path):
            raise FileNotFoundError(f"Grounding encoder onnx not found: {ge_path}")
        self.gr_encoder, _, self.gr_enc_input_names, self.gr_enc_output_names = _load(
            ge_path, input_aliases=["src", "src_pos", "prompt", "prompt_mask"])
        logger.info("Grounding encoder onnx loaded")

        # Grounding decoder (input aliases match prepare_decoder_inputs keys)
        gd_path = os.path.join(gnd_dir, "sam3_grounding_decoder.onnx")
        if not os.path.exists(gd_path):
            raise FileNotFoundError(f"Grounding decoder onnx not found: {gd_path}")
        self.gr_decoder, _, self.gr_dec_input_names, self.gr_dec_output_names = _load(
            gd_path, input_aliases=["memory", "memory_pos", "memory_mask",
                                    "spatial_shapes", "valid_ratios",
                                    "prompt", "prompt_mask"])
        logger.info("Grounding decoder onnx loaded")

    def _gnd_bmodel_path(self, stem):
        """Resolve a grounding/text bmodel path with f32->f16 fallback.

        ViT+neck run at the requested precision (f32 eliminates the 32-block
        quantization drift). Grounding/text are shallow transformers whose f16
        drift is negligible; if the f32 bmodel is not compiled (text f32 would
        be ~1.3 GB), gracefully fall back to f16 rather than crashing.
        Returns (path, precision_used).
        """
        for p in (self.precision, "f16"):
            path = os.path.join(self.model_dir, "grounding",
                                f"{stem}_{p}_1b.bmodel")
            if os.path.exists(path):
                if p != self.precision:
                    logger.info("%s: f32 bmodel absent, using f16", stem)
                return path, p
        return os.path.join(self.model_dir, "grounding",
                            f"{stem}_{self.precision}_1b.bmodel"), self.precision

    def close(self):
        """Free all engines."""
        for e in self.vit_engines:
            del e
        self.vit_engines.clear()
        if self.neck_engine:
            del self.neck_engine
        if self.text_encoder:
            del self.text_encoder
        if self.gr_encoder:
            del self.gr_encoder
        if self.gr_decoder:
            del self.gr_decoder


# ============================================================
# SAM3ImageEncoder — ViT + Neck (SAM2 ImageFeatureExtractor pattern)
# ============================================================

class SAM3ImageEncoder:
    """
    Image feature extraction: preprocess → ViT (5 TPU parts) → Neck (TPU) → FPN.

    Follows SAM2ImageFeatureExtractor pattern:
      - prepare_input() for preprocessing
      - __call__() for inference
      - Timing tracked as instance attributes

    Delegates ViT inference to SAM3ViTEncoder (from sam3_vit_infer.py)
    to avoid code duplication. Neck inference uses pre-loaded engine.
    """

    def __init__(self, engines: SAM3Engines, precision="f16",
                 dev_id=0, resolution=504):
        from sam3_vit_infer import SAM3ViTEncoder

        self.engines = engines
        self.resolution = resolution
        self.grid = resolution // 14  # patch_size=14
        self.scalp = 1  # discard lowest FPN level
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=256)

        # Use SAM3ViTEncoder with pre-loaded engines (no code duplication)
        self.vit_encoder = SAM3ViTEncoder(
            precision=precision, dev_id=dev_id, resolution=resolution,
            engines=engines.vit_engines,
            graph_names=engines.vit_graph_names,
            input_names=engines.vit_input_names,
            output_names=engines.vit_output_names)

        # Timing (SAM2 pattern)
        self.preprocess_time = 0
        self.vit_time = 0
        self.neck_time = 0

    def prepare_input(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess BGR image for ViT input.
        SAM3 normalization: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
        """
        import cv2
        t0 = time.time()

        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resolution, self.resolution),
                        interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]

        self.preprocess_time = time.time() - t0
        return img

    def __call__(self, image_bgr: np.ndarray):
        """
        Run ViT + Neck and return FPN features.

        Args:
            image_bgr: BGR image (H, W, 3) uint8

        Returns:
            fpn_feats: list of feature maps (after scalp)
            fpn_pos: list of position encodings (after scalp)
        """
        x = self.prepare_input(image_bgr)

        # --- ViT backbone (delegated to SAM3ViTEncoder) ---
        t0 = time.time()
        vit_features = self.vit_encoder(x)
        self.vit_time = time.time() - t0

        # --- Neck FPN ---
        t0 = time.time()
        neck_out = self.engines.neck_engine.process(
            self.engines.neck_graph_name,
            {self.engines.neck_input_name: vit_features})

        # Extract FPN levels in order: s4, s2, s1, s05
        on = self.engines.neck_output_names
        fpn_feats = [neck_out[on[i]] for i in range(4)]
        fpn_pos = [self.pos_encoder(f).astype(np.float32) for f in fpn_feats]
        self.neck_time = time.time() - t0

        # Apply scalp
        if self.scalp > 0:
            fpn_feats = fpn_feats[:-self.scalp]
            fpn_pos = fpn_pos[:-self.scalp]

        return fpn_feats, fpn_pos


# ============================================================
# SAM3TextEncoder — text tokenization + TPU inference
# ============================================================

class SAM3TextEncoder:
    """
    Text encoder: tokenize on CPU → TextTransformer on TPU.

    Uses lightweight SimpleTokenizer (no model weights needed).
    Falls back to CPU if bmodel not available.
    """

    def __init__(self, engines: SAM3Engines,
                 bpe_path=None, context_length=32):
        self.engines = engines
        self.context_length = context_length
        self.bpe_path = bpe_path
        self.tokenizer = None

        # Timing
        self.encode_time = 0

    def _ensure_tokenizer(self):
        if self.tokenizer is not None:
            return
        from sam3.model.tokenizer_ve import SimpleTokenizer
        bpe_path = self.bpe_path
        if bpe_path is None:
            # Auto-detect from sam3 package
            import sam3
            bpe_path = os.path.join(
                os.path.dirname(sam3.__file__), "assets",
                "bpe_simple_vocab_16e6.txt.gz")
        logger.info("Loading text tokenizer...")
        self.tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        logger.info("  Tokenizer loaded (context_length=%d)", self.context_length)

    def __call__(self, texts):
        """
        Encode text prompts.

        Args:
            texts: str or list of str

        Returns:
            text_features: np.ndarray [32, B, 256]
            text_mask: np.ndarray [B, 32] bool (True = ignore)
            inputs_embeds: np.ndarray [32, B, 1024]
        """
        if isinstance(texts, str):
            texts = [texts]

        t0 = time.time()

        if self.engines.text_encoder is not None:
            # TPU path: tokenize on CPU, run transformer on TPU
            self._ensure_tokenizer()
            import torch
            tokenized = self.tokenizer(texts, context_length=self.context_length)
            batch_size = len(texts)

            text_mask_np = ~(tokenized.cpu().numpy() != 0)

            te_engine = self.engines.text_encoder
            te_gname = te_engine.get_graph_names()[0]
            text_features_list = []
            for i in range(batch_size):
                tokens_i = tokenized[i:i + 1].cpu().numpy().astype(np.int32)
                tokens_i = np.ascontiguousarray(tokens_i)
                te_out = te_engine.process(
                    te_gname, {self.engines.te_input_name: tokens_i})
                text_features_list.append(
                    te_out[self.engines.te_output_name])

            text_features_np = np.concatenate(text_features_list, axis=1)
            inputs_embeds_np = np.zeros(
                (self.context_length, batch_size, 1024), dtype=np.float32)
        else:
            # CPU fallback
            _patch_sam3_model()
            # Need sam3 package's bpe_path for full model
            bpe_path = self.bpe_path
            if bpe_path is None:
                import sam3
                bpe_path = os.path.join(
                    os.path.dirname(sam3.__file__), "assets",
                    "bpe_simple_vocab_16e6.txt.gz")
            # Find checkpoint path
            ckpt_path = os.path.join(
                os.path.dirname(self.engines.model_dir), "sam3.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(self.engines.model_dir, "..", "sam3.pt")
            model = _load_sam3_model(ckpt_path, bpe_path)
            te = model.backbone.language_backbone
            import torch
            with torch.no_grad():
                mask, features, embeds = te(texts, device=torch.device('cpu'))
            text_features_np = features.cpu().numpy()
            text_mask_np = mask.cpu().numpy()
            inputs_embeds_np = embeds.cpu().numpy()

        self.encode_time = time.time() - t0
        return text_features_np, text_mask_np, inputs_embeds_np


# ============================================================
# Numpy Post-Processor — lightweight, no CPU model needed
# ============================================================

class NumpyPostProcessor:
    """
    Efficient numpy-based post-processing for grounding outputs.

    Replaces the 22s/1.7GB CPU model loading with lightweight operations:
      - dot_prod_scoring: mean-pool text → project → dot-product → scale
      - bbox_embed: 3-layer MLP (256→256→256→4)
      - inverse_sigmoid: 1 / (1 + exp(-x))

    Weights are extracted once from the PyTorch checkpoint and cached.
    """

    def __init__(self, ckpt_path=None, bpe_path=None):
        self.ckpt_path = ckpt_path
        self.bpe_path = bpe_path
        self._weights = None
        self._loaded = False

    def _get_weights_cache_path(self):
        """Get path for cached post-processing weights .npz file.

        Checks multiple locations: alongside the ckpt, in the current dir,
        and in the models dir.
        """
        candidates = []
        # Alongside the checkpoint
        if self.ckpt_path and os.path.exists(os.path.dirname(self.ckpt_path)):
            candidates.append(os.path.join(
                os.path.dirname(self.ckpt_path), "post_process_weights.npz"))
        # In the default models directory
        candidates.append(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "models", "post_process_weights.npz"))
        # In the current directory
        candidates.append(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "post_process_weights.npz"))
        return candidates  # return all, try each

    def _try_load_cached_weights(self):
        """Try to load previously extracted weights from cache."""
        for cache_path in self._get_weights_cache_path():
            if not os.path.exists(cache_path):
                continue
            try:
                cached = np.load(cache_path, allow_pickle=False)
                self._weights = {
                    "prompt_proj_weight": cached["prompt_proj_weight"],
                    "prompt_proj_bias": cached["prompt_proj_bias"],
                    "hs_proj_weight": cached["hs_proj_weight"],
                    "hs_proj_bias": cached["hs_proj_bias"],
                    "d_proj": int(cached["d_proj"]),
                    "bbox_layer0_weight": cached["bbox_layer0_weight"],
                    "bbox_layer0_bias": cached["bbox_layer0_bias"],
                    "bbox_layer1_weight": cached["bbox_layer1_weight"],
                    "bbox_layer1_bias": cached["bbox_layer1_bias"],
                    "bbox_layer2_weight": cached["bbox_layer2_weight"],
                    "bbox_layer2_bias": cached["bbox_layer2_bias"],
                }
                self._loaded = True
                logger.info("Loaded cached post-processing weights: %s",
                            cache_path)
                return True
            except Exception as e:
                logger.warning("Failed to load cached weights from %s: %s",
                             cache_path, e)
        return False

    def _extract_weights(self):
        """Extract post-processing weights from SAM3 checkpoint (one-time).

        Extracted weights are cached to disk as a small .npz file (~1MB),
        eliminating the need to load the full 1.7GB CPU model on subsequent runs.
        """
        import torch

        _patch_sam3_model()
        ckpt = self.ckpt_path
        if ckpt is None or not os.path.exists(ckpt):
            ckpt = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "sam3.pt")
        self.ckpt_path = ckpt  # remember for cache path
        bpe = self.bpe_path
        if bpe is None:
            import sam3
            bpe = os.path.join(os.path.dirname(sam3.__file__),
                               "assets", "bpe_simple_vocab_16e6.txt.gz")

        logger.info("Extracting post-processing weights from SAM3 checkpoint...")
        t0 = time.time()
        model = _load_sam3_model(ckpt, bpe)

        # Extract dot_prod_scoring weights
        dps = model.dot_prod_scoring
        w = {
            "prompt_proj_weight": dps.prompt_proj.weight.detach().cpu().numpy(),
            "prompt_proj_bias": dps.prompt_proj.bias.detach().cpu().numpy(),
            "hs_proj_weight": dps.hs_proj.weight.detach().cpu().numpy(),
            "hs_proj_bias": dps.hs_proj.bias.detach().cpu().numpy(),
            "d_proj": dps.d_proj,
        }

        # Extract bbox_embed weights (MLP: 256→256→256→4)
        bbox = model.transformer.decoder.bbox_embed
        for i, layer in enumerate(bbox.layers):
            w[f"bbox_layer{i}_weight"] = layer.weight.detach().cpu().numpy()
            w[f"bbox_layer{i}_bias"] = layer.bias.detach().cpu().numpy()

        del model

        # Cache to disk for fast loading next time
        cache_paths = self._get_weights_cache_path()
        cache_path = cache_paths[1] if len(cache_paths) > 1 else cache_paths[0]
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            np.savez_compressed(cache_path, **w)
            logger.info("  Cached to: %s", cache_path)
        except Exception as e:
            logger.warning("  Failed to cache weights: %s", e)

        self._weights = w
        self._loaded = True
        np_weights = {k: v for k, v in w.items() if isinstance(v, np.ndarray)}
        logger.info("  Weights extracted in %.1fs (%.1f KB)",
                    time.time() - t0,
                    sum(v.nbytes for v in np_weights.values()) / 1024)

    def _ensure_weights(self):
        if self._loaded:
            return
        # Try cached weights first (fast, ~1ms)
        if self._try_load_cached_weights():
            return
        # Fall back to extracting from checkpoint (one-time, ~10-30s)
        self._extract_weights()

    def inverse_sigmoid(self, x, eps=1e-3):
        """Inverse sigmoid: ln(x / (1-x)) with clipping for numerical stability.

        Matches sam3.model.model_misc.inverse_sigmoid exactly.
        """
        x = np.clip(x, 0, 1)
        x1 = np.clip(x, eps, None)
        x2 = np.clip(1 - x, eps, None)
        return np.log(x1 / x2)

    def mean_pool_text(self, prompt, prompt_mask):
        """Mean-pool text features over valid (non-masked) tokens.

        Args:
            prompt: [T, B, C] text features
            prompt_mask: [B, T] bool (True = ignore/masked)
        Returns:
            pooled: [B, C]
        """
        # is_valid: [B, T, 1] — 1=valid, 0=masked
        is_valid = (~prompt_mask.astype(bool)).astype(np.float32).transpose(1, 0)[..., None]
        num_valid = np.clip(is_valid.sum(axis=0), 1, None)
        pooled = (prompt * is_valid).sum(axis=0) / num_valid
        return pooled  # [B, C]

    def dot_prod_scoring(self, hs, prompt, prompt_mask):
        """Compute text-query dot-product scores.

        Args:
            hs: [L, B, Q, C] decoder hidden states
            prompt: [T, B, C] text features
            prompt_mask: [B, T] bool (True = ignore)
        Returns:
            scores: [L, B, Q, 1]
        """
        w = self._weights
        # Mean-pool text over valid tokens
        pooled = self.mean_pool_text(prompt, prompt_mask)  # [B, C]
        # Project pooled prompt and hidden states
        proj_pooled = pooled @ w["prompt_proj_weight"].T + w["prompt_proj_bias"]  # [B, D]
        proj_hs = hs @ w["hs_proj_weight"].T + w["hs_proj_bias"]  # [L, B, Q, D]
        # Dot product: per-batch dot product between proj_hs and proj_pooled
        scores = (proj_hs * proj_pooled[None, :, None, :]).sum(
            axis=-1, keepdims=True)  # [L, B, Q, 1]
        scale = 1.0 / np.sqrt(w["d_proj"])
        scores = scores * scale
        return scores

    def bbox_embed(self, x):
        """3-layer MLP: 256→256→256→4 with ReLU.

        Args:
            x: [..., 256]
        Returns:
            out: [..., 4]
        """
        w = self._weights
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        for i in range(3):
            weight = w[f"bbox_layer{i}_weight"]
            bias = w[f"bbox_layer{i}_bias"]
            x = x @ weight.T + bias
            if i < 2:
                x = np.maximum(x, 0)  # ReLU
        return x.reshape(*shape[:-1], 4)

    def process(self, hs, reference_boxes, prompt, prompt_mask):
        """Full post-processing: scoring + box refinement.

        Args:
            hs: [L, Q, B, C] decoder hidden states (TPU bmodel output)
            reference_boxes: [L, Q, B, 4] reference boxes (TPU bmodel output)
            prompt: [T, B, C] text features (seq-first)
            prompt_mask: [B, T] bool (True = ignore)

        Returns:
            dict with pred_logits, pred_scores, pred_boxes
        """
        self._ensure_weights()

        # TPU bmodel outputs [L, Q, B, C] — convert to [L, B, Q, C] for processing
        hs = np.ascontiguousarray(hs).transpose(0, 2, 1, 3)        # [L, B, Q, C]
        reference_boxes = np.ascontiguousarray(reference_boxes).transpose(0, 2, 1, 3)  # [L, B, Q, 4]

        # Dot-product scoring
        outputs_scores = self.dot_prod_scoring(hs, prompt, prompt_mask)

        # BBox offsets → boxes
        anchor_box_offsets = self.bbox_embed(hs)  # [L, B, Q, 4]

        # Box refinement: ref_inv + offset → sigmoid
        ref_inv = self.inverse_sigmoid(reference_boxes[-1:])  # [1, B, Q, 4]
        outputs_coord = 1.0 / (1.0 + np.exp(-(ref_inv + anchor_box_offsets[-1:])))

        # Take last layer
        pred_logits = outputs_scores[-1]       # [B, Q, 1]
        pred_boxes = outputs_coord[-1]          # [B, Q, 4]
        pred_scores = 1.0 / (1.0 + np.exp(-pred_logits))  # sigmoid

        return {
            "pred_logits": pred_logits,
            "pred_scores": pred_scores,
            "pred_boxes": pred_boxes,
        }


# ============================================================
# NumpyMaskDecoder — CPU mask prediction from TPU features
# ============================================================

class NumpyMaskDecoder:
    """Numpy-based mask prediction using decoder queries and FPN features.

    Replicates the UniversalSegmentationHead forward path:
      cross_attend(encoder_hs, prompt) → pixel_decoder(fpn+encoder) →
      instance_seg_head → mask_predictor(queries, pixel_embed) → masks

    Weights are extracted once from the SAM3 checkpoint and cached.
    """

    def __init__(self):
        self._weights = None
        self._loaded = False

    def _get_weights_path(self):
        candidates = []
        candidates.append(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "models", "seg_head_weights.npz"))
        candidates.append(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "seg_head_weights.npz"))
        return candidates

    def _ensure_weights(self):
        if self._loaded:
            return
        for path in self._get_weights_path():
            if os.path.exists(path):
                self._weights = dict(np.load(path, allow_pickle=False))
                self._loaded = True
                logger.info("Loaded mask decoder weights: %s", path)
                return
        # Extract from checkpoint
        self._extract_weights()

    def _extract_weights(self):
        import torch
        _patch_sam3_model()
        ckpt = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "sam3.pt")
        model = _load_sam3_model(ckpt, None, enable_segmentation=True)
        seg_head = model.segmentation_head
        w = {}
        pd = seg_head.pixel_decoder
        for i in range(pd.num_upsampling_stages):
            w[f'pixel_decoder.conv_{i}.weight'] = pd.conv_layers[i].weight.detach().cpu().numpy()
            w[f'pixel_decoder.conv_{i}.bias'] = pd.conv_layers[i].bias.detach().cpu().numpy()
            w[f'pixel_decoder.norm_{i}.weight'] = pd.norms[i].weight.detach().cpu().numpy()
            w[f'pixel_decoder.norm_{i}.bias'] = pd.norms[i].bias.detach().cpu().numpy()
        me = seg_head.mask_predictor.mask_embed
        for i, layer in enumerate(me.layers):
            if hasattr(layer, 'weight'):
                w[f'mask_embed.layer_{i}.weight'] = layer.weight.detach().cpu().numpy()
                w[f'mask_embed.layer_{i}.bias'] = layer.bias.detach().cpu().numpy()
        ish = seg_head.instance_seg_head
        w['instance_seg_head.weight'] = ish.weight.detach().cpu().numpy()
        w['instance_seg_head.bias'] = ish.bias.detach().cpu().numpy()
        ssh = seg_head.semantic_seg_head
        w['semantic_seg_head.weight'] = ssh.weight.detach().cpu().numpy()
        w['semantic_seg_head.bias'] = ssh.bias.detach().cpu().numpy()
        cap = seg_head.cross_attend_prompt
        if cap is not None:
            w['cross_attn.in_proj_weight'] = cap.in_proj_weight.detach().cpu().numpy()
            w['cross_attn.in_proj_bias'] = cap.in_proj_bias.detach().cpu().numpy()
            w['cross_attn.out_proj.weight'] = cap.out_proj.weight.detach().cpu().numpy()
            w['cross_attn.out_proj.bias'] = cap.out_proj.bias.detach().cpu().numpy()
        can = seg_head.cross_attn_norm
        w['cross_attn_norm.weight'] = can.weight.detach().cpu().numpy()
        w['cross_attn_norm.bias'] = can.bias.detach().cpu().numpy()
        self._weights = w
        self._loaded = True
        # Cache for next time
        cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "seg_head_weights.npz")
        np.savez_compressed(cache_path, **w)
        logger.info("Extracted and cached mask decoder weights: %s", cache_path)

    def _layer_norm(self, x, weight, bias, eps=1e-5):
        """LayerNorm: (x - mean) / std * weight + bias."""
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * weight + bias

    def _group_norm(self, x, weight, bias, num_groups=8, eps=1e-5):
        """GroupNorm for [N, C, H, W] or [C, H, W]."""
        if x.ndim == 3:
            x = x[np.newaxis, ...]
            squeeze_batch = True
        else:
            squeeze_batch = False
        N, C, H, W = x.shape
        x_reshaped = x.reshape(N, num_groups, C // num_groups, H, W)
        mean = x_reshaped.mean(axis=(2, 3, 4), keepdims=True)
        var = ((x_reshaped - mean) ** 2).mean(axis=(2, 3, 4), keepdims=True)
        x_norm = (x_reshaped - mean) / np.sqrt(var + eps)
        x_norm = x_norm.reshape(N, C, H, W)
        result = x_norm * weight[None, :, None, None] + bias[None, :, None, None]
        if squeeze_batch:
            result = result[0]
        return result

    def _conv2d(self, x, weight, bias):
        """Simple Conv2d using im2col for [N,C,H,W] or [C,H,W] input.
        Supports 1x1 and 3x3 kernels with padding=1 for 3x3.
        """
        squeeze_batch = False
        if x.ndim == 3:
            x = x[np.newaxis, ...]
            squeeze_batch = True
        N, C_in, H, W = x.shape
        K = weight.shape[0]
        k_h, k_w = weight.shape[2], weight.shape[3]
        pad = (k_h - 1) // 2

        if pad > 0:
            x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        else:
            x_pad = x

        out_h = H
        out_w = W
        result = np.zeros((N, K, out_h, out_w), dtype=x.dtype)

        for kh in range(k_h):
            for kw in range(k_w):
                # Extract patch: [N, C_in, out_h, out_w]
                patch = x_pad[:, :, kh:kh + out_h, kw:kw + out_w]
                # Contract over C_in
                result += np.tensordot(patch, weight[:, :, kh, kw], axes=([1], [1])).transpose(0, 3, 1, 2)
                # tensordot gives [N, out_h, out_w, K] → transpose to [N, K, out_h, out_w]

        result = result + bias[None, :, None, None]
        if squeeze_batch:
            result = result[0]
        return result

    def _mlp(self, x, weights_prefix, num_layers=3):
        """3-layer MLP: Linear → ReLU → Linear → ReLU → Linear."""
        for i in range(num_layers):
            w = self._weights[f'{weights_prefix}.layer_{i}.weight']
            b = self._weights[f'{weights_prefix}.layer_{i}.bias']
            x = x @ w.T + b
            if i < num_layers - 1:
                x = np.maximum(x, 0)  # ReLU
        return x

    def _cross_attend(self, encoder_hs, prompt, prompt_mask):
        """Cross-attention: encoder_hs attends to prompt (text features).

        Args:
            encoder_hs: [T_enc, B, C] encoder hidden states
            prompt: [T_txt, B, C] text features (seq-first)
            prompt_mask: [B, T_txt] bool (True = ignore/pad)

        Returns:
            [T_enc, B, C] cross-attended features
        """
        w = self._weights
        # Layer norm encoder_hs
        ln_w = w['cross_attn_norm.weight']
        ln_b = w['cross_attn_norm.bias']
        tgt = self._layer_norm(encoder_hs.transpose(1, 0, 2), ln_w, ln_b).transpose(1, 0, 2)

        # MultiheadAttention: embed_dim=256, num_heads=8
        embed_dim = 256
        num_heads = 8
        head_dim = embed_dim // num_heads

        # in_proj_weight: [3*embed_dim, embed_dim] = [768, 256]
        in_proj_w = w['cross_attn.in_proj_weight']
        in_proj_b = w['cross_attn.in_proj_bias']
        out_proj_w = w['cross_attn.out_proj.weight']
        out_proj_b = w['cross_attn.out_proj.bias']

        # Split in_proj into Q, K, V
        q_w, k_w, v_w = np.split(in_proj_w, 3, axis=0)
        q_b, k_b, v_b = np.split(in_proj_b, 3, axis=0)

        # Q from encoder_hs, K/V from prompt
        # encoder_hs: [T_enc, B, C], prompt: [T_txt, B, C]
        T_enc, B, C = encoder_hs.shape
        T_txt = prompt.shape[0]

        Q = encoder_hs @ q_w.T + q_b  # [T_enc, B, C=256]
        K = prompt @ k_w.T + k_b       # [T_txt, B, C=256]
        V = prompt @ v_w.T + v_b       # [T_txt, B, C=256]

        # Reshape to [B*num_heads, T, head_dim]
        def reshape_mha(t, T_len):
            return t.reshape(T_len, B * num_heads, head_dim).transpose(1, 0, 2)

        Q = reshape_mha(Q, T_enc)
        K = reshape_mha(K, T_txt)
        V = reshape_mha(V, T_txt)

        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(head_dim)
        attn_weights = Q @ K.transpose(0, 2, 1) * scale  # [B*num_heads, T_enc, T_txt]

        # Apply mask: prompt_mask [B, T_txt], True = ignore
        if prompt_mask is not None:
            # Expand mask: [B, T_txt] → [B, 1, T_txt] → [B*num_heads, T_txt]
            mask_val = prompt_mask.astype(np.float32)  # [B, T_txt], True where pad
            mask_val = np.repeat(mask_val, num_heads, axis=0)  # [B*num_heads, T_txt]
            attn_weights = attn_weights - 1e9 * mask_val[:, np.newaxis, :]

        attn_weights = self._softmax(attn_weights, axis=-1)
        attn_out = attn_weights @ V  # [B*num_heads, T_enc, head_dim]

        # Reshape back: [T_enc, B, C]
        attn_out = attn_out.transpose(1, 0, 2).reshape(T_enc, B, embed_dim)

        # Output projection
        attn_out = attn_out @ out_proj_w.T + out_proj_b

        # Residual
        return attn_out + encoder_hs

    def _softmax(self, x, axis=-1):
        x_max = x.max(axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def _pixel_decoder(self, backbone_feats):
        """PixelDecoder: upsample and merge multi-scale features.

        Args:
            backbone_feats: list of [B, C, H, W] from low to high resolution

        Returns:
            [B, C, H_high, W_high] pixel embeddings
        """
        w = self._weights
        prev_fpn = backbone_feats[-1]  # highest resolution
        fpn_feats = backbone_feats[:-1]

        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn = bb_feat
            # Upsample prev_fpn to match curr_fpn spatial size
            prev_up = self._interpolate(prev_fpn, size=curr_fpn.shape[-2:])
            prev_fpn = curr_fpn + prev_up

            conv_w = w[f'pixel_decoder.conv_{layer_idx}.weight']
            conv_b = w[f'pixel_decoder.conv_{layer_idx}.bias']
            norm_w = w[f'pixel_decoder.norm_{layer_idx}.weight']
            norm_b = w[f'pixel_decoder.norm_{layer_idx}.bias']

            prev_fpn = self._conv2d(prev_fpn, conv_w, conv_b)
            prev_fpn = self._group_norm(prev_fpn, norm_w, norm_b)
            prev_fpn = np.maximum(prev_fpn, 0)  # ReLU

        return prev_fpn

    def _interpolate(self, x, size, mode='nearest'):
        """Simple nearest-neighbor upsampling for [N, C, H, W]."""
        squeeze_batch = False
        if x.ndim == 3:
            x = x[np.newaxis, ...]
            squeeze_batch = True
        N, C, H, W = x.shape
        target_h, target_w = size
        scale_h = target_h / H
        scale_w = target_w / W
        h_idx = (np.arange(target_h) / scale_h).astype(np.int32)
        w_idx = (np.arange(target_w) / scale_w).astype(np.int32)
        h_idx = np.clip(h_idx, 0, H - 1)
        w_idx = np.clip(w_idx, 0, W - 1)
        result = x[:, :, h_idx[:, None], w_idx[None, :]]
        if squeeze_batch:
            result = result[0]
        return result

    def predict_masks(self, fpn_feats, encoder_hs, obj_queries, prompt, prompt_mask):
        """Generate masks from decoder queries and FPN features.

        Args:
            fpn_feats: list of [B, C, H_i, W_i] FPN features (s2, s1, s05)
            encoder_hs: [T, B, C] encoder hidden states (from grounding encoder TPU)
            obj_queries: [B, Q, C] decoder output queries (last layer)
            prompt: [T_txt, B, C] text features (seq-first)
            prompt_mask: [B, T_txt] bool (True = ignore)

        Returns:
            dict with pred_masks [B, Q, H, W], semantic_seg [B, 1, H, W]
        """
        self._ensure_weights()
        w = self._weights

        B, Q, C = obj_queries.shape
        T_enc = encoder_hs.shape[0]

        # 1. Cross-attend encoder with text prompt
        encoder_hs_attended = self._cross_attend(encoder_hs, prompt, prompt_mask)

        # 2. Create encoder visual embed from encoder_hs (spatial part)
        # encoder_hs: [T_enc, B, C] where T_enc includes spatial + text tokens
        last_feat = fpn_feats[-1]  # s1: [B, C, H, W]
        spatial_dim = last_feat.shape[-2] * last_feat.shape[-1]
        encoder_visual = encoder_hs_attended[:spatial_dim].transpose(1, 2, 0)
        encoder_visual = encoder_visual.reshape(B, C, last_feat.shape[-2], last_feat.shape[-1])

        # 3. Build backbone_feats with encoder visual replacing last level
        backbone_feats = [f.copy() for f in fpn_feats]
        backbone_feats[-1] = encoder_visual

        # 4. Pixel decoder
        pixel_embed = self._pixel_decoder(backbone_feats)
        # pixel_embed: [B, C, H_out, W_out]

        # 5. Instance seg head (1x1 conv)
        inst_w = w['instance_seg_head.weight'][:, :, 0, 0]  # [256, 256]
        inst_b = w['instance_seg_head.bias']
        instance_embeds = self._conv2d(pixel_embed,
                                       w['instance_seg_head.weight'],
                                       w['instance_seg_head.bias'])

        # 6. Mask predictor: MLP on queries + dot product with pixel features
        # obj_queries: [B, Q, C]
        mask_features = self._mlp(obj_queries, 'mask_embed')  # [B, Q, C]

        # Einsum: bqc, bchw → bqhw
        pred_masks = np.einsum('bqc,bchw->bqhw', mask_features, instance_embeds)

        # 7. Semantic seg
        # sem_w must be 1-D (256,) so [None,:,None,None] → (1,256,1,1) aligns
        # the in-channel axis with pixel_embed (1,256,H,W). The 2-D slice
        # [:,:,0,0] → (1,256) leaves the 256 axis unconsumed by the single ':'
        # and appends it at the end → (1,1,1,1,256), a broadcast crash.
        sem_w = w['semantic_seg_head.weight'][0, :, 0, 0]  # (256,)
        sem_b = w['semantic_seg_head.bias']
        # Manual 1x1 conv for semantic seg
        semantic_seg = (pixel_embed * sem_w[None, :, None, None]).sum(axis=1, keepdims=True) + sem_b[None, :, None, None]

        return {
            "pred_masks": pred_masks,
            "semantic_seg": semantic_seg,
        }


# ============================================================
# SAM3Grounding — TPU encoder + decoder + CPU post-processing
# ============================================================

class SAM3Grounding:
    """
    Grounding model: Encoder (TPU) → Decoder (TPU) → Scoring/Box refine (CPU).

    Follows SAM2ImageMaskPredictor pattern:
      - prepare_inputs() for building input dicts
      - __call__() for inference
      - process_output() for post-processing

    Uses NumpyPostProcessor for lightweight scoring and box refinement,
    eliminating the 22s/1.7GB CPU model loading overhead.
    """

    def __init__(self, engines: SAM3Engines,
                 cpu_model=None, ckpt_path=None, bpe_path=None):
        self.engines = engines
        self.ckpt_path = ckpt_path
        self.bpe_path = bpe_path

        # Lightweight numpy post-processor (replaces CPU model)
        self.post_processor = NumpyPostProcessor(
            ckpt_path=ckpt_path, bpe_path=bpe_path)

        # Pre-load weights at init time (one-time cost, not per-inference)
        self.post_processor._ensure_weights()

        # Numpy mask decoder
        self.mask_decoder = NumpyMaskDecoder()
        self.mask_decoder._ensure_weights()

        # Keep cpu_model reference for backward compatibility
        self.cpu_model = cpu_model
        self._owns_model = False

        # Timing (SAM2 pattern)
        self.encoder_time = 0
        self.decoder_time = 0
        self.postprocess_time = 0

    def _ensure_cpu_model(self):
        """Backward-compat: load CPU model for legacy code paths."""
        if self.cpu_model is not None:
            return
        _patch_sam3_model()
        ckpt = self.ckpt_path
        if ckpt is None:
            ckpt = os.path.join(
                os.path.dirname(self.engines.model_dir), "sam3.pt")
        bpe = self.bpe_path
        if bpe is None:
            import sam3
            bpe = os.path.join(os.path.dirname(sam3.__file__),
                               "assets", "bpe_simple_vocab_16e6.txt.gz")
        self.cpu_model = _load_sam3_model(ckpt, bpe)
        self._owns_model = True

    def prepare_encoder_inputs(self, fpn_feats, fpn_pos,
                               text_features, text_mask):
        """Build encoder input dict matching bmodel input names."""
        # Single-scale: use last FPN level (s1 = 36x36 for 504)
        src = np.ascontiguousarray(fpn_feats[-1].astype(np.float32))
        src_pos = np.ascontiguousarray(fpn_pos[-1].astype(np.float32))
        # Text features: [T, B, C] → batch-first [B, T, C]
        prompt = np.ascontiguousarray(
            text_features.transpose(1, 0, 2).astype(np.float32))
        prompt_mask = np.ascontiguousarray(text_mask.astype(np.float32))

        enc_inputs = {}
        for name in self.engines.gr_enc_input_names:
            if name == "src":
                enc_inputs[name] = src
            elif name == "src_pos":
                enc_inputs[name] = src_pos
            elif name == "prompt":
                enc_inputs[name] = prompt
            elif name == "prompt_mask":
                enc_inputs[name] = prompt_mask

        return enc_inputs, src

    def prepare_decoder_inputs(self, memory, memory_pos,
                               prompt, prompt_mask, src):
        """Build decoder input dict matching bmodel input names."""
        _, _, h, w = src.shape
        spatial_shapes = np.array([[h, w]], dtype=np.int32)
        valid_ratios = np.array([[[1.0, 1.0]]], dtype=np.float32)
        memory_mask = np.zeros((memory.shape[0], memory.shape[1]),
                               dtype=np.float32)

        memory = np.ascontiguousarray(memory)
        memory_pos = np.ascontiguousarray(memory_pos)
        memory_mask = np.ascontiguousarray(memory_mask)
        spatial_shapes = np.ascontiguousarray(spatial_shapes)
        valid_ratios = np.ascontiguousarray(valid_ratios)

        dec_inputs = {}
        for name in self.engines.gr_dec_input_names:
            if name == "memory":
                dec_inputs[name] = memory
            elif name == "memory_pos":
                dec_inputs[name] = memory_pos
            elif name == "memory_mask":
                dec_inputs[name] = memory_mask
            elif name == "spatial_shapes":
                dec_inputs[name] = spatial_shapes
            elif name == "valid_ratios":
                dec_inputs[name] = valid_ratios
            elif name == "prompt":
                dec_inputs[name] = prompt
            elif name == "prompt_mask":
                dec_inputs[name] = prompt_mask

        return dec_inputs

    def process_output(self, hs, reference_boxes, prompt, prompt_mask):
        """CPU post-processing: scoring + box refinement (numpy, no model needed)."""
        t0 = time.time()

        # Use numpy post-processor (lightweight, no CPU model loading)
        results = self.post_processor.process(
            hs, reference_boxes, prompt, prompt_mask)

        self.postprocess_time = time.time() - t0
        return results

    def __call__(self, fpn_feats, fpn_pos,
                 text_features, text_mask, inputs_embeds,
                 text_prompts, boxes=None, box_labels=None):
        """
        Run grounding model.

        Returns:
            dict with pred_logits, pred_scores, pred_boxes,
            or None if falling back to CPU.
        """
        # Box prompts not yet supported in TPU path
        if boxes is not None:
            logger.info("Box prompts not yet supported in TPU path")
            return None

        # --- Encoder ---
        t0 = time.time()
        enc_inputs, src = self.prepare_encoder_inputs(
            fpn_feats, fpn_pos, text_features, text_mask)
        enc_gname = self.engines.gr_encoder.get_graph_names()[0]
        enc_output = self.engines.gr_encoder.process(enc_gname, enc_inputs)
        memory = enc_output[self.engines.gr_enc_output_names[0]]
        memory_pos = enc_output[self.engines.gr_enc_output_names[1]]
        self.encoder_time = time.time() - t0

        # Prepare prompt for decoder (decoder bmodel expects batch-first [B, T, C])
        prompt = np.ascontiguousarray(
            text_features.transpose(1, 0, 2).astype(np.float32))
        prompt_mask_arr = np.ascontiguousarray(text_mask.astype(np.float32))
        # Scoring (NumpyPostProcessor.dot_prod_scoring/mean_pool_text) expects
        # seq-first [T, B, C]; text_features is already [T, B, C] from the text
        # encoder, so pass it directly. Using the transposed `prompt` here makes
        # mean_pool_text fail to aggregate over tokens -> scores become
        # [L, T, Q, 1] instead of [L, B, Q, 1], which mis-ranks the top-1 box.
        prompt_tbc = np.ascontiguousarray(text_features.astype(np.float32))

        # --- Decoder ---
        t0 = time.time()
        dec_inputs = self.prepare_decoder_inputs(
            memory, memory_pos, prompt, prompt_mask_arr, src)
        dec_gname = self.engines.gr_decoder.get_graph_names()[0]
        dec_output = self.engines.gr_decoder.process(dec_gname, dec_inputs)
        hs = dec_output[self.engines.gr_dec_output_names[0]]
        reference_boxes = dec_output[self.engines.gr_dec_output_names[1]]
        self.decoder_time = time.time() - t0

        # --- CPU Post-processing ---
        results = self.process_output(hs, reference_boxes, prompt_tbc, prompt_mask_arr)

        # --- Mask prediction ---
        t_mask = time.time()
        # Prepare text features in seq-first format for mask decoder
        prompt_seq_first = np.ascontiguousarray(
            text_features.astype(np.float32))  # [T, B, C]
        prompt_mask_bool = text_mask.astype(bool)  # [B, T]

        # Prepare obj_queries: hs from decoder [L, Q, B, C] → last layer [B, Q, C]
        obj_queries = np.ascontiguousarray(hs[-1]).transpose(1, 0, 2)  # [B, Q, C]

        mask_results = self.mask_decoder.predict_masks(
            fpn_feats, memory, obj_queries, prompt_seq_first, prompt_mask_bool)
        results.update(mask_results)
        self.mask_time = time.time() - t_mask

        logger.info("Grounding TPU: enc=%.0fms, dec=%.0fms, pp=%.0fms, mask=%.0fms",
                    self.encoder_time * 1000, self.decoder_time * 1000,
                    self.postprocess_time * 1000, self.mask_time * 1000)

        return results


# ============================================================
# SAM3Pipeline — full end-to-end orchestrator (SAM2Image pattern)
# ============================================================

class SAM3Pipeline:
    """
    Full SAM3 inference pipeline.

    Follows SAM2Image pattern:
      - Engines loaded once in __init__
      - set_image() / predict() pattern
      - Timing tracked cumulatively
    """

    version = "1.0.0"

    def __init__(self, model_dir, precision="f16", dev_id=0,
                 resolution=504, bpe_path=None, ckpt_path=None, mode="bmodel"):
        self.version = "1.0.0"
        self.resolution = resolution
        self.grid = resolution // 14
        self.model_dir = os.path.abspath(model_dir)
        self.mode = mode

        # Resolve default paths
        if ckpt_path is None:
            ckpt_path = os.path.join(
                os.path.dirname(model_dir), "sam3.pt")
        if bpe_path is None:
            import sam3
            bpe_path = os.path.join(
                os.path.dirname(sam3.__file__), "assets",
                "bpe_simple_vocab_16e6.txt.gz")

        logger.info("SAM3Pipeline init, model_dir=%s, precision=%s, resolution=%d",
                    model_dir, precision, resolution)

        # Load all TPU engines once (SAM2 pattern)
        self.engines = SAM3Engines(model_dir, precision, dev_id, mode=mode)

        # Component wrappers
        self.image_encoder = SAM3ImageEncoder(
            self.engines, precision, dev_id, resolution)
        self.text_encoder = SAM3TextEncoder(
            self.engines, bpe_path=bpe_path, context_length=32)
        self.grounding = SAM3Grounding(
            self.engines, ckpt_path=ckpt_path, bpe_path=bpe_path)

        # Timing (cumulative, SAM2 pattern)
        self.preprocess_time = 0
        self.vit_time = 0
        self.neck_time = 0
        self.text_encode_time = 0
        self.grounding_time = 0
        self.postprocess_time = 0

    def set_image(self, image_bgr):
        """Extract image features (ViT + Neck)."""
        if isinstance(image_bgr, str):
            import cv2
            image_bgr = cv2.imread(image_bgr)
            if image_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image_bgr}")
        self.fpn_feats, self.fpn_pos = self.image_encoder(image_bgr)
        self.preprocess_time = self.image_encoder.preprocess_time
        self.vit_time = self.image_encoder.vit_time
        self.neck_time = self.image_encoder.neck_time

    def predict(self, text_prompts, boxes=None, box_labels=None):
        """
        Run grounding on text prompts.

        Returns:
            dict with pred_logits, pred_scores, pred_boxes, pred_boxes_xyxy, timing
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        # --- Text encoding ---
        text_features, text_mask, inputs_embeds = self.text_encoder(text_prompts)
        self.text_encode_time = self.text_encoder.encode_time

        # --- Grounding (one prompt at a time via TPU) ---
        all_results = []
        for i, prompt in enumerate(text_prompts):
            tf_i = text_features[:, i:i + 1, :]      # [T, 1, C]
            tm_i = text_mask[i:i + 1, :]              # [1, T]
            ie_i = inputs_embeds[:, i:i + 1, :] if inputs_embeds is not None else None

            result = self.grounding(
                self.fpn_feats, self.fpn_pos,
                tf_i, tm_i, ie_i,
                [prompt], boxes, box_labels)

            if result is None:
                # Fall back to CPU for unsupported cases (e.g., box prompts)
                logger.info("Using CPU grounding fallback for: %s", prompt)
                t0 = time.time()
                result = self._run_grounding_cpu(
                    self.fpn_feats, self.fpn_pos,
                    tf_i, tm_i, ie_i,
                    [prompt], boxes, box_labels)
                self.grounding_time += time.time() - t0
            all_results.append(result)

        # Merge per-prompt results
        results = {
            "pred_logits": np.concatenate(
                [r["pred_logits"] for r in all_results], axis=0),
            "pred_scores": np.concatenate(
                [r["pred_scores"] for r in all_results], axis=0),
            "pred_boxes": np.concatenate(
                [r["pred_boxes"] for r in all_results], axis=0),
        }
        # Carry masks through (computed in grounding.__call__ via
        # NumpyMaskDecoder). The CPU fallback (_run_grounding_cpu) returns only
        # the best query and no masks, so guard on every prompt having them.
        if all_results and all("pred_masks" in r for r in all_results):
            results["pred_masks"] = np.concatenate(
                [r["pred_masks"] for r in all_results], axis=0)
        self.grounding_time = (
            self.grounding.encoder_time +
            self.grounding.decoder_time +
            self.grounding.postprocess_time)

        # Compute xyxy format boxes
        pred_boxes = results.get("pred_boxes")
        if pred_boxes is not None:
            pred_boxes_xyxy = np.zeros_like(pred_boxes)
            pred_boxes_xyxy[..., 0] = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
            pred_boxes_xyxy[..., 1] = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
            pred_boxes_xyxy[..., 2] = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
            pred_boxes_xyxy[..., 3] = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
            results["pred_boxes_xyxy"] = pred_boxes_xyxy

        # Timing info
        results["timing"] = {
            "preprocess_ms": self.preprocess_time * 1000,
            "vit_ms": self.vit_time * 1000,
            "neck_ms": self.neck_time * 1000,
            "text_ms": self.text_encode_time * 1000,
            "grounding_ms": self.grounding_time * 1000,
            "total_ms": (self.vit_time + self.neck_time +
                         self.text_encode_time + self.grounding_time +
                         self.preprocess_time) * 1000,
        }
        results["text_features"] = text_features

        return results

    def _run_grounding_cpu(self, fpn_feats, fpn_pos,
                           text_features, text_mask, inputs_embeds,
                           text_prompts, boxes, box_labels):
        """CPU grounding fallback for unsupported TPU cases (box prompts, batch>1).

        Uses the full SAM3 PyTorch model via forward_grounding(),
        reusing pre-computed backbone features (ViT + Neck).
        """
        import torch
        from sam3.model.data_misc import FindStage
        from sam3.model.geometry_encoders import Prompt

        self.grounding._ensure_cpu_model()
        model = self.grounding.cpu_model
        device = torch.device('cpu')
        batch_size = len(text_prompts)

        # Convert numpy features to torch tensors
        backbone_fpn = [torch.from_numpy(f.copy()).float()
                       for f in fpn_feats]
        vision_pos_enc = [torch.from_numpy(p.copy()).float()
                         for p in fpn_pos]
        language_features = torch.from_numpy(text_features.copy()).float()
        language_mask = torch.from_numpy(
            text_mask.copy() if isinstance(text_mask, np.ndarray)
            else np.array(text_mask)).bool()

        backbone_out = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": vision_pos_enc,
            "language_features": language_features,
            "language_mask": language_mask.to(device),
        }

        # Empty geometric prompt for text-only detection
        geometric_prompt = model._get_dummy_prompt(num_prompts=1)

        pred_boxes_list = []
        pred_scores_list = []
        pred_logits_list = []
        for i in range(batch_size):
            find_input = FindStage(
                img_ids=torch.tensor([0], dtype=torch.long),
                text_ids=torch.tensor([i], dtype=torch.long),
                input_boxes=torch.empty((0, 4), dtype=torch.float),
                input_boxes_mask=torch.empty((0,), dtype=torch.bool),
                input_boxes_label=torch.empty((0,), dtype=torch.long),
                input_points=torch.empty((0, 2), dtype=torch.float),
                input_points_mask=torch.empty((0,), dtype=torch.bool),
            )

            # Model modifies backbone_out in-place (pops backbone_fpn).
            # Make a fresh copy for each iteration.
            bo_copy = {
                "backbone_fpn": [f.clone() for f in backbone_fpn],
                "vision_pos_enc": [p.clone() for p in vision_pos_enc],
                "language_features": language_features.clone(),
                "language_mask": language_mask.clone(),
            }

            with torch.no_grad():
                out = model.forward_grounding(
                    backbone_out=bo_copy,
                    find_input=find_input,
                    find_target=None,
                    geometric_prompt=geometric_prompt,
                )

            pred_boxes = out["pred_boxes"].cpu().numpy()
            pred_scores = out["pred_logits"].sigmoid().cpu().numpy()
            pred_logits = out["pred_logits"].cpu().numpy()

            # forward_grounding returns [B, 200, 4/1], take best
            best_idx = pred_scores[0, :, 0].argmax()
            pred_boxes_list.append(pred_boxes[0:1, best_idx:best_idx+1])
            pred_scores_list.append(pred_scores[0:1, best_idx:best_idx+1])
            pred_logits_list.append(pred_logits[0:1, best_idx:best_idx+1])

        t0 = time.time()
        results = {
            "pred_logits": np.concatenate(pred_logits_list, axis=0),
            "pred_scores": np.concatenate(pred_scores_list, axis=0),
            "pred_boxes": np.concatenate(pred_boxes_list, axis=0),
        }
        self.grounding.postprocess_time = time.time() - t0
        return results

    def print_timing(self):
        t = {
            "preprocess_ms": self.preprocess_time * 1000,
            "vit_ms": self.vit_time * 1000,
            "neck_ms": self.neck_time * 1000,
            "text_ms": self.text_encode_time * 1000,
            "grounding_ms": self.grounding_time * 1000,
            "total_ms": (self.preprocess_time + self.vit_time +
                         self.neck_time + self.text_encode_time +
                         self.grounding_time) * 1000,
        }
        logger.info("=" * 50)
        logger.info("SAM3 Inference Timing:")
        logger.info(f"  ViT backbone (TPU):    {t['vit_ms']:7.1f} ms")
        logger.info(f"  Neck FPN (TPU):        {t['neck_ms']:7.1f} ms")
        te_mode = "TPU" if self.engines.text_encoder is not None else "CPU"
        logger.info(f"  Text encoder ({te_mode}):     {t['text_ms']:7.1f} ms")
        logger.info(f"  Grounding model (TPU):  {t['grounding_ms']:7.1f} ms")
        logger.info(f"  {'─' * 40}")
        logger.info(f"  Total:                  {t['total_ms']:7.1f} ms")

    def close(self):
        """Free all TPU engines."""
        self.engines.close()


# ============================================================
# Visualization
# ============================================================

def draw_results(image, results, score_thresh, output_path, prompts):
    """Draw top-1 detection box + mask on image."""
    import cv2

    img_h, img_w = image.shape[:2]
    draw = image.copy()

    for i, prompt in enumerate(prompts):
        scores = results["pred_scores"][i, :, 0]
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score < score_thresh:
            continue

        box = results["pred_boxes"][i, best_idx]
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)

        color = (0, 255, 0)

        # Mask overlay (top-1 query logit > 0), if the mask path produced one.
        # pred_masks is [B, Q, H, W]; [i, best_idx] is the (Hm, Wm) logit mask.
        if "pred_masks" in results:
            m = np.asarray(results["pred_masks"][i, best_idx])
            m_bin = (m > 0).astype(np.uint8)
            if m_bin.any():
                m_full = cv2.resize(m_bin, (img_w, img_h),
                                    interpolation=cv2.INTER_NEAREST)
                color_layer = np.zeros_like(draw)
                color_layer[m_full > 0] = color
                draw = cv2.addWeighted(color_layer, 0.4, draw, 1.0, 0)

        cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
        label = f"{prompt}: {best_score:.2f}"
        cv2.putText(draw, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not cv2.imwrite(output_path, draw):
        logger.error("Failed to save result image to: %s", output_path)
    else:
        logger.info("Result image saved to: %s", output_path)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Full Inference Pipeline")
    parser.add_argument("--model_dir", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "models", "BM1684X_504"),
                        help="Path to compiled bmodel directory")
    parser.add_argument("--precision", type=str, default="f16",
                        choices=["f32", "f16"])
    parser.add_argument("--dev_id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=504)
    parser.add_argument("--grid", type=int, default=36,
                        help="Feature grid size (36 for 504, 72 for 1008)")
    parser.add_argument("--image", type=str, default=None,
                        help="Input image path")
    parser.add_argument("--prompt", type=str, default="a person",
                        help="Text prompt for detection")
    parser.add_argument("--score_thresh", type=float, default=0.3,
                        help="Score threshold for detection")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="SAM3 PyTorch checkpoint path")
    parser.add_argument("--bpe_path", type=str, default=None,
                        help="BPE tokenizer vocabulary path")
    parser.add_argument("--mode", type=str, default="bmodel",
                        choices=["bmodel", "onnx"],
                        help="bmodel=TPU sail, onnx=onnxruntime CPU (no device needed)")
    args = parser.parse_args()

    pipeline = SAM3Pipeline(
        args.model_dir, args.precision, args.dev_id,
        resolution=args.resolution,
        bpe_path=args.bpe_path, ckpt_path=args.ckpt_path, mode=args.mode)

    if args.image is None:
        args.image = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "datasets", "truck.jpg")

    import cv2
    image = cv2.imread(args.image)
    if image is None:
        logger.error("Failed to load image: %s", args.image)
        sys.exit(1)

    logger.info("Loaded image: %s from %s", image.shape, args.image)

    prompts = args.prompt.split(",") if "," in args.prompt else [args.prompt]
    logger.info("Prompts: %s", prompts)

    pipeline.set_image(image)
    results = pipeline.predict(prompts)

    pipeline.print_timing()

    # Show detections
    logger.info("\nDetections (score > %.1f):", args.score_thresh)
    for i, prompt in enumerate(prompts):
        scores = results["pred_scores"][i, :, 0]
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        box = results["pred_boxes"][i, best_idx]
        if best_score >= args.score_thresh:
            logger.info("  %-24s %.4f  (cx=%.3f, cy=%.3f, w=%.3f, h=%.3f)",
                        prompt, best_score, box[0], box[1], box[2], box[3])
        else:
            logger.info("  %-24s %.4f  (no detection)", prompt, best_score)

    logger.info("")
    logger.info("Detected %d/%d prompts", int(sum(
        1 for i in range(len(prompts))
        if results["pred_scores"][i, :, 0].max() >= args.score_thresh)),
                len(prompts))

    # Save result image
    if args.output is None:
        os.makedirs("results", exist_ok=True)
        args.output = os.path.join("results", "sam3_detection.jpg")
    draw_results(image, results, args.score_thresh, args.output, prompts)

    pipeline.close()


if __name__ == "__main__":
    main()
