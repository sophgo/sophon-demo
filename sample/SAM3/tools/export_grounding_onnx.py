#!/usr/bin/env python3
"""
Export SAM3 Grounding Model (Encoder + Decoder) to ONNX.
=========================================================
The grounding model consists of:
  1. TransformerEncoderFusion (6 layers, image-text cross-attention)
  2. TransformerDecoder (6 layers, 200 queries, cross-attention to memory)
  3. DotProductScoring / ClassEmbed (text-query similarity)
  4. BBox Embedding (MLP)

This script exports the encoder and decoder as separate ONNX models
at 504x504 resolution (grid=36, 1296 feature tokens).

Usage:
  python export_grounding_onnx.py --checkpoint ../models/sam3.pt --output_dir ../models/onnx_504

Author: liheng.fang
Date: 2025-06-23
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/home/lihengfang/work/git_commits/developer')


# ============================================================
# Patch SAM3 model for CPU compatibility (same as ViT export)
# ============================================================

def patch_sam3_model():
    """Apply all necessary patches for ONNX export."""
    from sam3.model.vitdet import Mlp

    # Patch Mlp to avoid BFloat16 fused operation
    def patched_mlp_forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop1(x)
        x = self.norm(x); x = self.fc2(x); x = self.drop2(x)
        return x
    Mlp.forward = patched_mlp_forward

    # Patch PositionEmbeddingSine to avoid CUDA precomputation
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

    # Patch torch.arange for ONNX compatibility
    _orig_arange = torch.arange
    def _onnx_safe_arange(*args, **kwargs):
        kwargs.pop('device', None)
        return _orig_arange(*args, **kwargs)
    torch.arange = _onnx_safe_arange

    # Patch activation checkpointing to be a no-op during export
    from sam3.model import act_ckpt_utils
    def _noop_act_ckpt_wrapper(module):
        def wrapper(*args, act_ckpt_enable=True, use_reentrant=False, **kwargs):
            return module(*args, **kwargs)
        return wrapper
    act_ckpt_utils.activation_ckpt_wrapper = _noop_act_ckpt_wrapper

    # Disable autocast which causes tracer issues
    torch.set_autocast_enabled(False)

    # Patch sdpa_kernel (SDPBackend) - ONNX tracer doesn't support context managers
    import contextlib
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _orig_sdpa_kernel = sdpa_kernel
    def _noop_sdpa_kernel(*args, **kwargs):
        return contextlib.nullcontext()
    # Monkey-patch at module level
    import sam3.model.decoder as decoder_mod
    decoder_mod.sdpa_kernel = _noop_sdpa_kernel
    decoder_mod.SDPBackend = SDPBackend  # keep the enum

    # Patch forward_ffn to remove autocast
    from sam3.model.decoder import TransformerDecoderLayer
    _orig_forward_ffn = TransformerDecoderLayer.forward_ffn
    def _patched_forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    TransformerDecoderLayer.forward_ffn = _patched_forward_ffn

    # Patch TransformerDecoder._get_coords to force int H, W so that
    # torch.arange(0, H) traces as a static Range (constant) instead of a
    # dynamic Range op whose end comes from the spatial_shapes input.
    # tpu-mlir's RangeOp::shape_inference() aborts on that dynamic form.
    # At 504x504 resolution H=W=36 are fixed.
    from sam3.model.decoder import TransformerDecoder
    _orig_get_coords = TransformerDecoder._get_coords
    def _static_get_coords(H, W, device):
        H = int(H); W = int(W)
        return _orig_get_coords(H, W, device)
    TransformerDecoder._get_coords = staticmethod(_static_get_coords)

    print("[patch] SAM3 model patched for ONNX export")


# ============================================================
# Load SAM3 model
# ============================================================

def load_model(checkpoint_path):
    from sam3.model_builder import build_sam3_image_model

    bpe_path = '/home/lihengfang/work/git_commits/developer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'

    print(f"Loading SAM3 model from {checkpoint_path}...")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        device='cpu',
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=False,
    )
    model = model.float().eval()
    print(f"  Model loaded. num_feature_levels={model.num_feature_levels}")
    return model


# ============================================================
# Encoder Wrapper
# ============================================================

class EncoderWrapper(nn.Module):
    """
    Wraps TransformerEncoderFusion for ONNX export.

    Inputs:
      - src: image features [1, C, H, W]
      - src_pos: image position encoding [1, C, H, W]
      - prompt: text features [T, 1, C] (T=32 text tokens)
      - prompt_mask: text mask [1, T] (True=ignore)

    Outputs:
      - memory: encoded features [N, 1, C] (N=H*W feature tokens)
      - pos_embed: position embedding [N, 1, C]
      - padding_mask: padding mask [1, N]
      - level_start_index: [1] (only 1 level)
      - spatial_shapes: [1, 2] (H, W)
      - valid_ratios: [1, 1, 2]
    """

    def __init__(self, encoder, num_feature_levels):
        super().__init__()
        self.encoder = encoder
        self.num_feature_levels = num_feature_levels

    def forward(self, src, src_pos, prompt, prompt_mask):
        # src/src_pos are batch-first [1, C, H, W] (from ViT+Neck output)
        # The encoder expects seq-first (HW, batch, C) format
        # Same conversion as sam3_image.py:_get_img_feats line 129
        bs, c, h, w = src.shape
        feat_sizes = [(h, w)]

        # Convert from batch-first to seq-first: (bs, c, h, w) -> (hw, bs, c)
        # Use explicit reshape with hardcoded dimensions to avoid dynamic
        # Shape->Slice->Concat->Reshape patterns that crash tpu-mlir.
        # At 504x504 resolution: h=w=36, h*w=1296, c=256
        hw = h * w
        img_feats = [src.reshape(bs, c, hw).permute(2, 0, 1)]
        img_pos_embeds = [src_pos.reshape(bs, c, hw).permute(2, 0, 1)]

        # prompt is batch-first [1, T, C], convert to seq-first [T, 1, C]
        prompt_t = prompt.permute(1, 0, 2)  # [1, T, C] -> [T, 1, C]

        # Call the encoder
        memory = self.encoder(
            src=img_feats,
            src_key_padding_mask=None,
            src_pos=img_pos_embeds,
            prompt=prompt_t,
            prompt_pos=torch.zeros_like(prompt_t),
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=feat_sizes,
        )

        return (
            memory["memory"],
            memory["pos_embed"],
            memory["level_start_index"],
            memory["spatial_shapes"],
            memory["valid_ratios"],
        )


# ============================================================
# Decoder Wrapper
# ============================================================

class DecoderWrapper(nn.Module):
    """
    Wraps TransformerDecoder for ONNX export.

    Inputs:
      - memory: encoder output [N, 1, C]
      - memory_pos: position embedding [N, 1, C]
      - memory_mask: padding mask [N, 1] or None
      - level_start_index, spatial_shapes, valid_ratios
      - prompt: text features [T, 1, C]
      - prompt_mask: text mask [1, T]

    Outputs:
      - hs: hidden states [6, 200, 1, C]
      - reference_boxes: [6, 200, 1, 4+256] — boxes [:...,:4] PLUS the
        last-layer presence-token hidden state [1,1,C] smuggled into channels
        [4:] (broadcast). TPU-MLIR v3.4 bmodel-save FATALs on any 3rd decoder
        output (post-head logits AND pre-head hidden state both fail at codegen
        Save:424), so presence_feats is fused into reference_boxes to keep 2
        outputs. On CPU: boxes=ref[...,:4], presence_feats=ref[0,0,0,4:]; run
        head (presence_token_out_norm + presence_token_head) for the last-layer
        presence logit. Final score = sigmoid(orig) * sigmoid(presence_logit).
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.num_queries = decoder.num_queries

    def forward(self, memory, memory_pos, memory_mask,
                level_start_index, spatial_shapes, valid_ratios,
                prompt, prompt_mask):
        bs = memory.shape[1]
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # prompt is batch-first [1, T, C], convert to seq-first [T, 1, C]
        prompt_t = prompt.permute(1, 0, 2)  # [1, T, C] -> [T, 1, C]

        hs, reference_boxes, _dec_presence_logits, presence_feats = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=memory_mask,
            pos=memory_pos,
            reference_boxes=None,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=None,
            memory_text=prompt_t,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )

        # Return 2 outputs matching the downloaded/reference decoder ONNX:
        # hs [6,200,1,256] and reference_boxes [6,200,1,4]. The presence-token
        # hidden state (presence_feats) is intentionally NOT returned — the
        # host post-processing in sam3_infer.py only consumes hs + 4-ch
        # reference_boxes (dot-prod scoring + box refinement). Exposing a 3rd
        # output previously tripped TPU-MLIR v3.4 bmodel-save FATAL (Save:424);
        # 2 outputs avoids that and keeps the bmodel interface identical to
        # the pre-existing downloaded decoder bmodel.
        return hs, reference_boxes


# ============================================================
# Scoring Wrapper (DotProductScoring)
# ============================================================

class ScoringWrapper(nn.Module):
    """
    Wraps DotProductScoring + BBox head for ONNX export.

    Inputs:
      - hs: decoder hidden states [6, 200, 1, C] or [200, 1, C]
      - prompt: text features [T, 1, C]
      - prompt_mask: text mask [1, T]
      - reference_boxes: [200, 1, 4] (last layer)

    Outputs:
      - pred_scores: [1, 200, 1]
      - pred_boxes: [1, 200, 4]
    """

    def __init__(self, model, decoder):
        super().__init__()
        self.dot_prod_scoring = model.dot_prod_scoring
        self.bbox_embed = decoder.bbox_embed

    def forward(self, hs, prompt, prompt_mask, reference_boxes):
        # Take last layer
        hs_last = hs[-1]  # [200, 1, C]

        # Dot product scoring
        scores = self.dot_prod_scoring(hs, prompt, prompt_mask)

        # BBox offsets → boxes
        offsets = self.bbox_embed(hs)  # [6, 200, 1, 4]

        # Box refinement
        # reference_boxes + offset → final boxes
        # This follows sam3_image._update_scores_and_boxes logic
        from sam3.model.model_misc import inverse_sigmoid
        ref_inv = inverse_sigmoid(reference_boxes[-1:])  # [1, 200, 1, 4]
        outputs_coord = (ref_inv + offsets[-1:]).sigmoid()  # [1, 200, 1, 4]

        return scores[-1:], outputs_coord


# ============================================================
# Export functions
# ============================================================

def export_encoder(model, output_dir, grid=36):
    """Export the TransformerEncoderFusion to ONNX."""
    print("\n=== Exporting Encoder ===")

    encoder = model.transformer.encoder
    wrapper = EncoderWrapper(encoder, model.num_feature_levels)
    wrapper.eval()

    H = W = grid
    C = 256  # encoder dim

    # Create dummy inputs
    src = torch.randn(1, C, H, W, dtype=torch.float32)
    src_pos = torch.randn(1, C, H, W, dtype=torch.float32)
    prompt = torch.randn(1, 32, C, dtype=torch.float32)  # batch-first: [1, T, C]
    prompt_mask = torch.zeros(1, 32, dtype=torch.bool)

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "sam3_grounding_encoder.onnx")

    print(f"  Input shapes:")
    print(f"    src:       {list(src.shape)}")
    print(f"    src_pos:   {list(src_pos.shape)}")
    print(f"    prompt:    {list(prompt.shape)}")
    print(f"    prompt_mask: {list(prompt_mask.shape)}")

    torch.onnx.export(
        wrapper,
        (src, src_pos, prompt, prompt_mask),
        onnx_path,
        input_names=["src", "src_pos", "prompt", "prompt_mask"],
        output_names=["memory", "pos_embed",
                      "level_start_index", "spatial_shapes", "valid_ratios"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"  Exported: {onnx_path}")

    # Verify
    with torch.no_grad():
        out = wrapper(src, src_pos, prompt, prompt_mask)
    print(f"  Output shapes:")
    for name, val in zip(
        ["memory", "pos_embed", "level_start_index",
         "spatial_shapes", "valid_ratios"], out):
        if isinstance(val, torch.Tensor):
            print(f"    {name}: {list(val.shape)}")

    return onnx_path


def export_decoder(model, output_dir, grid=36):
    """Export the TransformerDecoder to ONNX."""
    print("\n=== Exporting Decoder ===")

    decoder = model.transformer.decoder
    wrapper = DecoderWrapper(decoder)
    wrapper.eval()

    H = W = grid
    C = 256
    N = H * W  # total feature tokens
    T = 32     # text tokens
    num_queries = decoder.num_queries  # 200

    # Dummy inputs matching encoder output
    memory = torch.randn(N, 1, C, dtype=torch.float32)
    memory_pos = torch.randn(N, 1, C, dtype=torch.float32)
    memory_mask = torch.zeros(N, 1, dtype=torch.bool)
    level_start_index = torch.zeros(1, dtype=torch.int64)
    spatial_shapes = torch.tensor([[H, W]], dtype=torch.int64)
    valid_ratios = torch.ones(1, 1, 2, dtype=torch.float32)
    prompt = torch.randn(1, T, C, dtype=torch.float32)  # batch-first: [1, T, C]
    prompt_mask = torch.zeros(1, T, dtype=torch.bool)

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "sam3_grounding_decoder.onnx")

    print(f"  Input shapes:")
    print(f"    memory:       {list(memory.shape)}")
    print(f"    memory_pos:   {list(memory_pos.shape)}")
    print(f"    memory_mask:  {list(memory_mask.shape)}")
    print(f"    prompt:       {list(prompt.shape)}")
    print(f"    prompt_mask:  {list(prompt_mask.shape)}")
    print(f"    num_queries:  {num_queries}")

    torch.onnx.export(
        wrapper,
        (memory, memory_pos, memory_mask,
         level_start_index, spatial_shapes, valid_ratios,
         prompt, prompt_mask),
        onnx_path,
        input_names=["memory", "memory_pos", "memory_mask",
                     "level_start_index", "spatial_shapes", "valid_ratios",
                     "prompt", "prompt_mask"],
        output_names=["hs", "reference_boxes"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"  Exported: {onnx_path}")

    # Verify
    with torch.no_grad():
        hs, ref_boxes = wrapper(
            memory, memory_pos, memory_mask,
            level_start_index, spatial_shapes, valid_ratios,
            prompt, prompt_mask)
    print(f"  Output shapes:")
    print(f"    hs:             {list(hs.shape)}")
    print(f"    reference_boxes:{list(ref_boxes.shape)}  ([:4]=boxes, [4:]=presence_feats)")

    return onnx_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Export SAM3 Grounding Model to ONNX")
    parser.add_argument("--checkpoint", type=str,
                        default="../models/sam3.pt",
                        help="Path to SAM3 checkpoint")
    parser.add_argument("--output_dir", type=str,
                        default="../models/onnx_504",
                        help="Output directory for ONNX files")
    parser.add_argument("--grid", type=int, default=36,
                        help="Feature grid size (36 for 504x504)")
    parser.add_argument("--encoder_only", action="store_true",
                        help="Export only the encoder")
    parser.add_argument("--decoder_only", action="store_true",
                        help="Export only the decoder")
    args = parser.parse_args()

    # Ensure output directory is absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = os.path.abspath(args.output_dir)
    checkpoint = os.path.abspath(args.checkpoint)

    print("=" * 60)
    print("SAM3 Grounding Model ONNX Export")
    print(f"  Grid: {args.grid}x{args.grid}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Patch and load model
    patch_sam3_model()
    model = load_model(checkpoint)

    export_all = not args.encoder_only and not args.decoder_only

    if export_all or args.encoder_only:
        export_encoder(model, output_dir, args.grid)

    if export_all or args.decoder_only:
        export_decoder(model, output_dir, args.grid)

    print("\nDone!")


if __name__ == "__main__":
    main()
