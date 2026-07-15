#!/usr/bin/env python3
"""
SAM3 Text Encoder ONNX Export
==============================
Exports the SAM3 Text Encoder (VETextEncoder) to ONNX for TPU-MLIR compilation.

The text encoder converts tokenized text to multi-modal features:
  TextTransformer (embedding + 24-layer Transformer + LN + projection) → [B, seq_len, 1024]
  Resizer (Linear) → [seq_len, B, 256]

The CLIP tokenizer runs on CPU as a preprocessing step, producing token IDs
that are fed to the ONNX model.

Architecture:
  TextTransformer:
    - token_embedding: Embedding(vocab_size=49408, dim=1024)
    - positional_embedding: Parameter([77, 1024])
    - transformer: 24 × ResidualAttentionBlock(width=1024, heads=16)
    - ln_final: LayerNorm(1024)
    - text_projection: None or Linear(1024, output_dim)

  ResidualAttentionBlock (per layer):
    - Self-Attention (16 heads, 1024 dim)
    - MLP (4096 hidden dim with GELU)
    - LayerScale (two learnable scalars)

  Resizer: Linear(1024 → 256)

Usage:
  python export_text_encoder_onnx.py --output_dir ../models/onnx

Author: liheng.fang
Date: 2025-06-17
"""

import os
import sys
import argparse

import torch
import torch.nn as nn

torch.set_autocast_enabled(False)


def patch_mlp_addmm_act():
    """Patch Mlp to use standard Linear+GELU (needed for ViT part)."""
    from sam3.model.vitdet import Mlp

    def patched_forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    Mlp.forward = patched_forward
    print("[Patch] Mlp.addmm_act → standard Linear + GELU")


def convert_rope_to_real(vit_trunk):
    """Convert complex RoPE frequencies (needed for model loading)."""
    if hasattr(vit_trunk, 'blocks'):
        for blk in vit_trunk.blocks:
            attn = blk.attn
            if hasattr(attn, 'freqs_cis') and attn.freqs_cis is not None:
                fc = attn.freqs_cis
                attn.register_buffer('freqs_cis_real', fc.real.float().detach().clone())
                attn.register_buffer('freqs_cis_imag', fc.imag.float().detach().clone())
                attn.use_rope_real = True
    print("[Patch] RoPE complex → real conversion done")


class TextEncoderExportWrapper(nn.Module):
    """
    Export wrapper for the text encoder.

    Takes token IDs [batch, seq_len] and produces:
      - tokens: [seq_len, batch, 256]  (resized text features, sequence-first for transformer input)
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.encoder = text_encoder.encoder  # TextTransformer
        self.resizer = text_encoder.resizer  # Linear(1024 → 256)

    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch, seq_len] int64 token IDs
        Returns:
            text_features: [seq_len, batch, 256] resized text features
        """
        # Get the text memory from TextTransformer (output_tokens=True)
        _, text_memory = self.encoder(token_ids)
        # text_memory: [batch, seq_len, 1024]

        # Transpose to sequence-first (matching the original VETextEncoder)
        text_memory = text_memory.transpose(0, 1)  # [seq_len, batch, 1024]

        # Apply resizer
        text_features = self.resizer(text_memory)  # [seq_len, batch, 256]

        return text_features


def load_model():
    """Load SAM3 model to extract text encoder."""
    sys.path.insert(0, '/home/lihengfang/work/git_commits/developer')
    from sam3.model_builder import build_sam3_image_model

    bpe_path = '/home/lihengfang/work/git_commits/developer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
    ckpt_path = '/home/lihengfang/work/git_commits/developer/sophon-demo/sample/SAM3/models/sam3.pt'

    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=ckpt_path,
        device='cpu',
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=False,
    )
    model = model.float().eval()
    print(f"Model loaded: {type(model).__name__}")

    return model


def export_text_encoder(model, output_path, context_length=77):
    """Export the text encoder to ONNX."""
    text_encoder = model.backbone.language_backbone  # VETextEncoder

    print(f"\n[Text Encoder Analysis]")
    enc = text_encoder.encoder
    print(f"  vocab_size: {enc.token_embedding.weight.shape[0]}")
    print(f"  embed_dim: {enc.token_embedding.weight.shape[1]}")
    print(f"  positional_embedding: {list(enc.positional_embedding.shape)}")
    print(f"  transformer layers: {enc.transformer.layers}")
    print(f"  resizer: {text_encoder.resizer.in_features} → {text_encoder.resizer.out_features}")

    # Use the actual context_length from the model (32, not CLIP's 77)
    actual_context_length = enc.context_length
    print(f"  context_length: {actual_context_length}")

    # Check for output_tokens
    if not enc.output_tokens:
        print("  [WARNING] output_tokens=False, text encoder won't return token features!")
        print("  [FIX] Setting output_tokens=True for export...")
        enc.output_tokens = True

    # Check for text_projection - if it exists, it's applied to pooled output
    if enc.text_projection is not None:
        print(f"  text_projection: {list(enc.text_projection.shape)}")

    wrapper = TextEncoderExportWrapper(text_encoder)
    wrapper.eval()

    # Test with dummy token IDs (use model's context_length, not CLIP's 77)
    dummy_tokens = torch.randint(0, enc.vocab_size, (1, actual_context_length), dtype=torch.int64)
    print(f"\n  Dummy input: {list(dummy_tokens.shape)}, dtype={dummy_tokens.dtype}")

    with torch.no_grad():
        output = wrapper(dummy_tokens)
        print(f"  Output: {list(output.shape)}, dtype={output.dtype}")

    # Export (sequence length = 32 is fixed, batch is dynamic)
    print(f"\n  Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_tokens,
        output_path,
        opset_version=16,
        input_names=['token_ids'],
        output_names=['text_features'],
        dynamic_axes={
            'token_ids': {0: 'batch'},
            'text_features': {1: 'batch'},
        },
        dynamo=False,
    )

    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ONNX size: {model_size_mb:.1f} MB")

    # Validate
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX validation: PASSED")

    # Op stats
    ops = {}
    for node in onnx_model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"  Op types ({len(ops)} unique):")
    for op, count in sorted(ops.items(), key=lambda x: -x[1])[:20]:
        print(f"    {op}: {count}")


def main():
    parser = argparse.ArgumentParser(description='SAM3 Text Encoder ONNX Export')
    parser.add_argument('--output_dir', type=str, default='../models/onnx',
                        help='Output directory for ONNX files')
    parser.add_argument('--context_length', type=int, default=77,
                        help='Maximum context length for the text encoder')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SAM3 Text Encoder ONNX Export")
    print("=" * 60)

    # Apply patches
    print("\n[Step 1] Applying patches...")
    patch_mlp_addmm_act()

    # Load model
    print("\n[Step 2] Loading model...")
    model = load_model()

    # Convert RoPE
    print("\n[Step 3] Converting RoPE...")
    convert_rope_to_real(model.backbone.vision_backbone.trunk)

    # Export
    print("\n[Step 4] Exporting text encoder...")
    output_path = os.path.join(args.output_dir, 'sam3_text_encoder.onnx')
    export_text_encoder(model, output_path, args.context_length)

    print("\n" + "=" * 60)
    print(f"Done! ONNX file: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
