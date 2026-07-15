#!/usr/bin/env python3
"""
SAM3 Text Encoder CPU Inference
=================================
Runs the VETextEncoder on CPU to produce text features for the grounding model.

The text encoder is kept on CPU because:
  1. Text encoding runs once per prompt (not per-image), so TPU benefit is minimal
  2. TPU-MLIR has dynamic shape limitations with transformer self-attention Reshape ops
  3. The text encoder is only ~200M parameters (much smaller than ViT's 640M)

Input:  text strings (e.g., "a person", "a cat sitting on a chair")
Output: text features [32, batch, 256] for cross-attention in grounding model

Usage:
  # Interactive mode
  python text_encoder_infer.py

  # Pre-compute embeddings for common prompts
  python text_encoder_infer.py --save embeddings.npz --prompts "person,cat,dog,car"

  # API usage
  from text_encoder_infer import SAM3TextEncoder
  encoder = SAM3TextEncoder()
  features = encoder.encode(["a person", "a cat"])

Author: liheng.fang
Date: 2025-06-22
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

torch.set_autocast_enabled(False)


class SAM3TextEncoder:
    """
    SAM3 Text Encoder wrapper for CPU inference.

    Encapsulates:
      - CLIP SimpleTokenizer (BPE tokenizer)
      - TextTransformer (embedding + 24-layer transformer + LN)
      - Resizer (Linear 1024 → 256)

    Output shape: [32, batch, 256] — sequence-first format for DETR cross-attention.
    """

    def __init__(self, bpe_path=None, checkpoint_path=None, device='cpu'):
        """
        Args:
            bpe_path: Path to BPE tokenizer vocabulary file
            checkpoint_path: Path to SAM3 checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = device

        # Patch ViT BFloat16 issue (needed when loading the full model)
        sys.path.insert(0, '/home/lihengfang/work/git_commits/developer')
        from sam3.model.vitdet import Mlp

        def patched_mlp_forward(self, x):
            x = self.fc1(x); x = self.act(x); x = self.drop1(x)
            x = self.norm(x); x = self.fc2(x); x = self.drop2(x)
            return x
        Mlp.forward = patched_mlp_forward

        # Default paths
        if bpe_path is None:
            bpe_path = '/home/lihengfang/work/git_commits/developer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
        if checkpoint_path is None:
            checkpoint_path = '/home/lihengfang/work/git_commits/developer/sophon-demo/sample/SAM3/models/sam3.pt'

        self.bpe_path = bpe_path
        self.checkpoint_path = checkpoint_path

        print(f"[TextEncoder] Loading SAM3 model...")
        from sam3.model_builder import build_sam3_image_model

        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
            device=device,
            load_from_HF=False,
            enable_segmentation=False,
            enable_inst_interactivity=False,
        )
        self.model = self.model.float().eval()

        # Extract text encoder components
        self.text_encoder = self.model.backbone.language_backbone  # VETextEncoder
        self.tokenizer = self.text_encoder.tokenizer
        self.context_length = self.text_encoder.context_length
        self.encoder = self.text_encoder.encoder  # TextTransformer
        self.resizer = self.text_encoder.resizer  # Linear(1024 → 256)

        print(f"[TextEncoder] Loaded successfully")
        print(f"  Context length: {self.context_length}")
        print(f"  Vocab size: {self.encoder.vocab_size}")
        print(f"  Embed dim: {self.encoder.width}")
        print(f"  Transformer layers: {self.encoder.transformer.layers}")
        print(f"  Resizer: {self.resizer.in_features} → {self.resizer.out_features}")

    def encode(self, texts, return_numpy=True):
        """
        Encode text strings into features for the grounding model.

        Args:
            texts: str or list of str — text prompt(s)
            return_numpy: bool — if True, return numpy array; else torch.Tensor

        Returns:
            text_features: [32, batch, 256] — text features (sequence-first)
            text_mask: [batch, 32] — attention mask (True = valid token)
            token_ids: [batch, 32] — token IDs

        Examples:
            >>> encoder = SAM3TextEncoder()
            >>> features, mask, tokens = encoder.encode("a person")
            >>> features.shape  # (32, 1, 256)
        """
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            # Use VETextEncoder.forward with string input
            # Returns: (text_attention_mask, text_memory_resized, inputs_embeds)
            text_mask, text_features, inputs_embeds = self.text_encoder(
                texts, device=torch.device(self.device)
            )

            # text_features: [32, batch, 256] — resized text features for cross-attention
            # text_mask: [batch, 32] — attention mask (inverted: True=ignore, False=attend)
            # inputs_embeds: [32, batch, 1024] — raw token embeddings (transposed)

            if return_numpy:
                text_features = text_features.cpu().numpy()
                text_mask = text_mask.cpu().numpy()
                inputs_embeds = inputs_embeds.cpu().numpy()

        return text_features, text_mask, inputs_embeds

    def get_tokenizer(self):
        """Return the tokenizer for standalone use."""
        return self.tokenizer

    def tokenize(self, texts):
        """Tokenize text strings without running the encoder."""
        if isinstance(texts, str):
            texts = [texts]
        tokenized = self.tokenizer(texts, context_length=self.context_length)
        return tokenized


def main():
    parser = argparse.ArgumentParser(description='SAM3 Text Encoder CPU Inference')
    parser.add_argument('--prompts', type=str, default=None,
                        help='Comma-separated list of text prompts to encode')
    parser.add_argument('--save', type=str, default=None,
                        help='Save encoded features to .npz file')
    parser.add_argument('--load', type=str, default=None,
                        help='Load pre-computed embeddings from .npz and print shapes')
    args = parser.parse_args()

    if args.load:
        print(f"Loading embeddings from: {args.load}")
        data = np.load(args.load, allow_pickle=True)
        for key in data.files:
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        return

    # Initialize encoder
    print("=" * 60)
    print("SAM3 Text Encoder CPU Inference")
    print("=" * 60)

    encoder = SAM3TextEncoder()

    # Determine prompts
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(',')]
    else:
        # Default test prompts
        prompts = [
            "a person",
            "a cat",
            "a dog",
            "a car",
            "a chair",
            "a table",
            "a cup",
            "a bottle",
            "a book",
            "a laptop",
        ]

    print(f"\n[Encoding] {len(prompts)} prompts:")
    for p in prompts:
        print(f"  - {p}")

    # Encode
    features, mask, inputs_embeds = encoder.encode(prompts)

    print(f"\n[Result]")
    print(f"  Text features (resized): shape={features.shape}, dtype={features.dtype}")
    print(f"  Text mask: shape={mask.shape}, dtype={mask.dtype}")
    print(f"  Input embeddings: shape={inputs_embeds.shape}, dtype={inputs_embeds.dtype}")

    # Print sample features
    print(f"\n[Sample] First 5 prompts feature stats:")
    for i, prompt in enumerate(prompts):
        feat = features[:, i, :]  # [32, 256]
        print(f"  [{i}] \"{prompt}\": mean={feat.mean():.6f}, std={feat.std():.6f}")

    # Save if requested
    if args.save:
        save_path = args.save
        np.savez(
            save_path,
            text_features=features,
            text_mask=mask,
            inputs_embeds=inputs_embeds,
            prompts=np.array(prompts, dtype=object),
        )
        print(f"\n[Saved] Embeddings saved to: {save_path}")


if __name__ == '__main__':
    main()
