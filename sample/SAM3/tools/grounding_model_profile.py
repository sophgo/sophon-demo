#!/usr/bin/env python3
"""
SAM3 Grounding Model CPU Performance Test
==========================================
Measure the CPU inference time of the grounding model components:
  - Transformer Encoder (vision-text cross-attention, 6 layers)
  - Transformer Decoder (object-query cross-attention, 6 layers, 200 queries)
  - DotProductScoring (text-query similarity)
  - BBox Embedding (MLP)
  - Geometry Encoder

This helps decide whether TPU porting is beneficial for these components.

Usage:
  python grounding_model_profile.py
"""

import sys
import time
import torch
import numpy as np

torch.set_autocast_enabled(False)


def patch_all():
    from sam3.model.vitdet import Mlp
    def patched_mlp_forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop1(x)
        x = self.norm(x); x = self.fc2(x); x = self.drop2(x)
        return x
    Mlp.forward = patched_mlp_forward


def load_model():
    sys.path.insert(0, '/home/lihengfang/work/git_commits/developer')
    from sam3.model_builder import build_sam3_image_model

    bpe_path = '/home/lihengfang/work/git_commits/developer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
    ckpt_path = '/home/lihengfang/work/git_commits/developer/sophon-demo/sample/SAM3/models/sam3.pt'

    print("Loading model...")
    t0 = time.time()
    model = build_sam3_image_model(
        bpe_path=bpe_path, checkpoint_path=ckpt_path, device='cpu',
        load_from_HF=False, enable_segmentation=False, enable_inst_interactivity=False,
    )
    model = model.float().eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model


def profile_components(model):
    """Measure inference time of each grounding component."""
    device = torch.device('cpu')
    warmup = 3
    repeat = 10

    # Create realistic dummy inputs
    # Neck outputs: the encoder uses only the last FPN level (num_feature_levels=1)
    # scale 0.5: [B, 256, 36, 36] → flattened: [36*36=1296, B, 256]
    num_feature_levels = model.num_feature_levels
    print(f"num_feature_levels: {num_feature_levels}")

    feat_s05 = torch.randn(1, 256, 36, 36, device=device)

    img_feats = [
        feat_s05.flatten(2).permute(2, 0, 1),  # [1296, 1, 256]
    ]
    img_pos_embeds = [f.clone() for f in img_feats]
    feat_sizes = [(36, 36)]

    # Text features
    txt_feats = torch.randn(32, 1, 256, device=device)
    txt_mask = torch.zeros(1, 32, dtype=torch.bool, device=device)

    # Combine prompt
    prompt = txt_feats  # simplified: no geo prompt
    prompt_mask = txt_mask

    # ============================================================
    # 1. Transformer Encoder
    # ============================================================
    encoder = model.transformer.encoder
    print("\n=== 1. Transformer Encoder (6 layers) ===")

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            enc_out = encoder(
                src=img_feats.copy(),
                src_key_padding_mask=None,
                src_pos=img_pos_embeds.copy(),
                prompt=prompt,
                prompt_pos=torch.zeros_like(prompt),
                prompt_key_padding_mask=prompt_mask,
                feat_sizes=feat_sizes,
            )

    # Measure
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        with torch.no_grad():
            enc_out = encoder(
                src=img_feats.copy(),
                src_key_padding_mask=None,
                src_pos=img_pos_embeds.copy(),
                prompt=prompt,
                prompt_pos=torch.zeros_like(prompt),
                prompt_key_padding_mask=prompt_mask,
                feat_sizes=feat_sizes,
            )
        times.append(time.perf_counter() - t0)

    enc_time = np.mean(times) * 1000
    print(f"  Avg: {enc_time:.1f} ms")
    print(f"  Memory shape: {list(enc_out['memory'].shape)}")
    print(f"  Memory params: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M")

    memory = enc_out['memory']
    src_mask = enc_out['padding_mask']
    pos_embed = enc_out['pos_embed']

    # ============================================================
    # 2. Transformer Decoder
    # ============================================================
    decoder = model.transformer.decoder
    bs = 1
    query_embed = decoder.query_embed.weight
    tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_queries, 1, 256]
    print(f"  Num queries: {tgt.shape[0]}")

    print(f"\n=== 2. Transformer Decoder (6 layers, 200 queries) ===")
    print(f"  Query shape: {list(tgt.shape)}")
    print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M")

    for _ in range(warmup):
        with torch.no_grad():
            dec_out = decoder(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=src_mask,
                pos=pos_embed,
                reference_boxes=None,
                level_start_index=enc_out['level_start_index'],
                spatial_shapes=enc_out['spatial_shapes'],
                valid_ratios=enc_out['valid_ratios'],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=False,
            )

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        with torch.no_grad():
            dec_out = decoder(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=src_mask,
                pos=pos_embed,
                reference_boxes=None,
                level_start_index=enc_out['level_start_index'],
                spatial_shapes=enc_out['spatial_shapes'],
                valid_ratios=enc_out['valid_ratios'],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=False,
            )
        times.append(time.perf_counter() - t0)

    dec_time = np.mean(times) * 1000
    hs, reference_boxes, dec_presence_out, dec_presence_feats = dec_out
    print(f"  Avg: {dec_time:.1f} ms")
    print(f"  hs (hidden states): {list(hs.shape)}")
    print(f"  reference_boxes: {list(reference_boxes.shape)}")

    # ============================================================
    # 3. DotProduct Scoring
    # ============================================================
    print(f"\n=== 3. DotProduct Scoring ===")
    scorer = model.dot_prod_scoring
    print(f"  Scorer params: {sum(p.numel() for p in scorer.parameters())/1e6:.3f}M")

    for _ in range(warmup):
        with torch.no_grad():
            scores = scorer(hs, prompt, prompt_mask)

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        with torch.no_grad():
            scores = scorer(hs, prompt, prompt_mask)
        times.append(time.perf_counter() - t0)

    score_time = np.mean(times) * 1000
    print(f"  Avg: {score_time:.1f} ms")
    print(f"  Scores shape: {list(scores.shape)}")

    # ============================================================
    # 4. BBox Embedding (MLP)
    # ============================================================
    print(f"\n=== 4. BBox Embedding (MLP) ===")
    bbox_head = decoder.bbox_embed

    for _ in range(warmup):
        with torch.no_grad():
            offsets = bbox_head(hs)

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        with torch.no_grad():
            offsets = bbox_head(hs)
        times.append(time.perf_counter() - t0)

    bbox_time = np.mean(times) * 1000
    print(f"  Avg: {bbox_time:.1f} ms")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*50}")
    print(f"Grounding Model CPU Performance Summary:")
    print(f"  Transformer Encoder:    {enc_time:.1f} ms")
    print(f"  Transformer Decoder:    {dec_time:.1f} ms")
    print(f"  DotProduct Scoring:     {score_time:.1f} ms")
    print(f"  BBox Embedding:         {bbox_time:.1f} ms")
    total = enc_time + dec_time + score_time + bbox_time
    print(f"  {'─'*40}")
    print(f"  Total Grounding:        {total:.1f} ms")


if __name__ == '__main__':
    patch_all()
    model = load_model()
    profile_components(model)
