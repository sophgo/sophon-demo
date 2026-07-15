#!/usr/bin/env python3
"""
SAM3 ViT Fine-Grained ONNX Export for SOC (950MB TPU memory)
=============================================================
Exports the ViT backbone in 2-block chunks for SOC deployment.
The existing 5-part model needs 3.48GB/part — far exceeding SOC's 950MB.
With 2 blocks per part: ~16 parts for blocks + 1 part for embedding = 17 parts.

Usage (inside tpu_mlir Docker):
  cd /workspace/git_commits/developer/sophon-demo/sample/SAM3/tools
  python3 export_vit_fine.py --output_dir ../models/onnx_soc

Author: liheng.fang
Date: 2025-06-23
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np

torch.set_autocast_enabled(False)


def apply_patches():
    """Apply patches for ONNX export compatibility."""
    import torch

    # Force CPU before touching any CUDA
    torch.set_default_device('cpu')
    torch.cuda.is_available = lambda: False

    # Add SAM3 to path (the repo is at /workspace/git_commits/developer/sam3,
    # and inside it the sam3/ package directory)
    sys.path.insert(0, '/workspace/git_commits/developer/sam3')

    # Patch pin_memory first — prevents CUDA initialization issues
    _pin_orig = torch.Tensor.pin_memory
    def cpu_safe_pin(self, device=None):
        if not self.is_cuda:
            return self
        return _pin_orig(self, device)
    torch.Tensor.pin_memory = cpu_safe_pin

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
    print("[Patch] Mlp.forward → standard ops")

    # Position embedding patch
    from sam3.model import position_encoding
    _orig = position_encoding.PositionEmbeddingSine.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["precompute_resolution"] = None
        return _orig(self, *args, **kwargs)

    position_encoding.PositionEmbeddingSine.__init__ = patched_init
    print("[Patch] PositionEmbeddingSine.__init__ → no CUDA precompute")


def load_model(checkpoint_path):
    """Load SAM3 model with weights."""
    from sam3.model_builder import build_sam3_image_model

    print(f"\n[Model] Loading from {checkpoint_path}...")
    model = build_sam3_image_model(
        bpe_path=None,  # BPE is embedded in checkpoint for VETextEncoder
        device='cpu',
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=False,
    )
    model = model.float().eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Loaded: {total_params:,} params")
    return model


def export_block_sequence(blocks, block_indices, output_path, input_shape=(1, 5184, 1024)):
    """
    Export a sequence of consecutive ViT blocks as a single ONNX model.

    Args:
        blocks: nn.ModuleList of ViT blocks
        block_indices: list of block indices in this sequence (e.g., [0,1])
        output_path: output ONNX file path
        input_shape: input tensor shape (B, seq_len, hidden_dim)
    """
    class BlockSequence(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.blocks = blocks

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

    seq = BlockSequence(blocks)
    seq.eval()

    dummy_input = torch.randn(*input_shape).float()

    # Verify forward pass
    with torch.no_grad():
        out = seq(dummy_input)
        print(f"  Forward: {list(dummy_input.shape)} → {list(out.shape)}")

    # Export
    torch.onnx.export(
        seq,
        dummy_input,
        output_path,
        opset_version=14,
        input_names=['x'],
        output_names=['out'],
        dynamic_axes={'x': {0: 'batch'}, 'out': {0: 'batch'}},
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ONNX: {output_path} ({size_mb:.1f} MB)")

    # Validate
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  Validation: PASSED")


def export_patch_embed(model, output_path, input_shape=(1, 3, 1008, 1008)):
    """
    Export the patch embedding + position embedding + pre-norm layer.
    This produces the initial feature tokens from the image.

    Args:
        model: SAM3 image model
        output_path: output ONNX file path
        input_shape: input image shape
    """
    vit_trunk = model.backbone.vision_backbone.trunk
    vit_trunk.eval()

    from sam3.model.data_misc import NestedTensor

    class PatchEmbedWrapper(nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.trunk = trunk

        def forward(self, x):
            B = x.shape[0]
            mask = torch.zeros(B, x.shape[2], x.shape[3], dtype=torch.bool, device=x.device)
            nt = NestedTensor(x, mask)
            # Run patch_embed + pos_embed + pre-norm only
            x = self.trunk.patch_embed(nt.tensors)
            if self.trunk.pos_embed is not None:
                x = x + self.trunk.pos_embed(nt)
            if self.trunk.ln_pre is not None:
                x = self.trunk.ln_pre(x)
            return x

    wrapped = PatchEmbedWrapper(vit_trunk)
    wrapped.eval()

    dummy_input = torch.randn(*input_shape).float()

    with torch.no_grad():
        out = wrapped(dummy_input)
        print(f"  Forward: {list(dummy_input.shape)} → {list(out.shape)}")

    torch.onnx.export(
        wrapped,
        dummy_input,
        output_path,
        opset_version=14,
        input_names=['image'],
        output_names=['features'],
        dynamic_axes={'image': {0: 'batch'}},
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ONNX: {output_path} ({size_mb:.1f} MB)")

    import onnx
    onnx.checker.check_model(onnx.load(output_path))
    print(f"  Validation: PASSED")


def main():
    parser = argparse.ArgumentParser(description='SAM3 ViT Fine-Grained ONNX Export for SOC')
    parser.add_argument('--checkpoint', type=str,
                        default='/workspace/git_commits/developer/sophon-demo/sample/SAM3/models/sam3.pt',
                        help='Path to SAM3 checkpoint (sam3.pt)')
    parser.add_argument('--output_dir', type=str, default='../models/onnx_soc',
                        help='Output directory for ONNX files')
    parser.add_argument('--blocks_per_part', type=int, default=2,
                        help='Number of transformer blocks per ONNX part')
    parser.add_argument('--start_export', action='store_true',
                        help='Actually run the export')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SAM3 ViT Fine-Grained ONNX Export")
    print(f"  Blocks per part: {args.blocks_per_part}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)

    # Apply patches
    apply_patches()

    if not args.start_export:
        print("\n[DRY RUN] Use --start_export to actually run the export.")
        print(f"\nPlanned export with {args.blocks_per_part} blocks/part:")
        print(f"  Part 0: patch_embed + pos_embed + ln_pre (no blocks)")
        total_blocks = 32
        num_block_parts = (total_blocks + args.blocks_per_part - 1) // args.blocks_per_part
        for i in range(num_block_parts):
            start = i * args.blocks_per_part
            end = min(start + args.blocks_per_part, total_blocks) - 1
            if start == end:
                print(f"  Part {i+1}: block {start}")
            else:
                print(f"  Part {i+1}: blocks {start}-{end}")
        print(f"\n  Total: 1 (embedding) + {num_block_parts} (blocks) = {num_block_parts + 1} parts")
        print(f"\n  Estimated TPU memory per 2-block part: ~870MB (fits 950MB)")
        return

    # Load model
    model = load_model(args.checkpoint)
    vit_trunk = model.backbone.vision_backbone.trunk

    # Export patch embedding (part 0)
    print("\n" + "=" * 60)
    print("Part 0: Patch Embedding")
    print("=" * 60)
    part0_path = os.path.join(args.output_dir, 'sam3_vit_part0.onnx')
    export_patch_embed(model, part0_path)

    # Export block sequences
    total_blocks = len(vit_trunk.blocks)
    blocks_per_part = args.blocks_per_part
    num_block_parts = (total_blocks + blocks_per_part - 1) // blocks_per_part

    print(f"\nViT has {total_blocks} blocks, exporting in {num_block_parts} parts "
          f"({blocks_per_part} blocks/part)")

    for part_idx in range(num_block_parts):
        start = part_idx * blocks_per_part
        end = min(start + blocks_per_part, total_blocks)
        block_indices = list(range(start, end))

        print(f"\n{'=' * 60}")
        blk_desc = f"blocks {block_indices[0]}-{block_indices[-1]}" if len(block_indices) > 1 else f"block {block_indices[0]}"
        print(f"Part {part_idx + 1}: {blk_desc}")
        print("=" * 60)

        part_path = os.path.join(args.output_dir, f'sam3_vit_part{part_idx + 1}.onnx')

        # Select blocks
        selected_blocks = nn.ModuleList([
            vit_trunk.blocks[i] for i in block_indices
        ])

        export_block_sequence(selected_blocks, block_indices, part_path)

    # Summary
    print("\n" + "=" * 60)
    print("Export Complete!")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Total parts: {num_block_parts + 1}")
    print(f"  Part 0: patch embedding")
    for i in range(num_block_parts):
        start = i * blocks_per_part
        end = min(start + blocks_per_part, total_blocks) - 1
        print(f"  Part {i+1}: blocks {start}-{end}")
    print("=" * 60)


if __name__ == '__main__':
    main()
