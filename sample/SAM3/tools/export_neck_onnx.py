#!/usr/bin/env python3
"""
SAM3 Neck (Sam3DualViTDetNeck FPN) ONNX Export
Exports the 4-branch FPN neck convs as a single ONNX for TPU-MLIR compilation.

Resolution is a parameter (grid = resolution // 14):
  python3 export_neck_onnx.py --start_export --resolution 504   # grid=36
  python3 export_neck_onnx.py --start_export --resolution 1008  # grid=72

Usage (inside tpu_mlir Docker):
  python3 export_neck_onnx.py --start_export
"""

import os, sys, argparse, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = lambda: None
_orig_arange = torch.arange
def _cpu_arange(*args, **kwargs):
    if 'device' in kwargs and 'cuda' in str(kwargs['device']):
        kwargs = dict(kwargs, device='cpu')
    return _orig_arange(*args, **kwargs)
torch.arange = _cpu_arange
torch.Tensor.pin_memory = lambda self, device=None: self

sys.path.insert(0, '/workspace/git_commits/developer/sam3')

from sam3.model.vitdet import Mlp
def _patched_mlp(self, x):
    x = self.fc1(x); x = self.act(x); x = self.drop1(x)
    x = self.norm(x); x = self.fc2(x); x = self.drop2(x)
    return x
Mlp.forward = _patched_mlp
print("[Patch] Mlp.forward -> standard ops")

from sam3.model import position_encoding
_pe_orig = position_encoding.PositionEmbeddingSine.__init__
def _patched_pe(self, *args, **kwargs):
    kwargs["precompute_resolution"] = None
    return _pe_orig(self, *args, **kwargs)
position_encoding.PositionEmbeddingSine.__init__ = _patched_pe
print("[Patch] PositionEmbeddingSine")

from sam3.model_builder import build_sam3_image_model


class NeckWrapper(nn.Module):
    """Export neck convs (4 FPN branches) as single ONNX."""
    def __init__(self, neck):
        super().__init__()
        self.convs = neck.convs  # ModuleList of 4 nn.Sequential

    def forward(self, x):
        return self.convs[0](x), self.convs[1](x), self.convs[2](x), self.convs[3](x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='/workspace/git_commits/developer/sophon-demo/sample/SAM3/models/sam3.pt')
    p.add_argument('--output_dir', default='../models/onnx_504')
    p.add_argument('--resolution', type=int, default=504)
    p.add_argument('--start_export', action='store_true')
    args = p.parse_args()

    grid = args.resolution // 14  # 36 for 504, 72 for 1008
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"SAM3 Neck ONNX Export (grid={grid}x{grid})")
    print("=" * 60)

    if not args.start_export:
        print(f"\n[DRY RUN] Use --start_export to run.")
        print(f"  Input:  [1, 1024, {grid}, {grid}]")
        print(f"  Output: 4x multi-scale features")
        return

    print("\n[Model] Loading...")
    model = build_sam3_image_model(
        bpe_path=None, device='cpu', load_from_HF=False,
        checkpoint_path=args.checkpoint,
        enable_segmentation=False, enable_inst_interactivity=False,
    )
    model = model.float().eval()
    neck = model.backbone.vision_backbone  # Sam3DualViTDetNeck
    print(f"[Model] Loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"[Neck] type={type(neck).__name__}")

    wrapped = NeckWrapper(neck).eval()
    dummy = torch.randn(1, 1024, grid, grid).float()

    print(f"\n[Test] Input: {list(dummy.shape)}")
    with torch.no_grad():
        feats = wrapped(dummy)
        for i, f in enumerate(feats):
            print(f"  feat[{i}]: {list(f.shape)}")

    out_path = os.path.join(args.output_dir, 'sam3_neck_combined.onnx')
    print(f"\n[Export] {out_path}...")
    torch.onnx.export(
        wrapped, dummy, out_path,
        opset_version=14,
        input_names=['vit_features'],
        output_names=['feat_s4', 'feat_s2', 'feat_s1', 'feat_s05'],
        dynamic_axes={'vit_features': {0: 'batch'}},
        do_constant_folding=True,
    )
    mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ONNX: {out_path} ({mb:.1f} MB)")

    import onnx
    onnx.checker.check_model(onnx.load(out_path))
    print(f"  Validation: PASSED")

    print("\n" + "=" * 60)
    print(f"Export complete! {out_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
