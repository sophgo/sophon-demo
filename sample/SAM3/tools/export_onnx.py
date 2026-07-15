#!/usr/bin/env python3
"""
SAM3 ViT Trunk ONNX Export
Exports the full SAM3 ViT trunk to ONNX for TPU-MLIR compilation.

The source model is trained at 1008x1008 (72x72 patches); this script
recomputes RoPE freqs_cis for the target resolution so a 504x504
(36x36 patches) trunk can be exported. Resolution is a parameter:
  python3 export_onnx.py --start_export --resolution 504
  python3 export_onnx.py --start_export --resolution 1008

Usage (inside tpu_mlir Docker):
  python3 export_onnx.py --start_export
"""

import os, sys, argparse, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import numpy as np
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

from sam3.model.vitdet import Mlp, get_abs_pos
def _patched_mlp(self, x):
    x = self.fc1(x); x = self.act(x); x = self.drop1(x)
    x = self.norm(x); x = self.fc2(x); x = self.drop2(x)
    return x
Mlp.forward = _patched_mlp
print("[Patch] Mlp.forward → standard ops")

from sam3.model import position_encoding
_pe_orig = position_encoding.PositionEmbeddingSine.__init__
def _patched_pe(self, *args, **kwargs):
    kwargs["precompute_resolution"] = None
    return _pe_orig(self, *args, **kwargs)
position_encoding.PositionEmbeddingSine.__init__ = _patched_pe
print("[Patch] PositionEmbeddingSine")

from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import NestedTensor
from sam3.model.vitdet import Attention


def patch_freqs_cis(model, new_grid_size=36):
    """Recompute RoPE freqs_cis for global attention blocks only.
    Window attention: input_size=window_size (24) — unchanged.
    Global attention (blocks 7,15,23,31): input_size=grid_size — update to new resolution."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Attention) and module.input_size is not None:
            # Only patch global attention blocks (window_size=0 in parent block)
            if module.input_size[0] > 30:  # was 72 (1008) or similar full-grid size
                module.input_size = (new_grid_size, new_grid_size)
                module.rope_pt_size = (new_grid_size, new_grid_size)
                if module.rel_pos_h is not None:
                    module.rel_pos_h = nn.Parameter(
                        torch.zeros(2 * new_grid_size - 1, module.head_dim))
                if module.rel_pos_w is not None:
                    module.rel_pos_w = nn.Parameter(
                        torch.zeros(2 * new_grid_size - 1, module.head_dim))
                module._setup_rope_freqs()
                count += 1
    print(f"[Patch] Global attn RoPE: {count} modules: 72→{new_grid_size}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='/workspace/git_commits/developer/sophon-demo/sample/SAM3/models/sam3.pt')
    p.add_argument('--output_dir', default='../models/onnx_504')
    p.add_argument('--resolution', type=int, default=504)
    p.add_argument('--start_export', action='store_true')
    args = p.parse_args()

    H = W = args.resolution
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"SAM3 ViT Full Trunk ONNX Export ({H}x{H})")
    print("=" * 60)

    if not args.start_export:
        print(f"\n[DRY RUN] Use --start_export to run.")
        print(f"  Export: full ViT trunk [1,3,{H},{H}] → multi-scale features")
        return

    print("\n[Model] Loading...")
    model = build_sam3_image_model(
        bpe_path=None, device='cpu', load_from_HF=False,
        checkpoint_path=args.checkpoint,
        enable_segmentation=False, enable_inst_interactivity=False,
    )
    model = model.float().eval()
    vit_trunk = model.backbone.vision_backbone.trunk
    vit_trunk.eval()
    print(f"[Model] {sum(p.numel() for p in model.parameters()):,} params")

    # Recompute RoPE frequencies for the target resolution (H//14 patches)
    patch_freqs_cis(vit_trunk, new_grid_size=H // 14)

    # Export full ViT trunk, bypassing NestedTensor dynamic branches
    _get_abs_pos = get_abs_pos  # capture for ONNX trace

    class ViTWrapper(nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.trunk = trunk

        def forward(self, x):
            # Run ViT internals directly (no NestedTensor, no mask.any())
            x = self.trunk.patch_embed(x)
            h, w = x.shape[1], x.shape[2]
            if self.trunk.pos_embed is not None:
                pe = self.trunk.pos_embed
                pretrain_cls = self.trunk.pretrain_use_cls_token
                retain_cls = self.trunk.retain_cls_token
                tiling = self.trunk.tile_abs_pos
                x = x + _get_abs_pos(pe, pretrain_cls, (h, w), retain_cls, tiling=tiling)
            x = self.trunk.ln_pre(x)
            for i, blk in enumerate(self.trunk.blocks):
                x = blk(x)
                if i == self.trunk.full_attn_ids[-1]:
                    x = self.trunk.ln_post(x)
            # Reshape to expected output format [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
            return x

    wrapped = ViTWrapper(vit_trunk).eval()
    dummy = torch.randn(1, 3, H, W).float()

    print(f"  Input: {list(dummy.shape)}")
    with torch.no_grad():
        out = wrapped(dummy)
        if isinstance(out, (list, tuple)):
            for i, o in enumerate(out):
                if isinstance(o, NestedTensor):
                    print(f"  Output[{i}]: NestedTensor({list(o.tensors.shape)})")
                elif isinstance(o, torch.Tensor):
                    print(f"  Output[{i}]: {list(o.shape)}")
        else:
            print(f"  Output: {type(out)}")
    print(f"  Forward: OK")

    out_path = os.path.join(args.output_dir, 'sam3_vit_trunk.onnx')
    print(f"\n[Export] {out_path}...")
    torch.onnx.export(
        wrapped, dummy, out_path,
        opset_version=14,
        input_names=['image'],
        output_names=['features'],
        dynamic_axes={'image': {0: 'batch'}},
        do_constant_folding=True,
    )
    mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ONNX: {out_path} ({mb:.1f} MB)")

    import onnx
    onnx.checker.check_model(onnx.load(out_path))
    print(f"  Validation: PASSED")

    print("\n" + "=" * 60)
    print(f"Export complete! {out_path} ({mb:.1f} MB)")
    print("=" * 60)


if __name__ == '__main__':
    main()
