#!/usr/bin/env python3
"""
Export FEARTracker to ONNX format for TPU-MLIR compilation.

Model: FEARNet (FBNet-C backbone + Siamese cross-correlation tracker)
Inputs:  template [1, 3, 128, 128], search [1, 3, 256, 256]
Outputs: bbox_pred [1, 4, 16, 16], cls_pred [1, 1, 16, 16]

Usage:
    PYTHONPATH=<feartracker_source> python3 tools/export_onnx.py \
        --checkpoint <path_to_ckpt> --output <output_onnx_path>
"""

import argparse
import os
import sys
import numpy as np
import torch


def export_model(checkpoint_path, output_path, opset_version=16):
    # Add source project to path for model imports
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from model_training.model.fear_net import FEARNet
    from model_training.utils.torch import load_from_lighting

    # Load config and model
    config = {
        "backbone": "custom_fbnet",
        "img_size": 256,
        "pretrained": False,
        "score_size": 16,
        "adjust_channels": 256,
        "total_stride": 16,
        "instance_size": 256,
        "towernum": 2,
        "max_layer": 4,
        "crop_template_features": False,
        "conv_block": "sep_conv",
        "mobile": True,
    }
    model = FEARNet(**config)
    model = load_from_lighting(model, checkpoint_path, "cpu")
    model.eval()

    template = torch.randn(1, 3, 128, 128)
    search = torch.randn(1, 3, 256, 256)

    print(f"template: {template.shape}")
    print(f"search:   {search.shape}")

    # Verify with PyTorch
    with torch.no_grad():
        result = model(template, search)
    print(f"bbox_pred: {result['TARGET_REGRESSION_LABEL_KEY'].shape}")
    print(f"cls_pred:  {result['TARGET_CLASSIFICATION_KEY'].shape}")

    torch.onnx.export(
        model,
        (template, search),
        output_path,
        input_names=["template", "search"],
        output_names=["bbox_pred", "cls_pred"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"\nExported ONNX to: {output_path}")

    # Verify ONNX
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX check: PASSED")

    ops = sorted(set(n.op_type for n in onnx_model.graph.node))
    print(f"ONNX ops ({len(ops)}): {ops}")

    # Verify with ONNX Runtime
    try:
        import onnxruntime
        sess = onnxruntime.InferenceSession(output_path)
        ort_inputs = {"template": template.numpy(), "search": search.numpy()}
        ort_outs = sess.run(None, ort_inputs)
        print(f"ORT bbox_pred: {ort_outs[0].shape}")
        print(f"ORT cls_pred:  {ort_outs[1].shape}")
        print(f"Max diff bbox: {np.abs(result['TARGET_REGRESSION_LABEL_KEY'].numpy() - ort_outs[0]).max():.6e}")
        print(f"Max diff cls:  {np.abs(result['TARGET_CLASSIFICATION_KEY'].numpy() - ort_outs[1]).max():.6e}")
    except ImportError:
        print("onnxruntime not installed, skipping verification")


def main():
    parser = argparse.ArgumentParser(description="Export FEARTracker to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to FEARTracker checkpoint (.ckpt)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__), "feartracker.onnx"),
                        help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=16)
    args = parser.parse_args()
    export_model(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()