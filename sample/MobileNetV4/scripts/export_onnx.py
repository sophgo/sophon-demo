#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""
Export MobileNetV4 from timm to ONNX format.
"""
import torch
import timm
import onnx
import onnxruntime
import numpy as np
import argparse
import os


def export_mobilenetv4_onnx(model_name, output_path, opset_version=13):
    """Export MobileNetV4 model to ONNX.

    Args:
        model_name: timm model name
        output_path: output ONNX file path
        opset_version: ONNX opset version
    """
    print(f"Loading model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    # Get model config
    input_size = model.default_cfg['input_size']  # (3, 224, 224)
    print(f"Input size: {input_size}")

    # Create dummy input with dynamic batch
    batch_size = 1
    dummy_input = torch.randn(batch_size, *input_size)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print("ONNX export done, merging external data...")

    # Merge external data into single file (TPU-MLIR requires single-file ONNX)
    onnx_model = onnx.load(output_path, load_external_data=True)
    tmp_path = output_path + ".tmp"
    onnx.save(onnx_model, tmp_path, save_as_external_data=False)
    os.rename(tmp_path, output_path)
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    print(f"Single-file ONNX size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    onnx.checker.check_model(onnx_model)
    print("ONNX checker: PASSED")

    # Compare ONNX Runtime vs PyTorch outputs
    ort_session = onnxruntime.InferenceSession(
        output_path, providers=['CPUExecutionProvider']
    )
    ort_inputs = {"input": dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)[0]

    with torch.no_grad():
        pt_outputs = model(dummy_input).numpy()

    max_diff = np.abs(ort_outputs - pt_outputs).max()
    mean_diff = np.abs(ort_outputs - pt_outputs).mean()
    print(f"PyTorch vs ONNX Runtime - max_diff: {max_diff:.6e}, mean_diff: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("ONNX export verified SUCCESS!")
    else:
        print("WARNING: output difference detected, but may be acceptable")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export MobileNetV4 to ONNX")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mobilenetv4_conv_medium.e500_r224_in1k",
        help="timm model name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../models/onnx/mobilenetv4_conv_medium.onnx",
        help="output ONNX file path"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version"
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    export_mobilenetv4_onnx(args.model_name, args.output, args.opset)


if __name__ == "__main__":
    main()
