#!/usr/bin/env python3
"""
Export all SAM3 Grounding ONNX models (encoder, decoder) for bmodel compilation.

Usage:
  python export_grounding_all.py [--grid 36] [--output_dir ../models/onnx_grounding_504]
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_grounding_onnx import (
    patch_sam3_model, load_model,
    export_encoder, export_decoder,
)


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM3 Grounding ONNX models")
    parser.add_argument("--checkpoint", type=str,
                        default="../models/sam3.pt")
    parser.add_argument("--output_dir", type=str,
                        default="../models/onnx_grounding_504")
    parser.add_argument("--grid", type=int, default=36,
                        help="Feature grid (36 for 504x504)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output_dir = os.path.abspath(args.output_dir)

    print("=" * 60)
    print("SAM3 Grounding ONNX Export (All Models)")
    print(f"  Grid:     {args.grid}x{args.grid}")
    print(f"  Output:   {output_dir}")
    print("=" * 60)

    # Patch and load
    patch_sam3_model()
    model = load_model(args.checkpoint)

    # Export encoder
    export_encoder(model, output_dir, args.grid)

    # Export decoder
    export_decoder(model, output_dir, args.grid)

    print("")
    print("=" * 60)
    print("Export Complete!")
    print(f"  Encoder: {output_dir}/sam3_grounding_encoder.onnx")
    print(f"  Decoder: {output_dir}/sam3_grounding_decoder.onnx")
    print("")
    print("Next steps:")
    print("  1. Copy ONNX files to onnx_504/ (or use onnx_grounding_504/)")
    print("  2. Inside tpu-mlir docker, run:")
    print(f"     ./gen_bmodel.sh --res 504 --chip bm1684x --mode f16")
    print("  3. Test with:")
    print("     python sam3_infer.py --image cat.jpg --prompt 'a cat'")
    print("=" * 60)


if __name__ == "__main__":
    main()
