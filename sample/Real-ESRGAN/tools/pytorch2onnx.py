#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import argparse
import torch
import torch.onnx
# from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact

def main(args):
    # An instance of the model
    # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu') #x4 v3
    if args.params:
        keyname = 'params'
    else:
        keyname = 'params_ema'
    state_dict = torch.load(args.input, map_location=torch.device("cpu"))[keyname]
    model.load_state_dict(state_dict)
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()
    # An example input
    x = torch.rand(args.batch_size, 3, 480, 640)
    # Export the model
    with torch.no_grad():
        torch.onnx.export(model, x, args.output, opset_version=11, export_params=True)

if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default='experiments/pretrained_models/RealESRGAN_x4plus.pth', help='Input model path')
    parser.add_argument('--output', type=str, default='realesrgan-x4.onnx', help='Output onnx path')
    parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    args = parser.parse_args()

    main(args)
