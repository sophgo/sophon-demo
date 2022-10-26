#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import torch
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
import argparse
import cv2
from yolact import Yolact
from config import set_cfg


if __name__=='__main__':
    SUPPORT_CONFIG = {"yolact_resnet50": "yolact_resnet50_config",
                      "yolact_resnet50_pascal": "yolact_resnet50_pascal_config",
                      "yolact_darknet53": "yolact_darknet53_config",
                      "yolact_base": "yolact_base_config",
                      "yolact_im700": "yolact_im700_config",
                      "yolact_im400": "yolact_im400_config",
                      }

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help='Assign the pytorch model path')
    parser.add_argument("--mode", type=str, help='Assign the mode: ["onnx", "tstrace"]')
    parser.add_argument("--cfg", type=str, help='Assign the model config: {}'.format(SUPPORT_CONFIG))
    parser.add_argument("--batch_size", type=int, default=1, help='Assign the batch size')
    parser.add_argument("--output", type=str, default=None, help='Assign the output directory')
    args = parser.parse_args()

    trained_model = args.input
    if not os.path.exists(trained_model):
        raise FileNotFoundError('{} is not existed.'.format(trained_model))

    if not isinstance(args.mode, str) or args.mode.lower() not in ["onnx", "tstrace"]:
        raise ValueError('mode must be ["onnx", "tstrace"], but got {}'.format(args.mode))
    mode = args.mode.lower()

    if args.output is None:
        output_dir = __dir__
    else:
        output_dir = os.path.abspath(args.output)
    output_basename = os.path.basename(trained_model)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('makedir: {}'.format(output_dir))

    if not isinstance(args.cfg, str) or args.cfg.lower() not in list(SUPPORT_CONFIG.keys()):
        raise ValueError('mode must be {}, but got {}'.format(list(SUPPORT_CONFIG.keys()), args.cfg))
    args.cfg = args.cfg.lower()
    cfg = SUPPORT_CONFIG[args.cfg]

    hw = (550, 550)
    if args.cfg == 'yolact_im400':
        hw = (400, 400)
    elif args.cfg == 'yolact_im700':
        hw = (700, 700)
    elif args.cfg in list(SUPPORT_CONFIG.keys()):
        hw = (550, 550)
    else:
        raise NotImplementedError('please define input shape.')

    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0, but got {}".format(args.batch_size))
    input_shape = (args.batch_size, 3, hw[0], hw[1])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_cfg(cfg)

    net = Yolact()
    net.load_weights(trained_model)
    net.eval()
    net.to(device)

    inputs = torch.randn(*input_shape).to(device)
    # warmup
    net(inputs)

    if mode == 'onnx':
        output_path = os.path.join(output_dir, os.path.splitext(output_basename)[0] + '.onnx')

        dynamic_axes = {'image': [0],
                        'loc': [0],
                        'conf': [0],
                        'mask': [0],
                        'proto': [0],
                        }
        # onnx
        torch.onnx.export(net, inputs, output_path, verbose=False, opset_version=12, input_names=['image'],
                          output_names=['loc', 'conf', 'mask', 'proto'],
                          dynamic_axes=dynamic_axes,
                          )
        # dynamic batch size
        import onnx
        onnx_model = onnx.load(output_path)
        onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
        onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
        onnx_model.graph.output[1].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
        onnx_model.graph.output[2].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
        onnx_model.graph.output[3].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
        onnx.save(onnx_model, output_path)

    elif mode == 'tstrace':
        # torch.jit.trace
        output_path = os.path.join(output_dir, os.path.splitext(output_basename)[0] + '.trace.pt')
        traced_model = torch.jit.trace(net, inputs)
        traced_model.save(output_path)
    else:
        raise NotImplementedError

    print('{} is saved.'.format(output_path))
