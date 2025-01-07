# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import skimage
import skimage.io
import cv2
from PIL import Image
from torchvision import transforms
import time
from core.lightstereo import LightStereo

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='LightStereo')
parser.add_argument('--MAX_DISP', type=int, default=192, help='maximum disparity')
parser.add_argument('--loadckpt', default='../models/ckpt/LightStereo-S-SceneFlow.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--LEFT_ATT',default=True)
parser.add_argument('--AGGREGATION_BLOCKS',default=[1,2,4],help="M:[4, 8, 16] S:[1,2,4],L:[8, 16, 32],LX:[8, 16, 32]")
parser.add_argument('--EXPANSE_RATIO',default=4,help="L,LX:8")
parser.add_argument('--onnx_path',default="LightStereo-S-SceneFlow.onnx",help="output onnx path")
# parse arguments
args = parser.parse_args()
model = LightStereo(args.MAX_DISP, args.LEFT_ATT,args.AGGREGATION_BLOCKS, args.EXPANSE_RATIO)
#model = nn.DataParallel(model)
model.cuda()

state_dict = torch.load(args.loadckpt)
#print(state_dict.keys())
model.load_state_dict(state_dict["model_state"],strict=False)
model = model.eval()

input_tensor = torch.randn(4, 3, 384, 1248)  # Example input shape (batch_size, channels, height, width)

# Scale the tensor to be in the range [0, 255]
scaled_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min()) * 255

# Convert the tensor to float32
scaled_tensor = scaled_tensor.float()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# input_tensor = input_tensor.to(device)
model = model.to("cpu")

#print(summary(model))

scaled_tensor = scaled_tensor.to("cpu")
#model = torch.jit.script(model)
torch.onnx.export(model,                     # PyTorch model
                  (scaled_tensor,scaled_tensor),             # Example input tensor
                  args.onnx_path,         # Output ONNX file path
                  input_names=["left","right"],    # Input names used in the ONNX model
                  output_names=['output'],  # Output names used in the ONNX model
                  export_params=True,
                  do_constant_folding=False,
                  opset_version=13,
                  dynamic_axes={
                    "left":{0: 'batch_size'},
                    "right":{0: 'batch_size'},
                    "output":{0: 'batch_size'}
                  }
                  )
                  #verbose=True)
