#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""生成 TPU-MLIR INT8 校准数据（npz，含模型全部输入）"""
import os, sys, glob
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "GLC_src"))
from src.models.image_model import GLC_Image
from src.utils.test_utils import get_state_dict

H, W = 512, 768
N_CALI = 10
OUT = "cali"
os.makedirs(OUT, exist_ok=True)

net = GLC_Image(inplace=False)
net.load_state_dict(get_state_dict(os.path.join(PROJ, "models", "GLC_image.pth.tar")), strict=True)
net.eval()

# q_enc / q_dec 取 4 个质量档位（q_index 0..3）
q_enc_all = net.q_enc.detach().numpy()
q_dec_all = net.q_dec.detach().numpy()

files = sorted(glob.glob("images/*.png"))[:N_CALI]
print("校准图像数:", len(files))

for i, f in enumerate(files):
    im = Image.open(f).convert("RGB")
    if im.size != (W, H):
        im = im.resize((W, H), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0          # [0,1]
    arr = arr * 2.0 - 1.0                                    # [-1,1]
    x = arr.transpose(2, 0, 1)[None]                         # 1,3,H,W
    qi = i % 4
    np.savez(os.path.join(OUT, "cali_%02d.npz" % i),
             x=x.astype(np.float32),
             q_enc=q_enc_all[qi:qi + 1].astype(np.float32))
    print("  ", os.path.basename(f), x.shape, "q_index=", qi)

print("校准数据已生成于", OUT)
