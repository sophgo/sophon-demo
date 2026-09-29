#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

"""生成解码器 INT8 校准数据：用编码器 + 码本查找得到 y / z_hat"""
import os, sys, glob
import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, "GLC_src")
from src.models.image_model import GLC_Image
from src.utils.test_utils import get_state_dict

H, W = 512, 768
OUT = "cali_dec"
os.makedirs(OUT, exist_ok=True)

net = GLC_Image(inplace=False)
net.load_state_dict(get_state_dict("GLC_image.pth.tar"), strict=False)
net.eval()
cb = net.z_vq.embedding.weight.detach()

se = ort.InferenceSession("onnx/glc_encoder.onnx", providers=["CPUExecutionProvider"])
files = sorted(glob.glob("cali/*.npz"))
print("校准样本数:", len(files))

for i, f in enumerate(files):
    d = np.load(f)
    x, q_enc = d["x"], d["q_enc"]
    y, z = se.run(None, {"x": x, "q_enc": q_enc})

    # 码本查找（与推理时一致）
    zz = torch.from_numpy(z).permute(0, 2, 3, 1).contiguous().view(-1, 256)
    dist = (zz ** 2).sum(1, keepdim=True) + (cb ** 2).sum(1) - 2 * zz @ cb.t()
    idx = torch.argmin(dist, 1).unsqueeze(1)
    z_hat = net.z_vq.get_quan_feat(idx, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))
    z_hat = z_hat.detach().numpy()

    np.savez(os.path.join(OUT, "cali_dec_%02d.npz" % i),
             y=y.astype(np.float32),
             z_hat=z_hat.astype(np.float32),
             q_dec=net.q_dec[i % 4:i % 4 + 1].detach().numpy().astype(np.float32))
    print("  %s  y=%s z_hat=%s" % (os.path.basename(f), y.shape, z_hat.shape))

print("解码器校准数据已生成于", OUT)
