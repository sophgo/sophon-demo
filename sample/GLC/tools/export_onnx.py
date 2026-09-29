#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import sys, os, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "GLC_src"))
from src.models.image_model import GLC_Image
from src.utils.test_utils import get_state_dict

torch.set_num_threads(8)
H = int(os.environ.get("IMG_H", 512))
W = int(os.environ.get("IMG_W", 768))
OUT = os.environ.get("ONNX_OUT", "onnx")
os.makedirs(OUT, exist_ok=True)
print("=== 导出尺寸 %dx%d ===" % (H, W))

net = GLC_Image(inplace=False)
net.load_state_dict(get_state_dict(os.path.join(PROJ, "models", "GLC_image.pth.tar")), strict=True)
net.eval()


class ScaledGroupNorm(nn.Module):
    """把 GroupNorm 的输入先缩小 SCALE 倍，同时把 eps 缩小 SCALE^2 倍。

    AttnBlock / ResBlock 的输入激活幅值可达数千，fp16 下 GroupNorm 内部要算 E[x^2]，
    在 blocks[16]/[18] 处达到 1.6e5 / 3.1e5，超过 fp16 上限 65504 而溢出成 Inf -> NaN。

    推导（GN 对输入整体缩放不变，但 eps 项会破坏不变性，故一并补偿）：
        GN'(x/s) = (x/s - m/s) / sqrt(v/s^2 + eps/s^2)
                 = (x - m) / sqrt(v + eps)
                 = GN(x)
    其中 eps' = eps/s^2。这样结果与原始 GroupNorm 逐位等价，而归约用的方差被缩小
    SCALE^2 倍（本例 1024 倍），安全落入 fp16 范围。

    仅 FP16 需要，由 F16_NORM_FIX=1 开启；BF16 指数范围与 FP32 相同，无需处理。
    """

    SCALE = 32.0

    def __init__(self, norm):
        super().__init__()
        gn = nn.GroupNorm(norm.num_groups, norm.num_channels,
                          eps=norm.eps / (self.SCALE ** 2), affine=norm.affine)
        if norm.affine:
            gn.weight = norm.weight
            gn.bias = norm.bias
        self.norm = gn

    def forward(self, x):
        return self.norm(x / self.SCALE)


if os.environ.get("F16_NORM_FIX") == "1":
    n_fix = 0
    for m in net.modules():
        if type(m) is nn.GroupNorm:
            n_fix += 1
    for name, parent in list(net.named_modules()):
        for cname, child in list(parent.named_children()):
            if type(child) is nn.GroupNorm:
                setattr(parent, cname, ScaledGroupNorm(child))
    print("=== 已对 %d 个 GroupNorm 启用输入预缩放 + eps 补偿（FP16 防溢出）===" % n_fix)


class EncExport(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.vq_enc = net.vqgan.encoder
        self.enc = net.enc
        self.hyper_enc = net.hyper_enc

    def forward(self, x, q_enc):
        y = self.enc(self.vq_enc(x), q_enc)
        z = self.hyper_enc(y)
        return y, z


class DecExport(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, y, z_hat, q_dec):
        n = self.net
        params = n.y_prior_fusion(n.hyper_dec(z_hat))
        y_res, y_q, y_hat, scales_hat = n.forward_four_part_prior(
            y, params,
            n.y_spatial_prior_adaptor_1, n.y_spatial_prior_adaptor_2,
            n.y_spatial_prior_adaptor_3, n.y_spatial_prior,
            y_spatial_prior_reduction=n.y_spatial_prior_reduction)
        x_hat = n.vqgan.generator(n.dec(y_hat, q_dec))
        return x_hat, y_q, scales_hat


enc = EncExport(net).eval()
dec = DecExport(net).eval()
q_enc = net.q_enc[0:1].detach().clone()
q_dec = net.q_dec[0:1].detach().clone()

x = torch.randn(1, 3, H, W)
y = torch.randn(1, 256, H // 16, W // 16)
z_h = torch.randn(1, 256, H // 64, W // 64)
print("y:", tuple(y.shape), " z:", tuple(z_h.shape))

torch.onnx.export(enc, (x, q_enc), OUT + "/glc_encoder.onnx",
                  input_names=["x", "q_enc"], output_names=["y", "z"],
                  opset_version=13, do_constant_folding=True, dynamo=False)
torch.onnx.export(dec, (y, z_h, q_dec), OUT + "/glc_decoder.onnx",
                  input_names=["y", "z_hat", "q_dec"],
                  output_names=["x_hat", "y_q", "scales_hat"],
                  opset_version=13, do_constant_folding=True, dynamo=False)

for f in ["glc_encoder.onnx", "glc_decoder.onnx"]:
    print(f, round(os.path.getsize(OUT + "/" + f) / 1e6, 1), "MB")
