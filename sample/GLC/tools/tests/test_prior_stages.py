#!/usr/bin/env python3
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""逐段比对 prior 的 scales/means：官方 forward_four_part_prior vs 我们的实现。

不能手工复刻官方循环——它的 y_hat_so_far 累加用的是 process_with_mask 返回的
y_hat_i（已经乘过 mask），手写容易差一步，会在段 2/3 制造假失败。
这里直接跑官方函数，用 forward hook 抓 y_hat_so_far 的真实值，
再喂给我们的 _stage_params 做逐元素对账。

运行：PYTHONPATH=tools/vendor:python:. python3 tests/test_prior_stages.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # tools/tests -> GLC
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "python"))

import glc_codec as gc  # noqa: E402

H, W = 512, 768
_failures = []


def check(name, ok, detail=""):
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         (" - " + detail) if detail else ""))
    if not ok:
        _failures.append(name)


def build_net():
    proj = PROJ
    sys.path.insert(0, os.path.join(proj, "GLC_src"))
    sys.path.insert(0, os.path.join(proj, "tools", "stubs"))
    from src.models.image_model import GLC_Image

    torch.manual_seed(1234)
    net = GLC_Image(inplace=False)
    net.eval()
    return net

def main():
    net = build_net()
    torch.manual_seed(7)
    x = torch.randn(1, 3, H, W).clamp(-1, 1)

    captured = []
    def hook(module, inputs, output):
        captured.append(output.detach().clone())

    # 直接跑官方函数，用 hook 抓 y_spatial_prior 每次被调用时的真实输出
    handle = net.y_spatial_prior.register_forward_hook(hook)
    with torch.no_grad():
        y = net.enc(net.vqgan.encoder(x), net.q_enc[0:1])
        z = net.hyper_enc(y)
        z_hat = net.z_vq.get_quan_feat(
            net.z_vq.get_indices(z), (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))
        params = net.y_prior_fusion(net.hyper_dec(z_hat))
        net.forward_four_part_prior(
            y, params, net.y_spatial_prior_adaptor_1, net.y_spatial_prior_adaptor_2,
            net.y_spatial_prior_adaptor_3, net.y_spatial_prior,
            y_spatial_prior_reduction=net.y_spatial_prior_reduction)
    handle.remove()

    check("官方 forward_four_part_prior 调用了 3 次 y_spatial_prior",
          len(captured) == 3, "%d 次" % len(captured))

    # 我们的实现：第 0 段来自 _prior_params，第 1..3 段来自 _stage_params
    with torch.no_grad():
        q_enc = gc._separate_prior(params)[0]
        _, scales0, means0, common = gc._prior_params(net, z_hat)
        masks = net.get_mask_four_parts(*y.shape, y.dtype, y.device)
        hat = torch.zeros_like(y)
        y_scaled = y * q_enc
        for stage in range(4):
            s, m = gc._stage_params(net, stage, hat, common, scales0, means0)
            if stage > 0:
                rs, rm = captured[stage - 1].chunk(2, 1)
                check("第 %d 段 scales 与官方一致" % stage, torch.equal(s, rs),
                      "max diff %.3e" % (s - rs).abs().max().item())
                check("第 %d 段 means 与官方一致" % stage, torch.equal(m, rm),
                      "max diff %.3e" % (m - rm).abs().max().item())
            # 与官方 process_with_mask 同款推进
            mask = masks[stage]
            m_hat = m * mask
            y_q = net.quant((y_scaled - m_hat) * mask)
            hat = hat + (y_q + m_hat)

    print()
    if _failures:
        print("失败 %d 项: %s" % (len(_failures), _failures))
        return 1
    print("全部通过（prior 四段逐段与官方一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())