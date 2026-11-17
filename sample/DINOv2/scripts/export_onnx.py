#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 DINOv2 (dinov2_vitb14_reg) 骨干 + 线性探针特征拼接为 ONNX。

输入口径（与官方仓库评估协议一致）
----------------------------------
官方 ImageNet-1k linear-probe 评估用的是 dinov2/data/transforms.py 的
``make_classification_eval_transform()``，其默认参数为

    Resize(256, BICUBIC)  ->  CenterCrop(224)  ->  ToTensor  ->  Normalize(mean,std)

即**模型输入 224x224**（官方 MODEL_CARD / 模型表：`dinov2_vitb14_reg` 的
ImageNet-1k linear 指标为 84.6%；仓库 README 另一张"已发布线性头"表对同一个
reg4 头记 84.4，两条都是官方数字，差 0.2 点来自评估记录口径）。
因此本脚本导出 1x3x224x224 的静态输入。

位置编码的处理
--------------
预训练权重的 pos_embed 是按 518x518（37x37 网格）训出来的，权重里
``pos_embed`` 形状为 (1, 1370, 768)。官方在喂 224 图时会在 forward 内调用
``interpolate_pos_encoding()`` 把 37x37 网格 bicubic 插值到 16x16（antialias=True,
offset=0.0）。为了不让导出图里出现 Resize 算子（TPU-MLIR 对 bicubic 支持有限），
这里在**导出前**用官方同一个函数把插值结果算好，直接替换 ``model.pos_embed``
（一次性烘焙，数值与官方逐次插值完全一致）；此后 forward 中
``npatch == N and w == h`` 短路，图中不含 Resize。

其他关键点
----------
1. DINOv2 的 Attention 默认走 torch>=2.0 的 F.scaled_dot_product_attention（xformers
   可用时走 memory_efficient_attention），二者均无法被 torch 导出为 ONNX。
   这里强制 XFORMERS_DISABLED=1 并 monkey-patch Attention.forward 为手写
   q·k^T/√d -> softmax -> ·v（eval 下数值等价），即可顺利导出。
2. 原始 forward 用 `def forward(self, *args, ...)`，dynamo 导出器会因 *args/variadic
   结构报 "inputs[0] is tuple" 的动态形状不匹配错误，因此包一层显式签名
   `def forward(self, x)` 的 wrapper。
3. 输出对齐官方线性探针特征（dinov2/eval/linear.py: create_linear_input，
   use_n_blocks=1 且 use_avgpool=True）：
       concat([x_norm_clstoken, mean(x_norm_patchtokens, dim=1)])  ->  [B, 1536]
   其中 x_norm_patchtokens = x_norm[:, 1+num_register_tokens:]。

用法:
  python3 scripts/export_onnx.py \
      --dinov2_src <path/to/dinov2_repo> \
      --weights models/torch/dinov2_vitb14_reg4_pretrain.pth \
      --out_onnx models/onnx/dinov2_vitb14_reg_224_1b.onnx

导出得到的 `--out_onnx` 是 **onnxsim 简化后**的成品（权重内嵌、单文件），
编译脚本只认这一个文件；未简化的中间产物写到 `<out_onnx 去后缀>_raw.onnx`。
"""
import argparse
import os
import sys

import numpy as np


def patch_attention():
    """将 Attention.forward 替换为 ONNX 可导出的手写 scaled-dot-product attention。"""
    import torch
    from dinov2.layers import attention as _attn

    def _manual_forward(self, x, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]  # [B, H, N, D]
        q = q * self.scale                                   # scale = head_dim ** -0.5
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x

    _attn.Attention.forward = _manual_forward


def bake_pos_embed(model, image_size):
    """把 518(37x37) 的 pos_embed 一次性插值烘焙成 image_size 对应的网格。

    调用的是官方 vision_transformer.VisionTransformer.interpolate_pos_encoding，
    参数（bicubic + antialias + offset）全部来自模型本身，
    因此结果与官方在 forward 中逐次插值完全一致。
    """
    import torch
    import torch.nn as nn

    with torch.no_grad():
        n_patch = (image_size // model.patch_size) ** 2
        # 只用到 x.shape[1]（= 1 + n_patch，注意 register token 在加 pos_embed 之后
        # 才拼接，见 prepare_tokens_with_masks），batch/通道数无影响。
        dummy = torch.zeros(1, 1 + n_patch, model.embed_dim)
        new_pos_embed = model.interpolate_pos_encoding(dummy, image_size, image_size)
    old_shape = tuple(model.pos_embed.shape)
    model.pos_embed = nn.Parameter(new_pos_embed.clone())
    print(
        "[bake] pos_embed {} -> {} (image_size={}, grid={}x{})".format(
            old_shape,
            tuple(model.pos_embed.shape),
            image_size,
            image_size // model.patch_size,
            image_size // model.patch_size,
        )
    )
    return model


def make_wrapper(model, torch, image_size):
    """显式签名包装器，输出官方线性探针的 1536 维特征。"""

    class DinoV2LinearFeature(torch.nn.Module):
        def __init__(self, backbone, image_size):
            super().__init__()
            self.backbone = backbone
            self.num_patches = (image_size // backbone.patch_size) ** 2
            # 均值用 MatMul 实现（patch^T @ (1/n)），避免 opset18 的 ReduceMean
            # （TPU-MLIR OnnxOpt 不支持 axes 作为输入的 ReduceMean）。
            self.register_buffer(
                "avg_w",
                torch.full((self.num_patches, 1), 1.0 / self.num_patches, dtype=torch.float32),
                persistent=False,
            )

        def forward(self, input):
            feats = self.backbone.forward_features(input)
            cls_token = feats["x_norm_clstoken"]        # [B, 768]
            patch_tokens = feats["x_norm_patchtokens"]  # [B, 256, 768]
            # [B, 768, 256] @ [256, 1] -> [B, 768, 1] -> reshape [B, 768]
            avg_patch = (
                patch_tokens.transpose(1, 2).matmul(self.avg_w).reshape(-1, self.backbone.embed_dim)
            )
            return torch.cat([cls_token, avg_patch], dim=-1)  # [B, 1536]

    wrapper = DinoV2LinearFeature(model, image_size)
    wrapper.eval()
    return wrapper


def export_onnx(wrapper, dummy, out_onnx, opset):
    import torch
    # 静态 batch=1（对齐 sophon-demo 其他分类例程的 1b 口径），无需 dynamic_shapes。
    torch.onnx.export(
        wrapper,
        dummy,
        out_onnx,
        dynamo=True,
        input_names=["input"],
        output_names=["x_linear_feature"],
        opset_version=opset,
        do_constant_folding=True,
    )


def run_ort(path, dummy_np):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": dummy_np})[0]


def verify_onnx(wrapper, raw_onnx, out_onnx, image_size, torch):
    import onnx
    model = onnx.load(raw_onnx)
    onnx.checker.check_model(model)
    print("onnx.checker OK: {}".format(raw_onnx))

    try:
        import onnxsim
        model_sim, check = onnxsim.simplify(raw_onnx)
        assert check, "onnxsim simplify check failed"
        onnx.save_model(model_sim, out_onnx)
        print("onnxsim -> {}".format(out_onnx))
        runtime_onnx = out_onnx
    except Exception as e:  # noqa: BLE001
        print("onnxsim skipped: {}".format(e))
        runtime_onnx = raw_onnx

    # 同时检查：导出图里不应残留 Resize（pos_embed 已烘焙）
    ops = {n.op_type for n in onnx.load(runtime_onnx).graph.node}
    assert "Resize" not in ops, "ONNX 图中残留 Resize，pos_embed 烘焙失败"
    print("no Resize node in graph (pos_embed baked) OK")

    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size)
        torch_out = wrapper(dummy)
        ort_out = run_ort(runtime_onnx, dummy.numpy())
        cos = float(
            np.dot(torch_out.numpy().reshape(-1), ort_out.reshape(-1))
            / (np.linalg.norm(torch_out.numpy()) * np.linalg.norm(ort_out))
        )
        maxdiff = float(np.max(np.abs(torch_out.numpy() - ort_out)))
        print(
            "batch=1: torch {} vs onnx {}  cosine={:.9f}  max_abs_diff={:.3e}".format(
                tuple(torch_out.shape), tuple(ort_out.shape), cos, maxdiff
            )
        )
        assert cos > 0.9999, "ONNX 输出与 PyTorch 不一致！"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinov2_src", type=str, default="", help="dinov2 仓库路径")
    ap.add_argument("--weights", type=str, required=True, help="预训练骨干 .pth 路径")
    ap.add_argument("--out_onnx", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=224, help="官方 eval 口径 224")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--keep_raw", action="store_true", help="保留未简化的 <out>_raw.onnx 中间产物")
    args = ap.parse_args()

    os.environ["XFORMERS_DISABLED"] = "1"
    if args.dinov2_src and os.path.isdir(args.dinov2_src) and args.dinov2_src not in sys.path:
        sys.path.insert(0, args.dinov2_src)

    import torch

    patch_attention()
    from dinov2.hub.backbones import dinov2_vitb14_reg  # noqa: E402

    # 按权重原生 518 构造（pos_embed 37x37），保证 load_state_dict(strict=True) 成功；
    # 随后再烘焙到 224 的网格。
    model = dinov2_vitb14_reg(pretrained=True, weights=args.weights)
    model.eval()
    bake_pos_embed(model, args.image_size)

    wrapper = make_wrapper(model, torch, args.image_size)
    raw_onnx = args.out_onnx.replace(".onnx", "_raw.onnx")
    export_onnx(wrapper, torch.randn(1, 3, args.image_size, args.image_size), raw_onnx, args.opset)
    # 简化后的成品才是交付/编译用的 ONNX（权重内嵌单文件）
    verify_onnx(wrapper, raw_onnx, args.out_onnx, args.image_size, torch)
    # 未简化的中间产物只在自检时用得到，默认清掉，避免 models/onnx 下留重复的大文件
    if not args.keep_raw and os.path.exists(raw_onnx):
        os.remove(raw_onnx)
        print("removed intermediate {} (use --keep_raw to keep)".format(raw_onnx))
    print("ALL ONNX EXPORT CHECKS PASSED")


if __name__ == "__main__":
    main()
