#!/usr/bin/env python3
"""
SeACoParaformer CPU 参考推理测试

该脚本用于在 CPU 上使用 FunASR 框架对 PyTorch 模型进行推理，
作为 TPU SAIL 推理的精度参考。需要先准备 PyTorch 模型文件（model.pt 等）
放在 ./model/ 目录下。

注意：正常使用 TPU 推理时不需要此脚本，请使用 python/seaco_paraformer.py。
"""

import torch
import os

model_dir = "./model"

print("=" * 60)
print("1. 加载 SeACoParaformer 模型 (CPU)")
print("=" * 60)

# 方式1: 使用 FunASR AutoModel 加载模型
from funasr import AutoModel

model = AutoModel(
    model=model_dir,
    device="cpu",
)

print("\n✅ 模型加载完成")
print(f"   设备: CPU")
print(f"   模型目录: {model_dir}")

# ============================================================
# 2. 打印模型结构
# ============================================================
print("\n" + "=" * 60)
print("2. 模型结构")
print("=" * 60)

# FunASR 模型内部持有多个子模型，通过 model 对象查看
if hasattr(model, 'model'):
    print("\n--- ASR 模型 (SeacoParaformer) ---")
    print(model.model)

    # 统计参数
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

# 打印各个子模块
if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
    print("\n--- Encoder (SANMEncoder) ---")
    print(model.model.encoder)
    enc_params = sum(p.numel() for p in model.model.encoder.parameters())
    print(f"Encoder 参数量: {enc_params:,}")

if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
    print("\n--- Decoder (ParaformerSANMDecoder) ---")
    print(model.model.decoder)
    dec_params = sum(p.numel() for p in model.model.decoder.parameters())
    print(f"Decoder 参数量: {dec_params:,}")

if hasattr(model, 'model') and hasattr(model.model, 'seaco_decoder'):
    print("\n--- SeACo Decoder (ParaformerSANMDecoder) ---")
    print(model.model.seaco_decoder)
    seaco_params = sum(p.numel() for p in model.model.seaco_decoder.parameters())
    print(f"SeACo Decoder 参数量: {seaco_params:,}")

if hasattr(model, 'model') and hasattr(model.model, 'predictor'):
    print("\n--- Predictor (CifPredictorV3) ---")
    print(model.model.predictor)
    pred_params = sum(p.numel() for p in model.model.predictor.parameters())
    print(f"Predictor 参数量: {pred_params:,}")

if hasattr(model, 'model') and hasattr(model.model, 'bias_encoder'):
    print("\n--- Bias Encoder (LSTM) ---")
    print(model.model.bias_encoder)
    bias_enc_params = sum(p.numel() for p in model.model.bias_encoder.parameters())
    print(f"Bias Encoder 参数量: {bias_enc_params:,}")

if hasattr(model, 'model') and hasattr(model.model, 'bias_decoder'):
    print("\n--- Bias Decoder ---")
    print(model.model.bias_decoder)
    bias_dec_params = sum(p.numel() for p in model.model.bias_decoder.parameters())
    print(f"Bias Decoder 参数量: {bias_dec_params:,}")

# 打印每个子模块的参数详情
print("\n" + "-" * 60)
print("各模块参数量统计")
print("-" * 60)
for name, param in model.model.named_parameters():
    print(f"  {name:<60s}  shape={str(list(param.shape)):<20s}  requires_grad={param.requires_grad}")

# ============================================================
# 3. CPU 推理 - 不带热词
# ============================================================
print("\n" + "=" * 60)
print("3. CPU 推理测试 (不带热词)")
print("=" * 60)

wav_file = f"{model_dir}/example/asr_example.wav"
if os.path.exists(wav_file):
    print(f"输入音频: {wav_file}")
    result = model.generate(input=wav_file)
    print(f"识别结果: {result}")

    if result and len(result) > 0:
        print(f"\n文本: {result[0].get('text', 'N/A')}")
else:
    print(f"⚠️  示例音频不存在: {wav_file}")

# ============================================================
# 4. CPU 推理 - 带热词
# ============================================================
print("\n" + "=" * 60)
print("4. CPU 推理测试 (带热词)")
print("=" * 60)

hotword_wav = f"{model_dir}/asr_example_hotword.wav"
if os.path.exists(hotword_wav):
    print(f"输入音频: {hotword_wav}")
    print(f"热词: 魔搭")
    result = model.generate(input=hotword_wav, hotword="魔搭")
    print(f"识别结果: {result}")

    if result and len(result) > 0:
        print(f"\n文本: {result[0].get('text', 'N/A')}")
else:
    print(f"⚠️  热词示例音频不存在: {hotword_wav}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)