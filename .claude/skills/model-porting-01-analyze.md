---
name: model-porting-01-analyze
description: 分析模型 README 和源码，提取架构、输入输出规格、预处理参数。输出 model_info.json。
---

# 步骤 01：分析模型

## 前置

用户已完成 `skills/model-porting-template.md` 模板填写。从模板中读取：模型名称、原始框架、代码仓库路径。

## 提示词

```
分析 [模型名] 的 README 和源码，提取以下信息，输出为 model_info.json：

1. 模型架构：backbone/neck/head 结构、单模型还是多模型串联、子模型个数
2. 输入规格：shape [b,c,h,w]、dtype（float32/int32/uint8）、通道顺序（RGB/BGR）、动态维度列表
3. 预处理参数：resize 尺寸、mean、std、scale、是否需要 BGR→RGB 转换
4. 输出规格：每个输出的 name、shape、dtype、含义
5. 原始框架和版本：PyTorch / TensorFlow / PaddlePaddle，及关键依赖（timm/torchvision 等）
6. 关键算子列表：列出所有独特算子（如 PReLU、HardSwish、GELU、LayerNorm、MultiHeadAttention）
7. 推荐参考 Sample：在 sample/ 目录下查找同算法类别中最完善的已有 sample

如果 README 信息不全，请阅读源码中的模型定义文件补充。
```

## 预期输出

```json
{
  "model_name": "mobilenetv4_conv_medium",
  "framework": "pytorch",
  "algorithm": "image_classification",
  "sub_models": 1,
  "input": {
    "shape": [1, 3, 224, 224],
    "dtype": "float32",
    "channel_order": "RGB",
    "dynamic_axes": ["batch"]
  },
  "preprocess": {
    "resize": [224, 224],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "bgr_to_rgb": true
  },
  "output": [
    { "name": "logits", "shape": [1, 1000], "dtype": "float32" }
  ],
  "ops": ["Conv2d", "BatchNorm", "HardSwish", "GlobalAveragePool", "Linear"],
  "ref_sample": "sample/ResNet"
}
```

## 内联知识

### 参考 Sample 速查

| 算法类别 | 推荐参考 Sample |
|---------|---------------|
| 图像分类 | `sample/ResNet` |
| 目标检测 | `sample/YOLOv5` |
| 人脸识别 | `sample/ArcFace` |
| 人脸检测 | `sample/RetinaFace` |
| 语音识别 | `sample/SeAcoParaformer` |
| OCR | `sample/PP-OCR` |
| 语义分割 | `sample/segformer` |
| 姿态估计 | `sample/HRNet_pose` |
| LLM | `sample/Qwen` |

### 通道顺序

PyTorch 模型默认 RGB，OpenCV 读图为 BGR。如果预处理需要用 OpenCV，必须做 BGR→RGB 转换。该信息会影响后续步骤 05 和 08 的预处理实现。
