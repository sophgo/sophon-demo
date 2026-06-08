# ArcFace 模型移植申请

> 基于 `model-porting-template.md` 填写，用于 SOPHON BM1684X 移植。

---

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| 模型名称 | ArcFace (InsightFace) |
| 算法类别 | 人脸识别 |
| 原始框架 | PyTorch |
| 原始代码仓库 | https://github.com/deepinsight/insightface |
| 预训练模型路径 | insightface/recognition/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth |
| 论文/参考文档 | https://arxiv.org/abs/1801.07698 |

---

## 2. 模型架构

### 2.1 模型结构

```
输入 (112x112 RGB 人脸图像) → 预处理 (Resize + Normalize)
  → 骨干网络 [IR-SE-ResNet50] (TPU)
  → FC layer → 特征向量 [1, 512] → L2 归一化 (CPU) → 输出 Embedding
```

### 2.2 子模型拆分

> 单模型，无需拆分。

| 子模型 | 名称 | 功能描述 | 输入 shape | 输出 shape |
|--------|------|---------|-----------|------------|
| 模型 | resnet50_backbone | IR-SE-ResNet50 特征提取 + FC | [1, 3, 112, 112] | [1, 512] |

### 2.3 关键算子

```
- 标准 Conv2d + BatchNorm + PReLU
- SE-Block (GlobalAvgPool + FC + Sigmoid)
- DepthwiseConv (stem 层)
- Linear (FC 512-dim)
- BatchNorm1d (FC 后)
```

---

## 3. 输入规格

| 参数 | 值 |
|------|-----|
| 输入类型 | 图像 |
| 输入格式 | .jpg / .png (RGB 对齐人脸) |
| 输入尺寸 | 112x112 RGB 图像 |
| 输入 shape (模型) | [batch, 3, 112, 112] |
| 输入 dtype | float32 |
| 动态维度 | batch 维度 (支持 1b 和 4b) |
| 归一化方式 | mean=[127.5, 127.5, 127.5], scale=[0.0078125, 0.0078125, 0.0078125]（即 (pixel/255 - 0.5) / 0.5） |

---

## 4. 预处理详情

### 4.1 预处理流程

```
1. 读取对齐后的人脸图片 (RGB 格式)
2. Resize 到 112x112 (直接拉伸，不需要保持宽高比)
3. 归一化: (pixel - 127.5) * 0.0078125 ≈ (pixel/255 - 0.5)/0.5 → 值域[-1, 1]
4. 转换为 NCHW float32 格式
```

### 4.2 预处理参数

| 参数 | 值 |
|------|-----|
| 目标尺寸 | 112x112 |
| 保持宽高比 | 否 (直接 resize) |
| Padding 填充值 | 无 |
| 颜色格式 | RGB |
| 均值 (mean) | [127.5, 127.5, 127.5] |
| 方差/缩放 (std/scale) | [0.0078125, 0.0078125, 0.0078125] |
| 输出值域 | [-1, 1] |

---

## 5. 后处理详情

### 5.1 后处理流程

```
1. 取模型输出 [1, 512] 特征向量
2. L2 归一化: embedding = embedding / sqrt(sum(embedding^2))
3. (应用层) 与注册库中的 N 个 embedding 做矩阵乘法 → N 个余弦相似度
4. 取 argmax 获得最佳匹配 ID
```

### 5.2 输出规格

| 输出 | shape | dtype | 含义 |
|------|-------|-------|------|
| output | [1, 512] | float32 | 人脸特征向量 (L2归一化前) |

---

## 6. BModel 编译需求

### 6.1 目标芯片和精度

| 参数 | 值 |
|------|-----|
| 目标芯片 | BM1684X |
| 需要编译的精度 | FP32, FP16, INT8, INT8_4b (四精度) |
| 需要编译的 batch | 1b, 4b |
| 最大输入长度 | N/A (固定 shape 112x112) |
| 是否启用动态 shape | 否（固定输入尺寸） |

### 6.2 INT8 校准数据（仅 INT8 需要）

| 参数 | 值 |
|------|-----|
| 校准数据集路径 | 使用标准人脸数据集 (如 LFW/MS1M 子集) |
| 校准样本数 | 100 |
| 校准数据格式 | 112x112 RGB 对齐人脸图片 |

---

## 7. 精度测试需求

### 7.1 精度指标

| 指标名称 | 计算方式 | 目标值 |
|---------|---------|--------|
| 余弦相似度 (FP32) | TPU vs PyTorch embedding 余弦相似度 | > 0.999 |
| 余弦相似度 (FP16) | TPU FP16 vs PyTorch FP32 embedding | > 0.999 |
| 余弦相似度 (INT8) | TPU INT8 vs PyTorch FP32 embedding | > 0.99 |

### 7.2 测试数据

| 参数 | 值 |
|------|-----|
| 测试集路径 | datasets/test (对齐人脸图片) |
| 测试样本数 | 100 |
| 参考模型 | insightface PyTorch backbone.pth |
| 参考推理环境 | PyTorch CPU |

---

## 8. 性能测试需求

| 参数 | 值 |
|------|-----|
| 测试平台 | x86 PCIE |
| 性能指标 | 推理延迟(ms)/FPS |
| 测试次数 | 5 次取平均 |
| 业务性能要求 | 单张推理 < 5ms |

---

## 9. 部署形态

| 参数 | 值 |
|------|-----|
| Python 推理 | 需要 |
| C++ 推理 | 需要 |
| C++ 推理 SDK | BMRT (bmrt) |
| 前后处理方式 | BMCV (Python/C++ 都使用 BMCV) |
| SoC 部署 | 暂不需要 |
| SoC 设备型号 | N/A |

---

## 10. 依赖和环境

### 10.1 Python 依赖

```
- torch >= 1.9.0
- numpy
- opencv-python (用于图片读取)
- sophon-sail (Python SAIL 推理)
```

### 10.2 C++ 依赖

```
- libsophon (bmrt, bmlib, bmcv)
- OpenCV (sophon-opencv)
- FFmpeg (sophon-ffmpeg，用于视频流场景)
```

### 10.3 特殊依赖

```
- onnx >= 1.10.0
- onnxruntime (ONNX 验证用)
- insightface (仅用于获取参考模型和测试)
```

---

## 11. 其他信息

### 11.1 已知问题或注意事项

```
- SE-Block 中的 GlobalAveragePool 需要确认 TPU-MLIR 支持
- BN1d 在 ONNX 导出时需要特殊处理（reshape 到 2D）
- PReLU 算子需要确认 TPU-MLIR 支持
- 建议使用 opset 13 导出 ONNX
```

### 11.2 参考已有 Sample

```
sample/RetinaFace (人脸检测 + C++/Python bmcv)
sample/SCRFD (人脸检测 + C++/Python bmcv)
sample/ResNet (图像分类 + 单模型推理，架构最接近)
```

### 11.3 额外需求

```
- 支持 batch=4 推理（一次推理 4 张人脸）
- 输出 embedding 需要 L2 归一化
```
