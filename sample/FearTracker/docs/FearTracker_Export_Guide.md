# FEARTracker ONNX 导出指南

## 目录

- [1. 模型架构](#1-模型架构)
- [2. 导出ONNX模型](#2-导出onnx模型)
- [3. 验证与常见问题](#3-验证与常见问题)

## 1. 模型架构

FEARTracker采用Siamese跟踪架构，由以下部分组成：

- **Backbone**: FBNet-C，预训练的分类网络，负责提取图像特征
- **Neck (AdjustLayer)**: 1x1卷积，将特征通道数调整为256
- **BoxTower**: 由MatrixMobile（特征编码）和MobileCorrelation（交叉相关）组成
- **Heads**: 回归头输出bbox_pred [4,16,16]和分类头输出cls_pred [1,16,16]

### 输入输出规格

| 名称       | 形状               | 类型   | 说明                        |
| --------- | -------------------| ------ | --------------------------- |
| template  | [1, 3, 128, 128]   | FP32   | 模板图像（第一帧裁剪的目标区域） |
| search    | [1, 3, 256, 256]   | FP32   | 搜索图像（后续帧中搜索目标的区域） |
| bbox_pred | [1, 4, 16, 16]     | FP32   | 边界框回归预测（左/上/右/下距离） |
| cls_pred  | [1, 1, 16, 16]     | FP32   | 分类预测（目标置信度）         |

### 跟踪器配置参数

| 参数                   | 值      | 说明                           |
| ---------------------- | ------- | ------------------------------ |
| template_size          | 128     | 模板图像尺寸                     |
| instance_size (search) | 256     | 搜索图像尺寸                     |
| score_size             | 16      | 分类/回归特征图尺寸               |
| total_stride           | 16      | 特征图stride                    |
| template_bbox_offset   | 0.2     | 模板裁剪时的边界框扩展比例          |
| search_context         | 2       | 搜索区域裁剪时的上下文扩展系数       |

## 2. 导出ONNX模型

### 2.1 环境准备

导出ONNX需要安装FEARTracker源码项目的依赖：

```bash
cd <feartracker_source>
pip install -r requirements.txt
```

### 2.2 运行导出脚本

```bash
cd <sophon-demo>/sample/FearTracker
PYTHONPATH=<feartracker_source> python3 tools/export_onnx.py \
    --checkpoint <feartracker_source>/evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt \
    --output tools/feartracker.onnx
```

参数说明：
- `--checkpoint`: 训练好的模型权重路径（.ckpt文件）
- `--output`: 输出ONNX文件路径
- `--opset`: ONNX opset版本，默认16

### 2.3 导出注意事项

1. **FBNet backbone**: 模型依赖`mobile_cv`包中的FBNet-C预训练权重。导出时需要将`pretrained=False`传入模型配置，避免加载预训练权重（实际权重来自checkpoint）。
2. **算子兼容性**: 模型中使用了exp、mul等算子，BM1684X和BM1688均支持。
3. **非图像模型**: 两个输入尺寸不同（128x128和256x256），编译时需指定`--channel_format none`。

## 3. 验证与常见问题

### 3.1 验证ONNX模型

```python
import onnx
import onnxruntime

model = onnx.load("tools/feartracker.onnx")
onnx.checker.check_model(model)
print("ONNX check passed")
```

### 3.2 常见问题

1. **mobile_cv导入失败**: 确保FBNet相关依赖已正确安装（`pip install -e git+https://github.com/facebookresearch/mobile-vision.git`）。

2. **checkpoint加载失败**: 导出脚本使用`load_from_lighting`加载PyTorch Lightning格式的checkpoint，确保checkpoint文件完整且格式正确。

3. **ONNX导出包含动态维度**: 确保torch.onnx.export时使用固定的dummy输入（batch_size=1，不使用dynamic_axes）。