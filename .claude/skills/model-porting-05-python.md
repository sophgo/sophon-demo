---
name: model-porting-05-python
description: 生成 Python SAIL 推理代码。使用 BMCV 前后处理，支持命令行参数和多 batch。
---

# 步骤 05：Python 推理

## 前置

步骤 04 已完成（BModel 文件已编译）。

## 提示词

```
参考已有 sample 的 Python 推理代码，生成 [模型名] 的 Python SAIL 推理脚本：

1. 使用 sail.Engine 加载 BModel，IOMode.SYSIO
   - 注意：output_names 在不同芯片上可能不同，用 self.output_names[0] 而非硬编码

2. 预处理（BMCV）：
   - 读取图像 → BGR→RGB（如果模型需要） → resize 到目标尺寸
   - 减均值除方差：(pixel - mean) / std
   - HWC→CHW，构造 batch
   - 参数从 model_info.json 中提取

3. 推理：engine.process(graph_name, input_tensor)

4. 后处理：根据模型类型实现
   - 分类：argmax(logits)
   - 检测：decode bbox + NMS
   - 识别：取 embedding

5. CLI：argparse 支持 --bmodel、--input、--dev_id、--loops

6. 计时：pre_process / inference / post_process 分阶段统计

输出目录：python/
```

## 预期输出

- `python/[model_name]_infer.py`
- `python/README.md`

## 内联知识：BMCV 预处理公式

```
convert_to 的 alpha 和 beta 参数：
  alpha = 1.0 / (255.0 * std)   # 如不需额外 scale
  beta  = -mean / std
```

如果模型有额外的 scale 因子：
```
  alpha = 1.0 / (255.0 * std) * scale
  beta  = (-mean / std) * scale
```

常见预处理参数速查：
| 模型系列 | mean | std | resize | 通道 |
|---------|------|-----|--------|------|
| ImageNet 标准 | [0.485,0.456,0.406] | [0.229,0.224,0.225] | 224/256/299 | RGB |
| 0~1 归一化 | [0,0,0] | [1,1,1] | 按模型 | RGB |
| -1~1 归一化 | [0.5,0.5,0.5] | [0.5,0.5,0.5] | 按模型 | RGB |

## Debug

| 问题 | 排查方向 |
|------|---------|
| BMRT_ASSERT 加载失败 | bmodel 与 libsophon 版本不兼容 |
| 输出 NaN | 检查预处理参数：alpha/beta 计算是否正确，值域是否匹配 |
| 输出与预期不符 | 对比 ONNX Runtime 输出，确认 BModel 编译正确 |
| 性能差 | 检查是否用了 SYSIO（非 SYSI），SYSIO 避免设备内存拷贝 |
