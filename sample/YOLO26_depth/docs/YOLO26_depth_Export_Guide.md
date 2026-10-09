# YOLO26 模型导出（深度估计）

## 1. 准备工作
YOLO26 深度估计模型导出是在 Pytorch 模型的生产环境下进行的，需提前根据 [YOLO26 官方开源仓库](https://github.com/ultralytics/ultralytics) 的要求安装好环境，准备好相应的代码和模型，并保证模型能够在 Pytorch 环境下正常推理运行。推荐导出环境版本：`torch>=2.1`、`onnx>=1.14`、`ultralytics==8.4.105`。

源模型为深度估计权重 `yolo26s-depth.pt`（`task="depth"`），与语义分割权重 `yolo26s-sem.pt`（`task="semantic"`）不同，请勿混用。官方默认评测集为 **NYU Depth V2（Eigen split，654 张 val 图片）**，官方默认输入尺寸为 **768×768**（`imgsz=768`），本例程与官方口径保持一致。

## 2. 导出 onnx 模型
如果使用 tpu-mlir 编译模型，则必须先将 Pytorch 模型导出为 onnx 模型。ultralytics 的 depth 头已经内置了导出烘焙路径（`head.export=True`），无需自研拼接脚本，直接用官方导出命令即可：

```bash
# 在 ultralytics venv 中执行
cd tools
python3 -c \
"from ultralytics import YOLO; \
 YOLO('../models/yolo26s-depth.pt').export(format='onnx', imgsz=768, opset=13, simplify=True, dynamic=False)"
```

导出完成后，onnx 单文件保存在 `models/yolo26s-depth.onnx`。编译 bmodel 前请将其移动到 `models/onnx/yolo26s-depth.onnx`（`scripts/gen_*bmodel_mlir.sh` 从该路径读取）。

> **说明**：源权重 `yolo26s-depth.pt` 不随 `download.sh` 提供，需自行准备；在 ultralytics 环境中直接 `YOLO('yolo26s-depth.pt')` 会自动从官方仓库下载该权重。

> **说明**：onnx 简化依赖 `onnxsim`，若未安装可去掉 `simplify=True`（tpu-mlir 建议对 onnx 做 simplify，以减少算子数量）。本例程不导出 dynamic shape（固定 `[1, 3, 768, 768]` 静态输入），与 tpu-mlir 编译脚本一致。

## 3. 关键点：深度解码全部烘焙进图
`yolo26s-depth.pt` 的 depth 头（`ultralytics.nn.modules.head.Depth`）在 `export=True` 时会自动把下面三步全部烘焙进 onnx，使模型直接输出绝对尺度度量深度图（单位米），推理端后处理退化为「取出深度图 + 按需伪彩色可视化」：

1. **log 深度还原**：`depth = exp(out.clamp(-4.0, 5.0))`，把网络输出的 logit 还原为线性深度并截断到合理范围；
2. **log-affine 标定**：`depth = depth.pow(cal_a) * exp(cal_b)`，其中 `cal_a=1.0`、`cal_b` 为标定 offset（随权重发布，烘焙为常量），把 Scale-only 深度对齐到绝对尺度；
3. **4× bilinear 上采样**：`F.interpolate(depth, scale_factor=4.0, mode="bilinear", align_corners=False)`，把 `H/4×W/4` 的稠密深度图上采样到 `H×W`（本例程 768×768）。

几点说明：

1. 特征融合阶段（P3/P4/P5 之间）的逐级 2× 上采样使用 `align_corners=True`，该配置已固化在发布的深度权重里，是官方评测口径的一部分，onnx 导出忠实保留；
2. 最终 4× 上采样使用 `align_corners=False`，与 ultralytics 官方后处理 `F.interpolate(mode="bilinear", align_corners=False)` 逐位一致；
3. bmodel 输出即为 `[1, 1, 768, 768]` float32 度量深度图（单位米），**无需任何 NMS / 阈值 / 裁边 / 逆上采样**，推理端直接读取即可。

## 4. onnx 输入/输出

| 项   | 名称   | shape            | dtype   | 说明 |
| ---- | ------ | ---------------- | ------- | ---- |
| 输入 | images | [1, 3, 768, 768] | float32 | RGB，`stretch resize` 到 768×768 后 `/255` 归一化（0~1）|
| 输出 | output0| [1, 1, 768, 768] | float32 | 绝对尺度度量深度图（单位米）|

> **说明**：官方 val 采用 `LetterBox(new_shape=768, scale_fill=True)`（即无 padding 的 stretch resize），因此预处理统一为 **stretch resize（cv2.INTER_LINEAR）到 768×768** → **BGR→RGB** → **/255**，不使用 `--keep_aspect_ratio`。

## 5. 预处理与后处理口径（与官方 val 对齐）
- **图片侧**：`stretch resize`（`cv2.resize(..., INTER_LINEAR)`）到 768×768，`BGR→RGB`，除以 255；
- **GT 侧**：NYU Depth V2 深度真值为 480×640 float16 `.npy`，官方 val 先 `nan_to_num(0)` 再 `cv2.INTER_NEAREST` resize 到 768×768，与预测逐像素比对；
- **评测指标**：Eigen 协议 + 逐图 median 对齐（Depth-Anything 口径），仅在 `gt∈(0.001, 100)` 的有效像素上统计 `delta1/delta2/delta3 / abs_rel / rmse / silog`，逐像素池化（float64）。以上口经由 `tools/eval_depth.py` 完整复刻，无需额外处理 GT。

官方 PyTorch 基线（`model.val(data=..., imgsz=768)`，CPU）为：`delta1=0.7856`，`abs_rel=0.1540`，`rmse=0.6518`，`silog=22.28`，可用于核对导出前后无掉点。