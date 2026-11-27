# MoGe-2 模型导出与移植指南

本文档说明如何将微软 [MoGe](https://github.com/microsoft/MoGe) 官方仓库的 MoGe-2 (ViT-L/14) PyTorch 权重导出为静态 ONNX，并编译为 SOPHON TPU 可运行的 BModel。

## 1. 准备工作

MoGe-2 模型导出/推理在 PyTorch 环境下进行，需安装 MoGe 官方仓库及其依赖：

```bash
git clone https://github.com/microsoft/MoGe.git
cd MoGe
pip install -e .        # 安装 moge 包（依赖 torch、torchvision、utils3d 等）
```

MoGe-2 预训练权重（`moge-2-vitl-normal` 为官方默认 331M 模型）托管在 HuggingFace：

```bash
# 国内网络可通过镜像加速
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Ruicheng/moge-2-vitl-normal moge-2-vitl-normal.pt \
    --local-dir models --local-dir-use-symlinks False
```

## 2. 导出 ONNX 模型

MoGe 官方仓库的 `docs/onnx.md` 给出了固定 token 数的静态导出方式。采样点数 `num_tokens` 决定 patch 内采样的 token 数量，官方默认 `num_tokens=1800`、输入 `FIXED_IMAGE_INPUT = torch.rand(1, 3, 518, 518)`，即静态输入 **1×3×518×518**。

导出脚本（权重即上文 HuggingFace 下载的 `models/moge-2-vitl-normal.pt`）：

```python
# export_onnx.py
import os
os.environ['XFORMERS_DISABLED'] = '1'
import sys
sys.path.insert(0, 'MoGe')
import torch
from moge.model.v2 import MoGeModel

NUM_TOKENS = 1800
INPUT_H, INPUT_W = 518, 518
CKPT = 'models/moge-2-vitl-normal.pt'
ONNX_FILE = 'models/onnx/moge-2-vitl-normal.onnx'

class MoGeStatic(MoGeModel):
    def forward(self, image: torch.Tensor):
        return super().forward(image, NUM_TOKENS)

model = MoGeStatic.from_pretrained(CKPT)
model.onnx_compatible_mode = True
model.eval()

dummy = torch.rand(1, 3, INPUT_H, INPUT_W)
torch.onnx.export(
    model, (dummy,), ONNX_FILE,
    input_names=['image'],
    output_names=['points', 'normal', 'mask', 'metric_scale'],
    dynamic_axes=None,
    opset_version=14,
)
print('exported', ONNX_FILE)
```

> **要点：**
> - `onnx_compatible_mode = True` 是官方为 ONNX 导出保留的（避免导出含控制流的算子）；
> - 输入为原始 RGB 图像除以 255 后的浮点张量 `[0, 1]`，**ImageNet 均值/方差归一化已内置于 DINOv2 encoder 中**，导出时不做外部归一化，后续 TPU-MLIR 编译（`--pixel_format rgb`、不做 mean/scale）与板端前处理都应保持这一口径；
> - 输出顺序为 `points(1,H,W,3)`、`normal(1,H,W,3)`、`mask(1,H,W)`、`metric_scale(1,)`，均为 fp32，NHWC 布局；
> - opset 14 即可，无动态轴，输入形状固定为 518×518。

## 3. 编译 BModel

导出 ONNX 后，进入例程 `sample/MoGe` 目录，在 TPU-MLIR 环境中使用 `scripts/gen_*bmodel_mlir.sh` 编译：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1688   # FP32
./scripts/gen_fp16bmodel_mlir.sh bm1688   # FP16
```

编译脚本内部流程（以 `model_transform.py` + `model_deploy.py` 为例）：

```bash
model_transform.py \
    --model_name moge-2-vitl-normal \
    --model_def ../models/onnx/moge-2-vitl-normal.onnx \
    --input_shapes [[1,3,518,518]] \
    --pixel_format rgb \
    --mlir moge-2-vitl-normal_1b.mlir

model_deploy.py \
    --mlir moge-2-vitl-normal_1b.mlir \
    --quantize F16 \
    --chip bm1688 \
    --model moge-2-vitl-normal_fp16_1b.bmodel
```

> **要点：**
> - `--input_shapes [[1,3,518,518]]` 固定 batch=1、静态形状；
> - `--pixel_format rgb` 表示输入按 RGB 排布，配合 `model_transform` 默认 `mean=0, scale=1`（即不做归一化），与第 2 节的输入口径一致；
> - BM1688 额外生成 `_2core.bmodel`（`--num_core 2`）。

## 4. 后处理参考

bmodel 输出的四个头需经过与官方 `moge/model/v2.py` 一致的后处理才能得到度量深度/点云/法向，核心步骤：

1. `recover_focal_shift`：对 `points` 做 64×64 最近邻下采样、按 mask 过滤，用黄金分割法（golden-section，等价官方 `moge.utils.geometry_torch.recover_focal_shift`）求解焦距与 z 位移；
2. 由焦距与 aspect 恢复 `fx = focal/2 · √(1+a²)/a`、`fy = focal/2 · √(1+a²)`、`cx = cy = 0.5`；
3. `depth = (points.z + shift) × metric_scale`；
4. `force_projection`：按 pixel-center 定义 `uv = (arange+0.5)/W` 把深度重投影回点云，保证点图与深度/内参自洽。

本例程将此流程实现为纯 NumPy 的 `python/postprocess_numpy.py`（板端无 torch/PIL/scipy 依赖），C++ 例程则逐字节复刻同一口径，详见 [cpp/README.md](../cpp/README.md#4-实现说明)。