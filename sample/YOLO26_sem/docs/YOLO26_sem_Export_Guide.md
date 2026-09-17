# YOLO26 模型导出（语义分割）

## 1. 准备工作
YOLO26 语义分割模型导出是在 Pytorch 模型的生产环境下进行的，需提前根据 [YOLO26 官方开源仓库](https://github.com/ultralytics/ultralytics) 的要求安装好环境，准备好相应的代码和模型，并保证模型能够在 Pytorch 环境下正常推理运行。推荐导出环境版本：`torch>=2.1`、`onnx>=1.14`、`ultralytics>=8.x`。

源模型为语义分割权重 `yolo26s-sem.pt`（`task="semantic"`，Cityscapes 19 类），与实例分割权重 `yolo26s-seg.pt`（`task="segment"`）不同，请勿混用。

## 2. 导出 onnx 模型
如果使用 tpu-mlir 编译模型，则必须先将 Pytorch 模型导出为 onnx 模型。本例程会把「8 倍 bilinear 上采样 + argmax」一并烘焙进 onnx 图（见下文第 3 节），使导出的 onnx 直接输出逐像素类别图。将以下脚本保存为 `tools/export_onnx.py`，并在 ultralytics venv 中执行：

```python
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO


class BakedSem(nn.Module):
    """对语义分割模型追加 8x bilinear 上采样 + argmax，输出逐像素类别图。"""

    def __init__(self, mm):
        super().__init__()
        self.mm = mm

    def forward(self, x):
        logits = self.mm(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        up = F.interpolate(logits, scale_factor=8, mode="bilinear", align_corners=False)
        return torch.argmax(up, dim=1, keepdim=True)  # [B, 1, H, W] int64


def parse_imgsz(s):
    """解析输入尺寸：支持 '1024,2048' / '1024x2048'（H,W）或单个 int（正方形，反向兼容）。"""
    s = str(s).strip().lower().replace("x", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 1:
        v = int(parts[0])
        return (v, v)
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise ValueError(f"invalid imgsz: {s!r}, expect e.g. 640 or 1024,2048")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[export] load {args.weights} on {device}")
    model = YOLO(args.weights)

    mm = model.model          # ultralytics.nn.tasks.SemanticSegmentationModel (nn.Module)
    mm.eval()
    head = mm.model[-1]       # ultralytics.nn.modules.head.SemanticSegment
    assert type(head).__name__ == "SemanticSegment", f"unexpected head: {type(head).__name__}"
    # 关键：关闭 ultralytics 自带的导出烘焙路径，拿到原始 logits [B, nc, H/8, W/8]，
    # 再由 BakedSem 追加与官方后处理一致的 8x bilinear 上采样 + argmax。
    head.export = False

    H, W = args.imgsz
    dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)
    baked = BakedSem(mm)
    with torch.no_grad():
        ref = baked(dummy)
    print(f"[export] pytorch output shape = {tuple(ref.shape)}, dtype = {ref.dtype}")

    torch.onnx.export(
        baked,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["images"],
        output_names=["segmap"],
        do_constant_folding=True,
        # torch 2.9+ 默认走 dynamo 导出器（会导出 opset 18 + 外部数据），
        # 这里强制走 legacy TorchScript 导出器以获得与仓库一致的 opset 13 单文件 onnx。
        dynamo=False,
        external_data=False,
    )
    print(f"[export] onnx saved to {args.output}")

    # 可选：onnx-simplifier
    if not args.no_simplify:
        try:
            import onnx
            from onnxsim import simplify
            onnx_model = onnx.load(args.output)
            model_simp, check = simplify(onnx_model)
            assert check, "onnxsim check failed"
            onnx.save(model_simp, args.output)
            print("[export] onnxsim simplified ok")
        except ImportError:
            print("[export] onnxsim not installed, skip simplify")

    # 校验合法性
    try:
        import onnx
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("[export] onnx.checker pass")
    except ImportError:
        print("[export] onnx not installed, skip checker")


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument("--weights", type=str, default="yolo26s-sem.pt", help="YOLO semantic weights")
    parser.add_argument("--output", type=str, default="../models/onnx/yolo26s-sem.onnx", help="output onnx path")
    parser.add_argument("--imgsz", type=str, default="1024,2048", help="input size H,W (e.g. 1024,2048) or square int")
    parser.add_argument("--opset", type=int, default=13, help="onnx opset")
    parser.add_argument("--no_simplify", action="store_true", help="skip onnxsim")
    return parser.parse_args()


if __name__ == "__main__":
    args = argsparser()
    args.imgsz = parse_imgsz(args.imgsz)
    main(args)
```

```bash
# 在 ultralytics venv 中执行
python3 tools/export_onnx.py --weights yolo26s-sem.pt --imgsz 1024,2048 --output models/onnx/yolo26s-sem.onnx
```

> **说明**：本例程输入为 Cityscapes 原生分辨率 1024×2048（H×W，宽高比 2:1，无需 letterbox），`--imgsz 1024,2048` 即导出 `[1, 3, 1024, 2048]` 的静态输入。`scripts/download.sh --onnx` 下载的 `models/onnx/yolo26s-sem.onnx` 已按此分辨率导出并完成烘焙，直接用于编译即可。

## 3. 关键点：把 bilinear 上采样 + argmax 烘焙进图
导出的语义分割头输出逐像素类别 logits `[B, 19, H/8, W/8]`，而本仓库的 bmodel 需要模型直接输出类别图，因此还需在 onnx 图上追加两个算子（下载的 `models/onnx/yolo26s-sem.onnx` 已完成此处理）：

- `Resize`：8 倍 bilinear 上采样，`coordinate_transformation_mode="half_pixel"`，与 ultralytics 官方后处理 `F.interpolate(mode="bilinear", align_corners=False)` 逐位一致；
- `ArgMax`：axis=1, keepdims=1，得到逐像素类别图 `segmap [B, 1, H, W]`。

这样模型直接输出类别图，推理端后处理退化为「去 letterbox padding + 缩放回原图」。几点说明：

1. bilinear 上采样放在模型内、且顺序与 ultralytics 一致（先上采样再 argmax），保证精度对齐；
2. TPU-MLIR 会把 `Resize`+`ArgMax` 降低为 `tpu.Interp` + `tpu.Arg`，二者均为 TPU 上高效支持的算子；
3. onnx 侧 `ArgMax` 输出 dtype 为 int64（ONNX 规范约定），编译成 bmodel 后 TPU 的 `ArgMax` 原生输出为 int32，故 bmodel 输出为 `[B, 1, H, W]` int32 类别图，推理端按 int32 读取即可。

导出后的 onnx 输入/输出为：

| 项   | 名称     | shape             | dtype   | 说明 |
| ---- | -------- | ----------------- | ------- | ---- |
| 输入 | images   | [1, 3, 1024, 2048]  | float32 | RGB，/255 归一化（0~1）|
| 输出 | segmap   | [1, 1, 1024, 2048]  | int64    | 逐像素类别图（ArgMax 输出 int64；编译成 bmodel 后为 int32）|