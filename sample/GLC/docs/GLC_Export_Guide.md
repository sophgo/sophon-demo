# GLC 模型导出指南（PyTorch → ONNX）

GLC（Generative Latent Coding, CVPR 2024）是 VQ-VAE + 空间类别超先验的生成式图像压缩模型。
本文说明如何把它拆成两个可上 TPU 的 ONNX 子模型，以及拆分边界的选取依据。

## 1. 为什么必须拆成两个模型

GLC 的推理链路（`GLC_Image.test()`）：

```
x ──► vqgan.encoder ──► enc(·, q_enc) ──► y ──► hyper_enc ──► z
                                                              │
                                          z_vq.get_indices(z) │  ← argmin，数据依赖
                                          z_vq.get_quan_feat  │
                                                              ▼
                                                             z_hat
                                                              │
y ──► forward_four_part_prior(·, params) ◄── y_prior_fusion ◄── hyper_dec
                │
                ├──► y_hat ──► dec(·, q_dec) ──► vqgan.generator ──► x_hat
                ├──► y_q          （用于熵模型算 bpp）
                └──► scales_hat   （用于熵模型算 bpp）
```

`z_vq.get_indices()` 是一个 **16384 类码本上的最近邻搜索（argmin）**。argmin 的输出是离散索引，
后续 `get_quan_feat` 又把它变回连续特征——这种「离散选择 → 连续查表」的跳变在 TPU 上
既难展开也不划算（`argmin` 在 TPU-MLIR 中支持有限，且 16384 类查表会显著增加图规模）。
GLC 官方实现同样把这一步留在通用算子上，因此我们在编码器与解码器之间插入一段 CPU 码本查找。

**拆分边界**：

| 子模型 | 输入 | 输出 |
|--------|------|------|
| `glc_encoder` | `x` (1,3,H,W)、`q_enc` (1,256,1,1) | `y` (1,256,H/16,W/16)、`z` (1,256,H/64,W/64) |
| `glc_decoder` | `y`、`z_hat`、`q_dec` (1,256,1,1) | `x_hat` (1,3,H,W)、`y_q`、`scales_hat` |

解码器额外输出 `y_q` 与 `scales_hat`，是为了让设备端能就地算 bpp，无需再跑一次熵模型。

## 2. 导出步骤

### 2.1 环境

GLC 源码依赖 `torchvision`，但 `layers.py` 里的 `from torchvision import models` 是**无用 import**
（该文件从未使用 `models`）。若环境里没有匹配 torch 版本的 torchvision，可用 `tools/stubs/` 里的
轻量替身绕过。

`tools/stubs/torchvision/` 只提供 `models`（占位）与 `transforms.ToTensor`（GLC 的指标脚本会用到）。

### 2.2 导出

导出脚本需要读取 GLC 源码，因此**在 GLC 源码根目录下执行**，通过 `ONNX_OUT` 指定产物目录：

```bash
cd <GLC源码根目录>
PYTHONPATH=<本demo路径>/tools/stubs ONNX_OUT=<本demo路径>/models/onnx \
    IMG_H=512 IMG_W=768 python3 <本demo路径>/tools/export_onnx.py
```

执行后会在 `models/onnx/` 下生成 `glc_encoder.onnx` 与 `glc_decoder.onnx`。

要点：

- **必须用 `inplace=False` 构造 `GLC_Image`**。默认的 `inplace=True` 会让 Split 的输出被原地修改，
  `torch.onnx.export` 会直接报
  `RuntimeError: Output 1 of SplitBackward0 is a view and is being modified inplace`。
- 导出函数需显式 `dynamo=False`。新版 torch 默认走 `torch.export` 路径，需要额外的 `onnxscript` 依赖。
- 输入尺寸必须是 64 的整数倍（GLC 的 padding_size=64），且 H 能被 64 整除才能得到整数倍的 `z` 尺寸。

### 2.3 数值校验

导出后务必与 PyTorch 原模型逐张量对齐，确认拆分没有改变语义。把编码器 ONNX 的输出与
PyTorch 的 `net.hyper_enc(net.enc(net.vqgan.encoder(x), q))` 逐元素比对即可：

```bash
cd <GLC源码根目录>
PYTHONPATH=<本demo路径>/tools/stubs python3 - <<'PY'
import sys, numpy as np, torch, onnxruntime as ort
sys.path.insert(0, "<GLC源码根目录>")
from src.models.image_model import GLC_Image
from src.utils.test_utils import get_state_dict
from PIL import Image

net = GLC_Image(inplace=False)
net.load_state_dict(get_state_dict("GLC_image.pth.tar"), strict=False)
net.eval()

img = np.asarray(Image.open("<本demo路径>/datasets/kodak24/kodim01.png").convert("RGB"))
x = np.ascontiguousarray(((img.astype(np.float32) / 255.0 - 0.5) * 2.0).transpose(2, 0, 1)[None])
q = np.load("<本demo路径>/models/BM1684X/q_enc.npy")[0:1]

with torch.no_grad():
    y_ref = net.enc(net.vqgan.encoder(torch.from_numpy(x)), torch.from_numpy(q))
    z_ref = net.hyper_enc(y_ref)
    y_ref, z_ref = y_ref.numpy(), z_ref.numpy()

s = ort.InferenceSession("<本demo路径>/models/onnx/glc_encoder.onnx",
                         providers=["CPUExecutionProvider"])
y, z = s.run(None, {"x": x, "q_enc": q})
print("y 相对误差 %.3e" % (np.abs(y - y_ref).max() / np.abs(y_ref).max()))
print("z 相对误差 %.3e" % (np.abs(z - z_ref).max() / np.abs(z_ref).max()))
PY
```

实测结果：编码器输出 `y`/`z` 相对误差 ~9e-5，码本索引完全一致，解码器相对误差 ~3e-5，
均属 float32 舍入级别。设备端 `python/glc_infer.py` 用 FP32 bmodel 跑出的 bpp/PSNR/SSIM
与 PyTorch 基线逐位一致，可作为端到端校验。

## 3. 固定 shape 的限制

导出的 bmodel 是**静态 shape**，只接受导出时指定的分辨率。Kodak24 中有 6 张是竖构图
（768×512），无法直接送入 512×768 的模型；`glc_eval.py` 会自动跳过并在日志中说明。

若要支持其它分辨率，需按该分辨率重新导出 ONNX 并编译，或改用动态 shape 编译
（`model_transform.py` 的 `--input_shapes` 支持 `-1` 维度，但动态 shape 在 BM1684X 上
性能和算子支持都更受限）。

## 4. 精度选择

| 精度 | 可用性 | 说明 |
|------|--------|------|
| FP32 | ✅ | 与 PyTorch 逐位一致，但 BM1684X 的 FP32 吞吐远低于 FP16/BF16，延迟约为 BF16 的 4 倍 |
| FP16 | ❌ | 模型激活动态范围超出 fp16，详见下方 |
| BF16 | ✅ | **推荐**。指数范围与 FP32 相同，精度损失可忽略 |
| INT8 | ❌ | 校准代价过高，详见 [FAQ 的 Q9](../README.md#q9-int8-校准为什么这么慢) |
| W4BF16/W8BF16/INT8BF16DYN | ❌ | 免校准，但在本模型上静默退化为 BF16，无收益 |

### FP16 为什么不可用

GLC 有两处激活幅值超出 fp16 上限（65504），且是模型固有属性：

1. **AttnBlock 的 GroupNorm 方差溢出**。三个 AttnBlock 的归一化输入方差分别达
   6.0e4 / 1.3e5 / 2.4e5，fp16 下 `E[x²]` 直接溢出成 Inf。
   缓解手段：把 GroupNorm 输入先除以 32、同时把 eps 除以 32²（数学上严格等价，
   见 `export_onnx.py` 的 `ScaledGroupNorm`，由 `F16_NORM_FIX=1` 开启）。
2. **解码器 generator 的残差流溢出**。`generator/blocks.22~25` 的残差累加值达
   9.6e4 ~ 1.05e5，超过 65504。这一处无法靠缩放修复（残差分支与主分支共享同一张量，
   缩放会破坏恒等映射），只能整层保持 FP32，而当前 TPU-MLIR 的 F16 编译未提供
   可用的逐算子精度表（MLIR 中算子未携带 ONNX 节点名，`--quantize_table` 无法按名匹配）。

因此本 demo 用 **BF16** 作为降精度方案：指数范围与 FP32 一致，上述两处都不会溢出，
实测 PSNR 与 PyTorch 基线相差 &lt;0.05 dB。

诊断方法见 `tools/export_probe.py`（逐块探测）与 `tools/export_attn_probe.py`（AttnBlock 逐算子探测），
两者都会导出一个多输出 ONNX，编译后在设备上跑一次即可看出第一个出现 NaN 的张量。
