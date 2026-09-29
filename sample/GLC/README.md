# GLC 生成式图像压缩

## 目录

- [GLC 生成式图像压缩](#glc-生成式图像压缩)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
    - [3.1 自动下载](#31-自动下载)
    - [3.2 手动准备](#32-手动准备)
  - [4. 模型编译](#4-模型编译)
    - [4.1 导出 ONNX](#41-导出-onnx)
    - [4.2 编译 BModel](#42-编译-bmodel)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
  - [8. FAQ](#8-faq)

## 1. 简介

本例程对 [GLC](https://github.com/jzyustc/GLC)（Generative Latent Coding, CVPR 2024）生成式图像压缩模型进行移植，使之能在 SOPHON BM1684X（SE7-32）与 BM1688（SE9-16）上进行推理测试。

GLC 是 VQ-VAE 加空间类别超先验的生成式压缩模型，推理链路需要拆成两个子模型，中间夹一段 CPU 码本查找：

```
x ──► [glc_encoder.bmodel] ──► y, z
                                │
                    [CPU] 16384 类码本 argmin ──► z_hat
                                │
y, z_hat ──► [glc_decoder.bmodel] ──► x_hat, y_q, scales_hat
                                          │
                          bpp = (bit_y + bit_z) / (H·W)
```

编码器与解码器之间必须插一段 CPU 码本查找：`z_vq` 的 argmin 是数据依赖的离散选择，在 TPU 上展开不经济，GLC 官方实现同样将其置于通用算子上。解码器额外输出 `y_q` 与 `scales_hat`，使设备端能就地计算 bpp，无需再跑一次熵模型。拆分边界的选取依据见 [GLC 模型导出](./docs/GLC_Export_Guide.md)。

## 2. 特性

### 2.1 目录结构说明

```bash
├── datasets              # 存放数据集、量化校准数据集（由 scripts/download.sh 拉取）
├── docs                  # 存放本例程专用文档，如 ONNX 导出、移植常见问题等
├── models                # 存放模型
|   ├── BM1684X           # 存放编译好的 bmodel 与派生资产（SE7-32）
|   ├── BM1688            # 同上（SE9-16）
|   └── onnx              # 存放导出的 onnx
├── python                # 存放 Python 例程及其 README
|   ├── README.md
|   ├── bmrt.py           # libbmrt 的 ctypes 绑定
|   ├── glc_infer.py      # 单图推理例程
|   ├── glc_eval.py       # 数据集批量评估例程
|   ├── glc_encode.py     # 真熵编码压缩端：图 -> 码流
|   ├── glc_decode.py     # 真熵编码还原端：码流 -> 重建图
|   ├── glc_codec.py      # 编解码核心（四段先验 + 码流协议）
|   ├── rans_coder.py    # rANS 的 ctypes 绑定
|   ├── entropy_numpy.py # entropy 层的 numpy 重写（设备端无 torch）
|   └── metrics.py       # PSNR / SSIM
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译、数据下载等 shell 脚本
└── tools                 # 存放辅助脚本
    ├── rans              # rANS 纯 C 实现（真熵编码用，含双架构构建脚本）
    ├── tests             # 真熵编码开发期对账套件
    └── export_*.py       # ONNX 导出、校准数据生成、精度比对等
```

### 2.2 SDK特性

* 支持 BM1684X(SoC)、BM1688(SoC)
* 支持 FP32、BF16 模型编译和推理
* 支持 Python 推理
* 支持图片测试

> FP16 编译可通过但推理结果为 NaN，INT8 校准代价过高，原因见 [8. FAQ](#8-faq) 的 Q3、Q9。

## 3. 准备模型与数据

### 3.1 自动下载

本例程在 `scripts` 目录下提供了模型和数据集的下载脚本 `download.sh`：

```bash
chmod -R +x scripts/
./scripts/download.sh
```

脚本会拉取 `models/BM1684X/`、`models/BM1688/`（两个平台的全部 bmodel 与派生资产）
与 `datasets/kodak24/`，目录已存在时自动跳过（需更新请先删掉对应目录）。

> 注：ONNX 与 GLC 官方权重不在下载范围内，需要时请参考 3.2 自行准备。

每个平台的目录内容相同（以 BM1684X 为例，BM1688 对应 SE9-16）：

```
./models/BM1684X
├── glc_encoder_{fp32,bf16}.bmodel   # 编码器（基础路径）
├── glc_decoder_{fp32,bf16}.bmodel   # 解码器（基础路径）
├── glc_prior_init_{fp32,bf16}.bmodel # 真熵编码：z_hat -> 初始 scales/means
├── glc_stage_{1,2,3}_{fp32,bf16}.bmodel # 真熵编码：四段先验逐段参数
├── glc_tail_{fp32,bf16}.bmodel      # 真熵编码：y_hat -> 重建图
├── codebook.npy                     # z_vq 码本 (16384, 256)，CPU 码本查找
├── q_enc.npy / q_dec.npy            # 编/解码端质量向量 (4, 256, 1, 1)
└── entropy_assets.npz               # 熵编码 CDF 资产（真熵编码路径必需）
```

每个平台共 18 个文件，**BF16 与 FP32 两套子图都在包内**，因此 `bf16` / `mixed` / `fp32`
三个精度档案下载后都开箱可用（`mixed` 是跨芯片部署推荐档案，见 [8. FAQ](#8-faq) 的 Q10）。

两个平台的 bmodel 不可互换，按实际芯片选目录；具体用哪个芯片由 `--models` 指定。


下载的数据包括：

```
./datasets
└── kodak24                           # Kodak24 测试图片（24 张，512x768 与 768x512）
```

### 3.2 手动准备

**GLC 官方权重**：从 https://github.com/jzyustc/GLC 获取 `GLC_image.pth.tar`，放到 `models/` 下。

**派生资产**：码本与质量向量需从权重导出：

```bash
PYTHONPATH=tools/stubs python3 tools/export_assets.py \
    --ckpt models/GLC_image.pth.tar --glc-src <GLC源码> --out models/BM1684X/
```

**ONNX**：参考 [4.1 导出 ONNX](#41-导出-onnx) 自行导出。

**测试集**：从 https://r0k.us/graphics/kodak/ 下载 Kodak24 解压到 `datasets/kodak24/`。

> **注意**：导出的 bmodel 是静态 shape，仅支持导出时指定的分辨率（本例程为 512x768）。Kodak24 中有 6 张是竖构图（768x512），会被 `glc_eval.py` 自动跳过，实际评估 18 张。要支持其它分辨率需按该分辨率重新导出并编译。

## 4. 模型编译

模型编译前需要安装 TPU-MLIR，具体可参考 [TPU-MLIR 环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在 TPU-MLIR 环境中进入例程目录。

### 4.1 导出 ONNX

GLC 的推理链路必须拆成编码器与解码器两个 ONNX，拆分依据与注意事项见 [GLC 模型导出](./docs/GLC_Export_Guide.md)。

```bash
PYTHONPATH=tools/stubs ONNX_OUT=models/onnx \
    IMG_H=512 IMG_W=768 python3 tools/export_onnx.py
```

### 4.2 编译 BModel

脚本带平台参数：SE7-32 传 `bm1684x`（下面示例），SE9-16 传 `bm1688`，产物分别落到 `models/BM1684X/` 与 `models/BM1688/`。

- 生成 FP32 BModel

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

执行上述命令会在 `models/BM1684X` 文件夹下生成 `glc_encoder_fp32.bmodel` 与 `glc_decoder_fp32.bmodel`。

- 生成 BF16 BModel（**推荐**）

```bash
./scripts/gen_bf16bmodel_mlir.sh bm1684x
```

- 生成 FP16 BModel

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

> 当前 TPU-MLIR 的 F16 编译对本模型不可用，产物推理结果为 NaN，保留脚本仅为复现该结论，详见 [8. FAQ](#8-faq) 的 Q3。

- 生成 INT8 BModel

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684x
```

> INT8 校准耗时极高，脚本内已注明，详见 [8. FAQ](#8-faq) 的 Q9。

## 5. 例程测试

- [Python例程](./python/README.md)
    - 理论值路径：`glc_infer.py` / `glc_eval.py`（编解码同进程，bpp 为估算值）
    - 真熵编码路径：`glc_encode.py` / `glc_decode.py`（压缩端与还原端分离，真实码流）
      见 [Python例程 第 4 节](./python/README.md#4-真熵编码压缩端与还原端分离部署)

## 6. 精度测试

在 Kodak24 中尺寸匹配的 18 张测试图片（512x768）上，与 GLC 官方 PyTorch 实现（CPU）的结果进行对比，计算重建图像的 PSNR 与 SSIM，以及码率 bpp：

| 测试平台 | 测试程序 | 测试模型 | 质量档位 | bpp | PSNR(dB) | SSIM |
| -------- | -------- | -------- | -------- | --- | -------- | ---- |
| SE7-32 | glc_eval.py | PyTorch基线 | q0 | 0.0243 | 21.05 | 0.4937 |
| SE7-32 | glc_eval.py | glc_fp32 | q0 | 0.0243 | 21.05 | 0.4937 |
| SE7-32 | glc_eval.py | glc_bf16 | q0 | 0.0244 | 21.01 | 0.4952 |
| SE7-32 | glc_eval.py | PyTorch基线 | q1 | 0.0281 | 21.44 | 0.5119 |
| SE7-32 | glc_eval.py | glc_fp32 | q1 | 0.0281 | 21.44 | 0.5119 |
| SE7-32 | glc_eval.py | glc_bf16 | q1 | 0.0282 | 21.41 | 0.5112 |
| SE7-32 | glc_eval.py | PyTorch基线 | q2 | 0.0327 | 21.78 | 0.5247 |
| SE7-32 | glc_eval.py | glc_fp32 | q2 | 0.0327 | 21.78 | 0.5247 |
| SE7-32 | glc_eval.py | glc_bf16 | q2 | 0.0328 | 21.79 | 0.5251 |
| SE7-32 | glc_eval.py | PyTorch基线 | q3 | 0.0369 | 22.00 | 0.5340 |
| SE7-32 | glc_eval.py | glc_fp32 | q3 | 0.0369 | 22.00 | 0.5340 |
| SE7-32 | glc_eval.py | glc_bf16 | q3 | 0.0370 | 21.99 | 0.5343 |
| SE9-16 | glc_eval.py | glc_bf16 | q0 | 0.0244 | 21.01 | 0.4953 |
| SE9-16 | glc_eval.py | glc_bf16 | q1 | 0.0282 | 21.40 | 0.5106 |
| SE9-16 | glc_eval.py | glc_bf16 | q2 | 0.0328 | 21.78 | 0.5244 |
| SE9-16 | glc_eval.py | glc_bf16 | q3 | 0.0370 | 21.98 | 0.5337 |

> **测试说明**：
> 1. `glc_fp32` 指 `glc_encoder_fp32.bmodel` + `glc_decoder_fp32.bmodel`，`glc_bf16` 同理；
> 2. FP32 在全部四个质量档位、每一张测试图上都与 PyTorch 基线逐位一致，单图最大 PSNR 偏差 0.00 dB，码本索引完全一致；
> 3. BF16 单图最大 PSNR 偏差 0.20 dB，平均偏差在 0.04 dB 以内；
> 4. bpp 由解码器输出的 `y_q` 与 `scales_hat` 按高斯熵模型**就地估算**，是信息论下界，**不是真实码流长度**；`bit_z` 为码本索引的定长开销（8x12 个索引 x log2(16384) bit = 1344 bit）；
>    如需真实码流与编解码分离部署，见 [GLC 真熵编码](./docs/GLC_Entropy_Coding.md)；
> 5. q0 为最低码率档位，q3 为最高；
> 6. SE9-16 目前只跑了 BF16 全量（FP32 单张实测 bpp 0.0268 / PSNR 18.95 dB，与 BF16 同档），需 FP32 全量结果可在设备上跑 `glc_eval.py --precision fp32`。

精度测试的复现方法见 [Python例程](./python/README.md#3-精度与性能测试)。

## 7. 性能测试

### 7.1 bmrt_test

用 `bmrt_test` 测各子模型的纯 TPU 推理时间（不含前后处理与 CPU 码本查找）。
真熵编码路径不需要 `glc_decoder`，改用 `glc_prior_init` + `glc_stage_1..3` + `glc_tail`：

```bash
bmrt_test --bmodel models/BM1684X/glc_encoder_bf16.bmodel --devid 0
```

| 测试平台 | 测试模型 | calculate time(ms) |
| -------- | -------- | ------------------ |
| SE7-32 | glc_encoder_bf16 | 222.11 |
| SE7-32 | glc_decoder_bf16 | 381.86 |
| SE7-32 | glc_prior_init_bf16 | 2.75 |
| SE7-32 | glc_stage_1_bf16 | 2.25 |
| SE7-32 | glc_stage_2_bf16 | 2.28 |
| SE7-32 | glc_stage_3_bf16 | 2.25 |
| SE7-32 | glc_tail_bf16 | 375.23 |
| SE9-16 | glc_encoder_bf16 | 968.14 |
| SE9-16 | glc_decoder_bf16 | 1740.53 |
| SE9-16 | glc_prior_init_bf16 | 16.29 |
| SE9-16 | glc_stage_1_bf16 | 15.94 |
| SE9-16 | glc_stage_2_bf16 | 15.93 |
| SE9-16 | glc_stage_3_bf16 | 15.95 |
| SE9-16 | glc_tail_bf16 | 1675.81 |

> **测试说明**：
> 1. `calculate time` 已折算为单次推理时间，测试输入为 512x768 对应的静态 shape；
> 2. SE9-16（BM1688）的 TPU 推理时间为 SE7-32 的 4.4~7.1 倍，与两芯片算力差距相符。

### 7.2 程序运行性能

参考 [Python例程](./python/README.md) 运行程序，查看统计的预处理、编码、码本查找、解码、后处理时间，已折算为单张图片。

| 测试平台 | 测试程序 | 测试模型 | preprocess_time | encoder_time | vq_time | decoder_time | inference_time | postprocess_time |
| -------- | -------- | -------- | --------------- | ------------ | ------- | ------------ | -------------- | ---------------- |
| SE7-32 | glc_eval.py | glc_fp32 | 42.54 | 977.61 | 57.40 | 1818.42 | 2853.42 | 121.58 |
| SE7-32 | glc_eval.py | glc_bf16 | 42.64 | 232.02 | 60.66 | 394.96 | 687.65 | 122.05 |
| SE9-16 | glc_eval.py | glc_bf16 | 55.83 | 980.37 | 2441.02 | 1754.37 | 5175.76 | 156.43 |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间，输入分辨率 512x768，测试集为 Kodak24 中尺寸匹配的 18 张，每张图循环 3 次取平均；
> 2. `glc_fp32` 指 `glc_encoder_fp32.bmodel` + `glc_decoder_fp32.bmodel`，`glc_bf16` 同理；
> 3. `inference_time` 为 `encoder_time + vq_time + decoder_time`，`vq_time` 是 CPU 上的码本最近邻查找；SE7-32 上约占端到端耗时的 9%，SE9-16 上因 CPU 较慢占约 47%；
> 4. 后处理包含反归一化、裁剪与 bpp 估算（高斯熵模型），**不含** PSNR/SSIM 指标计算——纯 numpy 的 SSIM 很慢，计入会淹没真实后处理开销；
> 5. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 6. BM1684X 的 FP32 吞吐显著低于 FP16/BF16，BF16 将端到端延迟压缩至约四分之一，而精度损失可忽略，因此**推荐使用 BF16**。

## 8. FAQ

通用问题（环境安装、模型导出、编译量化等）请先参考 [公共 FAQ](../../docs/FAQ.md)。
下面只列本例程特有的问题。

### Q1: 为什么不用 sophon-sail？

SE7 的 OTA 出厂镜像只带 `libsophon`（含 `libbmrt` / `libbmlib`），没有预装 `sophon-sail`，且它不在公共 PyPI 上（SoC 版 wheel 由 SDK 单独分发）。

`python/bmrt.py` 用 ctypes 直接调用 `libbmrt`，提供与 `sail.Engine` 等价的推理能力，只依赖 numpy，无需编译。设备端因此只需要 `numpy` 与 `pillow` 两个 pip 包。

### Q2: 加载 bmodel 时段错误（segmentation fault）？

`bm_device_mem_t` 在 aarch64 上是 **24 字节**（union 16 + flags 4 + size 4），`bm_tensor_t` 是 72 字节。若按 16 字节定义，`bm_malloc_device_byte` 会写越界并 segfault。

`python/bmrt.py` 在导入时会校验这些结构体的尺寸与 `bm_tensor_t.device_mem` 的偏移，不符会直接报错而不是崩溃。若将来 libsophon 升级导致布局变化，会在这里提前暴露。

### Q3: 为什么 FP16 模型推理出 NaN？

GLC 有两处激活幅值超出 fp16 上限 65504，都是模型固有属性：

- **AttnBlock 的 GroupNorm**：三个 AttnBlock 的归一化输入方差达 6.0e4 / 1.3e5 / 2.4e5，fp16 下 `E[x²]` 溢出成 Inf。这一处可以用「GroupNorm 输入除以 32、同时把 eps 除以 32²」严格等价地缓解（见 `tools/export_onnx.py` 的 `ScaledGroupNorm`，用 `F16_NORM_FIX=1` 开启）。
- **解码器 generator 的残差流**：`generator/blocks.22~25` 的残差累加值达 9.6e4 ~ 1.05e5。残差分支与主分支共享张量，无法用缩放修复，只能整层保持 FP32，而当前 TPU-MLIR 的 F16 编译没有可用的逐算子精度表。

**结论：本模型请用 BF16。** BF16 指数范围与 FP32 相同，上述两处都不会溢出。

### Q4: 怎么定位到是哪个算子先出 NaN 的？

`tools/export_probe.py` 会导出一个「每层都作为输出」的 ONNX，`tools/export_attn_probe.py` 进一步把 AttnBlock 内部逐算子展开。编译成 fp16 后在设备上跑一次，就能看到第一个全 NaN 的张量：

```bash
PYTHONPATH=tools/stubs python3 tools/export_probe.py --glc-src <GLC源码> \
    --ckpt models/GLC_image.pth.tar --image datasets/kodak24/kodim01.png --out onnx_probe
```

注意探测必须用**真实图像**——随机噪声会让 `GLC_Encoder` 本身发散出 NaN。

### Q5: 为什么跑不了竖构图的图？

bmodel 是静态 shape。Kodak24 有 6 张是 768x512，与 512x768 的模型不匹配，`glc_eval.py` 会跳过并在日志里列出。要支持其它分辨率需按该分辨率重新导出并编译。

### Q6: 精度不达标怎么办？

按顺序隔离：

1. 先在主机用 `tools/pytorch_baseline.py` 得到基线；
2. 用 FP32 bmodel 跑 `tools/eval_accuracy.py`，FP32 应逐位一致——若不一致说明 ONNX 导出或预处理有偏差；
3. 再看降精度版本。若 BF16 偏差大，检查输入值域是否与导出时一致（本模型是 [-1,1]，无 mean/std 归一化）。

### Q7: 导出 ONNX 时报 `No module named 'torchvision'`？

GLC 源码的 `layers.py` 里有 `from torchvision import models`，但该 import 从未被使用。若环境里没有匹配 torch 版本的 torchvision，可用 `tools/stubs/torchvision/` 里的轻量替身绕过：

```bash
PYTHONPATH=tools/stubs python3 tools/export_onnx.py
```

### Q8: 导出 ONNX 时报 `Output 1 of SplitBackward0 is a view and is being modified inplace`？

构造 `GLC_Image` 时必须用 `inplace=False`。默认的 `inplace=True` 会让 Split 的输出被原地修改，`torch.onnx.export` 会直接报这个错。

### Q9: INT8 校准为什么这么慢？

校准在 TPU-MLIR 的 MLIR CPU 解释器上执行，而 GLC 解码器单次前向约 1 TFLOP，解释器比 TPU 慢约三个数量级。实测解码器 1 个样本需 87 分钟激活采集，加 625 个算子的逐算子调优（慢算子约 26 s/it）；编码器 43 分钟才推进到 5%；而 TPU-MLIR 官方建议 100+ 样本，本环境不具备可行性。

另外，免校准的权重量化模式（`W4BF16` / `W8BF16` / `INT8BF16DYN`）虽然能编译通过，但产物与 BF16 **逐字节同尺寸、推理结果逐位相同**，在本模型上静默退化为 BF16，没有收益。

若后续要推进 INT8，建议：

1. 在专用机器上预留充足时间，并按官方建议把样本数提到 100+；
2. 用 `--search fast_search` 替代默认的逐算子阈值搜索，可显著缩短 `tune op` 阶段（`gen_int8bmodel_mlir.sh` 已支持 `SEARCH` 环境变量）；
3. 采用混合精度划分：**编码器保持 BF16**（其 `z` 输出要经过 16384 类码本的 argmin，对量化扰动最敏感），**解码器用 INT8**（占端到端耗时的大头）；
4. 评估时重点关注码本索引一致性：`python/glc_infer.py` 的 `codebook_lookup()` 会返回索引，可比对 INT8 与 FP32 的索引差异比例——这是本模型对量化最敏感的指标。

### Q10: 用 `glc_decode.py` 解另一台机器产出的码流，为什么要两端 `--precision` 一致？

因为熵模型的 CDF 行是按 64 档 scale 表查出来的，而 BF16 与 FP32 算出的 scale 有微小差异，
实测约 **1.2%** 的 scale 会落到不同档位。rANS 是算术编码，码流与 CDF 一旦对不上，解码
**不会报错**，而是一路解出错误符号，最终产出退化图像（整幅常量或 NaN）。

用 `--precision bf16` 编码的码流，还原端必须也用 `--precision bf16`。

`glc_decode.py` 内置了自检：解出的 `y_hat` 若含 NaN/Inf、量级异常，或重建图退化成常量，
会直接报错并提示检查 `--precision`，不会静默交出坏图。实测混用精度时：

```
$ python3 python/glc_decode.py --precision bf16 --input fp32_encoder.glc
ValueError: 解码出的 y_hat 含 NaN/Inf：码流与当前 --precision 模型不匹配
```

注意：这不是本实现的缺陷，而是算术编码的固有性质——**任何**两端分离的熵编码方案都要求
两端熵模型完全一致（同样的 scale_table、同样的量化精度）。

**跨芯片是同一个道理，但成因不同。** SE7-32（BM1684X）与 SE9-16（BM1688）各自编出的
BF16 码流**互不可解**——两芯片的 BF16 先验输出并不逐位相同，实测另一组数字：**1.27%**
的 scale 落到不同 CDF 档位（注意与上面同芯片跨精度的 1.2% 是两个独立实测值，不要混用），
rANS 随即失步。实测同一张图**都走 BF16** 时，SE7 编出 1416 字节、SE9 编出 1412 字节，
正是两个平台上量化结果不同的直接体现（改用 `mixed` 后两端都是 1416 字节）。

**解决办法：跨芯片部署用 `--precision mixed`（推荐）。** 跨芯片不必两端全程 FP32
——不是所有子图都影响互解，只需让「决定 CDF 选择与 y_hat 重建」的子图逐位一致即可：

| 子图 | mixed 下的精度 | 原因 |
|------|---------------|------|
| `glc_prior_init` | FP32 | 输出 scales（选 CDF 行）、means（重建 y_hat）、common、q_dec |
| `glc_stage_1..3` | FP32 | 同上；本段重建的 y_hat 还是下一段的输入，误差会累积 |
| `glc_encoder` | BF16 | 只决定写进码流的符号，由熵编码保证；不影响可解性 |
| `glc_tail` | BF16 | 只决定最终重建图，不影响可解性 |

实测（SE7-32 BM1684X ↔ SE9-16 BM1688 双向）：

```
SE7 编码 -> SE9 解码   两端 y_hat sha256 相同 (739641608e723fd7)  跨芯片互解正常
SE9 编码 -> SE7 解码   两端 y_hat sha256 相同 (55ffdfe3d57e8b79)  跨芯片互解正常
SE7 BF16 编码 -> SE9 BF16 解码   ValueError: y_hat 含 NaN/Inf（旧行为，互不可解）
```

`prior_init` / `stage_1..3` 用 FP32 后，两芯片的这四个子图输出**逐位相同**；
`encoder` / `tail` 留 BF16 不影响互解，而它们恰是耗时大头，因此 mixed 只比纯 BF16
慢约 10%（SE7 编码 1312 → 1438 ms），重建质量不变（PSNR 18.98 → 18.95 dB）。
**需要跨芯片部署时两端都用 `--precision mixed`**；同芯片内闭环继续用 `bf16`。

若确实需要全程 FP32（例如要逐张量对账 PyTorch），两端都用 `--precision fp32`
也可消除差异，代价是明显更慢（SE9 编码 5725 → 9579 ms）。

> `mixed` / `fp32` 用到的 FP32 子图已随下载包下发，无需自行编译。
> 若需改 shape 或自建产物，见 [GLC 真熵编码 §5.2](./docs/GLC_Entropy_Coding.md)。

> **编译产物必须逐张量验证**：本环境实测遇到过 BM1688 的 `glc_stage_2_fp32.bmodel`
> 两个输出张量别名到同一块内存（`scales` 与 `means` 取到相同值），此时该子图即使
> FP32 也会跨芯片不一致。重编一次即恢复正常。TPU-MLIR 在本模型上不是逐位可复现的，
> 自编译后务必跑 `tools/tests/test_bmodel_accuracy.py` 或
> `tools/subgraph_probe.py` 对账。

与客户 GPU 端互操作时，需要对齐的是**参数**，实测**每个参数都会改变码流**：
`SCALE_MIN=0.11`、`SCALE_MAX=256.0`、`SCALE_LEVELS=64`、`tail_mass=1e-9`。
而 compressai 版本在 1.1.9 ~ 1.2.8 范围内实测**不影响**码流（五个版本 CDF 与码流
md5 完全相同）。验证见 `tools/tests/test_param_alignment.py`。
