# MoGe

## 目录

- [MoGe](#moge)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 模型编译](#32-模型编译)
  - [4. 例程测试](#4-例程测试)
  - [5. 精度测试](#5-精度测试)
    - [5.1 测试方法](#51-测试方法)
    - [5.2 测试结果](#52-测试结果)
  - [6. 性能测试](#6-性能测试)
    - [6.1 bmrt\_test](#61-bmrt_test)
    - [6.2 程序运行性能](#62-程序运行性能)
  - [7. FAQ](#7-faq)

## 1. 简介

MoGe（Monocular Geometry Estimation）是微软开源的**单目几何估计**模型，能从单张 RGB 图像同时估计**度量深度图（metric depth）**、**点云（point map）**、**法向图（surface normal）**以及相机内参（焦距 / 视场角）。本例程适配 [MoGe 官方开源仓库](https://github.com/microsoft/MoGe)中的 **MoGe-2 (ViT-L/14, 331M)** 模型，支持在 SOPHON BM1684X / BM1688 / CV186X 上进行推理测试。

MoGe-2 由 DINOv2 ViT-L/14 编码器、ConvStack 颈部与四个并行的回归头构成，输出（均为 fp32）为：

| 输出 | 形状 | 说明 |
| ---- | ---- | ---- |
| `points` | (1, H, W, 3) | 相机坐标系下的点图 |
| `normal` | (1, H, W, 3) | 单位法向图 |
| `mask` | (1, H, W) | 有效区域，sigmoid 后取阈值 |
| `metric_scale` | (1,) | 度量尺度因子 |

模型的 ImageNet 均值/方差归一化已内置于 DINOv2 编码器中，因此输入即为 **RGB float [0, 1]**，无需任何外部归一化；输出经后处理（焦点/位移恢复、投影）得到度量深度、内参、点云与法向图。

## 2. 特性

### 2.1 目录结构说明

```bash
├── cpp                   # 存放C++例程及其README
│   └── moge_bmcv         # C++例程
├── docs                  # 存放本例程专用文档，如ONNX导出等
├── pics                  # 存放README等说明文档中用到的图片
├── python                # 存放Python例程及其README
│   ├── moge_bmcv.py      # Python例程
│   └── postprocess_numpy.py  # 后处理（焦点/位移恢复、投影、着色）
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译、数据下载、自动测试等shell脚本
└── tools                 # 存放精度测试、性能比对等python脚本
```

### 2.2 SDK特性

* 支持 BM1688/CV186X(SoC) 和 BM1684X(x86 PCIe、SoC、riscv PCIe)
* 支持 FP32、FP16 模型编译和推理；**实际部署默认 FP16**，FP32 用作精度基线
* 支持 C++、Python 推理
* 支持单张图片与图片目录测试
* 输入形状固定为 518×518（MoGe 官方静态导出，`num_tokens=1800`），与官方 `docs/onnx.md` 导出口径一致

## 3. 数据准备与模型编译

### 3.1 数据准备

本例程在 `scripts` 目录下提供了相关模型和数据的下载脚本 `download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
./scripts/download.sh --all
```

`download.sh` 默认只下载测试图片，`models` 和数据集可通过指定参数分项下载：

```bash
--all     # 下载所有（含测试图片、ONNX、评测数据集与各平台 bmodel）
--onnx    # 下载官方 ONNX 模型
--eval    # 下载官方评测数据集 NYUv2
--BM1684X # 下载 BM1684X 平台的 bmodel
--BM1688  # 下载 BM1688 平台的 bmodel
--CV186X  # 下载 CV186X 平台的 bmodel
```

下载的模型包括：

```bash
models/
├── BM1684X    # 在BM1684X上运行的模型
│   ├── moge-2-vitl-normal_fp32_1b.bmodel
│   └── moge-2-vitl-normal_fp16_1b.bmodel
├── BM1688     # 在BM1688上运行的模型
│   ├── moge-2-vitl-normal_fp32_1b.bmodel
│   ├── moge-2-vitl-normal_fp16_1b.bmodel
│   └── moge-2-vitl-normal_fp16_1b_2core.bmodel
├── CV186X     # 在CV186X上运行的模型
│   ├── moge-2-vitl-normal_fp32_1b.bmodel
│   └── moge-2-vitl-normal_fp16_1b.bmodel
└── onnx
    └── moge-2-vitl-normal.onnx
```

下载的数据包括：

```bash
./datasets
├── test            # 测试图片（MoGe 官方 example_images）
└── NYUv2           # 官方评测数据集（Ruicheng/monocular-geometry-evaluation）
    ├── .index.txt  # 样本子目录名列表
    └── <00123>/    # 每个样本：image.jpg（原图）+ depth.png（16bit 对数编码度量深度）+ meta.json（相机内参）
```

### 3.2 模型编译

**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成 BModel 才能在 SOPHON TPU 上运行。MoGe-2 需先将 PyTorch 权重导出为**静态 518×518** 的 ONNX（详细步骤见[模型导出](./docs/MoGe_Export_Guide.md)），再由 TPU-MLIR 编译成 BModel。

建议使用 TPU-MLIR 编译 BModel，模型编译前需要安装 TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在 TPU-MLIR 环境中进入例程目录，使用本例程提供的脚本将 onnx 模型编译为 BModel：

- 生成 FP32 BModel

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1688   # bm1684x/bm1688/cv186x
```

执行上述命令会在 `models/BM1688` 等文件夹下生成转换好的 FP32 BModel。

- 生成 FP16 BModel

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1688
```

执行上述命令会在 `models/BM1688` 等文件夹下生成转换好的 FP16 BModel（BM1688 额外生成 2core 版本）。

## 4. 例程测试

- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

以测试图片 `01_HouseIndoor.jpg` 为例，输入与输出（度量深度着色图、法向着色图）如下：

| 输入图片 | 深度图（度量深度着色） | 法向图 |
| -------- | ---------------------- | ------ |
| ![input](./pics/input.jpg) | ![depth](./pics/depth.png) | ![normal](./pics/normal.png) |

## 5. 精度测试

### 5.1 测试方法

本例程精度测试使用 MoGe 官方评测数据集的 **NYUv2** 配置（`data/eval/NYUv2`，来自 HuggingFace `Ruicheng/monocular-geometry-evaluation`），并采用官方 `--mg moge2` 的 `depth_metric` 指标口径：对**度量深度**直接计算

$$\text{rel} = \text{mean}\left(\frac{|pred - gt|}{gt + \epsilon}\right),\qquad \delta_1 = \text{mean}\left(\max\left(\frac{gt}{pred},\frac{pred}{gt}\right) < 1.25\right).$$

rel 越低越好，δ1 越高越好。

由于官方 dataloader 的相机重投影 warp 依赖 torch + PIL + utils3d（仅 x86 可跑），而 bmodel 部署在板端（numpy + sail 环境），本例程采用一套**统一口径（square-panel, 无 warp）**以保证各后端可比：对每张原图按例程部署路径直接缩放到 518×518 → 推理得到 518×518 度量深度与 mask → 双线性缩放回 GT 分辨率 → 与原始 `depth.png`（近/远元数据解码为米）做掩蔽比较。

```bash
# bmodel（板端，自动复用例程 python/moge_bmcv.py 的推理与后处理）
python3 tools/eval_moge.py --backend bmodel --dataset datasets/NYUv2 \
    --bmodel models/BM1688/moge-2-vitl-normal_fp16_1b.bmodel --dev_id 0

# onnx fp32 参考（x86，onnxruntime 跑官方静态 518×518 ONNX，作为基线）
python3 tools/eval_moge.py --backend onnx --dataset datasets/NYUv2 \
    --onnx models/onnx/moge-2-vitl-normal.onnx
```

脚本会打印一行 `[NYUv2] depth_metric rel=... delta1=...` 供 `tools/compare_acc.py` 解析比对。

> **与官方评测协议的差异说明：** 官方 NYUv2 协议先把原图按相机内参 **warp 到目标视图**（LANCZOS 缩放 + 单应 remap），在 640×480 目标分辨率上与**同样 warp 过的** GT 深度比较，且 `depth_metric` 只按 GT 有效掩码（含深度分位裁剪）过滤、不再叠加模型预测 mask。本例程 bmodel/onnx 为固定 518×518 静态输入、板端无 torch/utils3d，故采用**统一 square-panel、无 warp 口径**：直接缩放 518×518 → 推理 → 双线性缩回 GT 分辨率 → 与原始 `depth.png` 掩蔽比较（有效掩码额外叠加了模型预测 mask）。因此绝对数值与官方报告不可直接比较，差异仅来自四点——① 输入分辨率 518² vs 640×480；② 方块面板的**非方形图像畸变**（aspect=1 直接 squish，不改输入图宽高比）；③ 不做目标视角 warp；④ `depth_metric` 掩码额外叠加 pred mask。四点对 bmodel 与 fp32 onnx 参考的影响完全一致，因此**相对对比完全公平**，仅用于本例程跨后端/跨精度的一致性校验。

### 5.2 测试结果

在 NYUv2 官方评测集上，depth_metric 精度测试结果如下：

|   测试平台    |      测试程序     |              测试模型              |  rel  | delta1 |
| ------------ | ---------------- | ---------------------------------- | ----- | ------ |
|  x86(参考)   |  eval_moge.py    |   moge-2-vitl-normal.onnx(fp32)     | 0.1027 | 0.9170 |
|   SE9-16     |  moge_bmcv.py    |   moge-2-vitl-normal_fp32_1b.bmodel | 0.1034 | 0.9166 |
|   SE9-16     |  moge_bmcv.py    |   moge-2-vitl-normal_fp16_1b.bmodel | 0.1041 | 0.9146 |
|   SE7        |  moge_bmcv.py    |   moge-2-vitl-normal_fp32_1b.bmodel | 0.1048 | 0.9110 |
|   SE7        |  moge_bmcv.py    |   moge-2-vitl-normal_fp16_1b.bmodel | 0.1056 | 0.9089 |
|   SE9-8      |  moge_bmcv.py    |   moge-2-vitl-normal_fp16_1b.bmodel | 0.1041 | 0.9146 |

> **测试说明：**
> 1. 由于 SDK 版本之间可能存在差异，实际运行结果与本表有 <0.01 的精度误差是正常的；
> 2. 在搭载相同 TPU 与 SOPHONSDK 的 PCIe 或 SoC 平台上，相同程序的精度一致；SE5 系列对应 BM1684，SE7 系列对应 BM1684X，SE13 系列对应 CV84X6，SE9 系列中 SE9-16 对应 BM1688、SE9-8 对应 CV186X；
> 3. 评测采用统一口径（见 5.1），不作为与官方论文数值直接比较的依据，仅用于本例程跨后端/跨精度的一致性校验。
> 4. **SE9-8 (CV186X) 说明：** SE9-8 与 SE9-16 共用同一 TPU1686/BM1688 后端，FP16 精度与 SE9-16 完全一致（见上表）。SE9-8 板载内存仅 1.6GB，无法加载 1.52GB 的 FP32 bmodel（`bm_alloc_gmem failed`），故上表无 SE9-8 的 FP32 行；CV186X 为单核设备，亦无 2core 变体。
> 5. **SE7 (BM1684X) 说明：** BM1684X 与 BM1688/CV186X 是**真正不同的芯片后端**（不同的 TPU 架构与浮点实现），故 SE7 的 FP32/FP16 精度与 SE9-16 有一致的小幅偏移（rel 高 ~0.0015、δ1 低 ~0.006），均落在注 1 所述 <0.01 误差范围内，属不同芯片的数值实现差异、非 bug。这与 SE9-8 复用 BM1688 后端而逐位一致的情况不同。

## 6. 性能测试

### 6.1 bmrt_test

使用 bmrt_test 测试模型的理论性能：

```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数（输出中 calculate time 单位是秒）
bmrt_test --bmodel models/BM1688/moge-2-vitl-normal_fp32_1b.bmodel
```

测试结果中的 `calculate time` 即为模型纯推理时间（单 batch，已换算为 ms），测试各模型的理论推理时间如下：

|   测试平台   |                   测试模型                   | calculate time(ms) |
| ----------- | ------------------------------------------- | ------------------ |
|   SE9-16    | BM1688/moge-2-vitl-normal_fp32_1b.bmodel    |       11577.80     |
|   SE9-16    | BM1688/moge-2-vitl-normal_fp16_1b.bmodel    |        1964.28     |
|   SE9-16    | BM1688/moge-2-vitl-normal_fp16_1b_2core.bmodel |     1453.44     |
|   SE7       | BM1684X/moge-2-vitl-normal_fp32_1b.bmodel   |        3461.12     |
|   SE7       | BM1684X/moge-2-vitl-normal_fp16_1b.bmodel   |         557.37     |
|   SE9-8     | CV186X/moge-2-vitl-normal_fp16_1b.bmodel    |        1937.20     |

> **测试说明：** MoGe-2 为 ViT-L 大模型（331M 参数，518² 输入约 1370 个 token），BM1688 上 FP32 推理时间较长，实际部署推荐 FP16（速度优先）。SE9-8 (CV186X) 与 SE9-16 同后端，FP16 的 calculate time 略快（1937.20ms vs 1964.28ms）；SE9-8 的 1.6GB 内存无法加载 1.52GB 的 FP32 bmodel（OOM），CV186X 为单核无 2core。SE7 (BM1684X) 为不同架构芯片，整体约快 3.5×（FP16 557.37ms vs BM1688 1964.28ms；FP32 3461.12ms vs 11577.80ms）。

### 6.2 程序运行性能

参考[Python例程](python/README.md)与[C++例程](cpp/README.md)运行图片目录，统计 decode / preprocess / inference / postprocess 四段平均耗时（`tools/compare_statis.py` 自动解析日志）。各部门耗时以实测回填，示例：

|     测试平台   |      测试程序      |                测试模型                |decode_time|preprocess_time|inference_time|postprocess_time|
| ------------- | ------------------ | -------------------------------------- | --------- | ------------- | ------------ | -------------- |
|    SE9-16     |  moge_bmcv.py      |  moge-2-vitl-normal_fp16_1b.bmodel     | 15.31 | 15.23 | 1975.00 | 124.09 |
|    SE9-16     |  moge_bmcv.soc     |  moge-2-vitl-normal_fp16_1b.bmodel     | 14.85 | 14.34 | 1964.94 |  54.00 |
|    SE9-8      |  moge_bmcv.py      |  moge-2-vitl-normal_fp16_1b.bmodel     | 10.40 | 14.76 | 1947.56 | 109.70 |
|    SE9-8      |  moge_bmcv.soc     |  moge-2-vitl-normal_fp16_1b.bmodel     | 10.71 | 14.58 | 1938.64 |  53.88 |
|    SE7        |  moge_bmcv.py      |  moge-2-vitl-normal_fp32_1b.bmodel     |  7.24 |  8.43 | 3466.87 |  87.32 |
|    SE7        |  moge_bmcv.soc     |  moge-2-vitl-normal_fp32_1b.bmodel     |  7.36 |  9.58 | 3461.66 |  42.20 |
|    SE7        |  moge_bmcv.py      |  moge-2-vitl-normal_fp16_1b.bmodel     |  7.28 |  8.54 |  566.71 |  91.48 |
|    SE7        |  moge_bmcv.soc     |  moge-2-vitl-normal_fp16_1b.bmodel     |  7.47 |  9.94 |  557.20 |  42.27 |

## 7. FAQ

- **模型输入是否能动态变分辨率？** 不能。MoGe 官方 ONNX 为静态 518×518（`num_tokens=1800`）导出，例程将任意输入直接缩放到 518×518 进行推理。
- **输出是什么坐标系、用什么内参？** 后处理将 center-shift 点图按 pixel-center 定义投影，返回 `intrinsics`（fx, fy, cx=cy=0.5）、`depth`、`points`、`normal`，均为相机坐标系下的度量值。
- **为什么测试图片和 official 差异大？** 请优先确认图片是否被整体 squish 到 518×518；MoGe-2 对宽高比敏感，非方形输入建议预先裁剪。