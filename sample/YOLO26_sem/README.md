# YOLO26_sem

## 目录

- [YOLO26\_sem](#yolo26_sem)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
  - [4. 模型编译](#4-模型编译)
    - [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    - [4.2 生成FP16 BModel](#42-生成fp16-bmodel)
    - [4.3 生成INT8 BModel](#43-生成int8-bmodel)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)
    - [6.1 测试方法](#61-测试方法)
    - [6.2 测试结果](#62-测试结果)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt\_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
  - [8. FAQ](#8-faq)

## 1. 简介
YOLO26_sem 将 [YOLO26 官方开源仓库](https://github.com/ultralytics/ultralytics)（v8.4.105）的**语义分割**模型 `yolo26s-sem.pt` 移植到 SOPHON BM1684X/BM1688/CV186X 上进行推理测试。

注意：YOLO26 存在两种分割任务，二者模型与输出不同，请勿混淆：

| 任务 | Ultralytics 权重 | 输出 | 后处理 |
| ---- | ---------------- | ---- | ------ |
| **实例分割**（本仓库 `YOLO26_seg` 例程） | `yolo26s-seg.pt` | 检测框 + 实例 mask | NMS + mask 采样 |
| **语义分割**（本例程 `YOLO26_sem`） | `yolo26s-sem.pt` | 逐像素类别图 `[1, 1, 1024, 2048]` int32 | 去 letterbox padding + 缩放回原图 |

本例程把 logits 的 8 倍 bilinear 上采样与 argmax **一并烘焙进模型**，使模型直接输出逐像素类别图（Cityscapes 19 类 trainId，`[1, 1, 1024, 2048]` int32），推理端后处理只剩去 letterbox padding 与缩放回原图（先在模型内对 logits 做 8x bilinear 上采样再 argmax，与 ultralytics 官方后处理顺序一致，保证精度对齐），与仓库 [segformer](./../segformer) 例程的语义分割范式保持一致。

## 2. 特性
* 支持BM1688(SoC)、BM1684X(x86 PCIe、SoC)、CV186X(SoC)
* 支持 FP32、FP16、INT8 模型编译和推理
* 支持基于 BMCV 预处理的 C++ 推理
* 支持基于 OpenCV 和 BMCV 预处理的 Python 推理
* 支持语义分割（bmodel 直接输出逐像素类别图），无需 NMS
* 支持图片和视频测试

## 3. 数据准备与模型编译
### 3.1 数据准备
本例程在 `scripts` 目录下提供了相关模型和数据的下载脚本 `download.sh`。**如果您希望自己准备模型和数据集，可以跳过本小节，参考[4. 模型编译](#4-模型编译)进行模型转换。**

```bash
# 安装 unzip，若已安装请跳过，非 ubuntu 系统视情况使用 yum 或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh --all
```

`download.sh` 默认只下载 `datasets`，`models` 可以通过指定参数分平台下载，参数如下：
```bash
--all      # 下载所有模型
--BM1684X  # 下载 BM1684X 的 bmodel
--BM1688   # 下载 BM1688 的 bmodel
--CV186X   # 下载 CV186X 的 bmodel
--onnx     # 下载 onnx
```

下载的模型包括：
```
./models
├── BM1684X  # 在 BM1684X 上运行的模型
│   ├── yolo26s_fp32_1b.bmodel
│   ├── yolo26s_fp16_1b.bmodel
│   └── yolo26s_int8_1b.bmodel
├── BM1688   # 在 BM1688 上运行的模型
│   ├── yolo26s_fp32_1b.bmodel
│   ├── yolo26s_fp16_1b.bmodel
│   ├── yolo26s_int8_1b.bmodel
│   └── yolo26s_int8_1b_2core.bmodel
├── CV186X   # 在 CV186X 上运行的模型
│   ├── yolo26s_fp32_1b.bmodel
│   ├── yolo26s_fp16_1b.bmodel
│   └── yolo26s_int8_1b.bmodel
└── onnx
    └── yolo26s-sem.onnx   # 导出的 onnx 模型（输出逐像素类别图）
```
下载的数据包括：
```
./datasets
├── test              # 测试图片
├── cali              # 量化校准图片（Cityscapes 抽样）
├── cityscapes        # Cityscapes 验证集（val 500 张，含 gtFine 标签与 val.txt，用于精度评测）
└── cityscapes_video.avi  # 测试视频
```

## 4. 模型编译
**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成 BModel 才能在 SOPHON TPU 上运行，源模型在编译前要导出成 onnx 模型，具体可参考 [YOLO26 模型导出](./docs/YOLO26_sem_Export_Guide.md)。同时，您需要准备用于测试的数据集；如果量化模型，还要准备用于量化的数据集。

建议使用 TPU-MLIR 编译 BModel，模型编译前需要安装 TPU-MLIR，具体可参考 [TPU-MLIR 环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在 TPU-MLIR 环境中进入例程目录，并使用本例程提供的脚本将 onnx 模型编译为 BModel。脚本中命令的详细说明可参考《TPU-MLIR 开发手册》（请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的 SDK 中获取）。

### 4.1 生成FP32 BModel

本例程在 `scripts` 目录下提供了 TPU-MLIR 编译 FP32 BModel 的脚本，请注意修改 `gen_fp32bmodel_mlir.sh` 中的 onnx 模型路径、生成模型目录和输入大小 shapes 等参数，并在执行时指定 BModel 运行的目标平台（**支持 BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x   # bm1688 / cv186x
```

执行上述命令会在 `models/BM1684X`（或 `models/BM1688`、`models/CV186X`）文件夹下生成转换好的 FP32 BModel。

### 4.2 生成FP16 BModel

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x   # bm1688 / cv186x
```

执行上述命令会在 `models/BM1684X/`（或 `models/BM1688/`、`models/CV186X/`）文件夹下生成转换好的 FP16 BModel。

### 4.3 生成INT8 BModel

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684x   # bm1688 / cv186x
```

上述脚本会在 `models/BM1684X`（或 `models/BM1688`、`models/CV186X`）文件夹下生成转换好的 INT8 BModel（BM1688 平台会额外生成 `yolo26s_int8_1b_2core.bmodel`）。量化采用 `part_asymmetric` + `percentile9999` 校准。如果您需要量化自己微调过的模型，可以参考[量化指南](./docs/YOLO26_sem_Calibration_Guide.md)中的方法。

> **提示**：本例程把 bilinear 上采样 + argmax 烘焙进模型（模型直接输出逐像素类别图），argmax 为单调算子、无需拆分敏感层/维护 qtable，FP16/INT8 编译可直接使用上述脚本完成。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考 [C++例程](cpp/README.md) 或 [Python例程](python/README.md) 推理要测试的数据集，生成逐像素类别图（灰度 segmap，保存在 `results/segmaps/` 下）。

> **说明**：Cityscapes 验证图片按城市存放在 `datasets/cityscapes/leftImg8bit/val/{city}/` 子目录中（共 500 张，`datasets/cityscapes/val.txt` 为评价图片列表），Python 与 C++ 例程的图片输入**均递归扫描子目录**，可直接指向 `leftImg8bit/val` 目录进行评测。

然后，使用 `tools` 目录下的 `eval_cityscapes.py` 脚本，将生成的 segmap 与 Cityscapes gtFine 标签图（labelIds）进行比对，计算 mIoU 与像素准确率，命令如下：

```bash
# 以下命令在例程根目录下执行（请根据实际情况修改 pred_dir / gt_dir 路径）
python3 tools/eval_cityscapes.py \
    --pred_dir results/segmaps \
    --gt_dir datasets/cityscapes/gtFine/val \
    --img_suffix _leftImg8bit.png \
    --gt_suffix _gtFine_labelIds.png
```

> **说明**：`eval_cityscapes.py` 会自动把 gtFine 的 labelIds 映射为训练用的 trainIds（19 类），并与预测 segmap 逐像素计算 IoU。gtFine 标签按城市分子目录存放，脚本会递归检索，无需提前扁平化。

### 6.2 测试结果
在 Cityscapes 验证集（val，500 张）上，精度测试结果如下：

| 测试平台 | 测试程序          | 测试模型                | mIoU    | Pixel Acc |
| -------- | ----------------- | ----------------------- | ------- | --------- |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |  80.81 |    96.33 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |  80.81 |    96.33 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |  80.81 |    96.33 |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |  80.56 |    96.29 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |  80.56 |    96.29 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |  80.56 |    96.29 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |  80.81 |    96.33 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |  80.80 |    96.33 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |  80.80 |    96.33 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |  80.56 |    96.29 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |  80.56 |    96.30 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |  80.56 |    96.30 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_int8_1b_2core.bmodel |  80.56 |    96.29 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_int8_1b_2core.bmodel |  80.56 |    96.30 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b_2core.bmodel |  80.56 |    96.30 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |  80.80 |    96.33 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |  80.81 |    96.33 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |  80.80 |    96.33 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |  80.80 |    96.33 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |  80.56 |    96.29 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |  80.56 |    96.30 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |  80.56 |    96.30 |

> **测试说明**：
> 1. 由于 SDK 版本之间可能存在差异，实际运行结果与本表有 <0.1 的精度误差是正常的；
> 2. C++（bmcv）与 Python 在部分图片上存在微小差异，原因是 bmcv 硬件 resize 与 OpenCV 软件插值的实现差异，属正常现象；
> 3. 在搭载相同 TPU 和 SOPHONSDK 的平台上，相同程序的精度一致，SE7 系列对应 BM1684X，SE9-16 对应 BM1688，SE9-8 对应 CV186X。

## 7. 性能测试
### 7.1 bmrt_test
使用 bmrt_test 测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的 bmodel 路径和 devid 参数
bmrt_test --bmodel models/BM1684X/yolo26s_fp32_1b.bmodel --devid 0
```
测试结果中的 `calculate time` 就是模型推理的时间（本例程 batch size 为 1，无需再除以 batch size）。测试各个模型的理论推理时间，结果如下：

| 测试平台 | 测试模型                        | calculate time(ms) |
| -------- | ------------------------------- | ------------------ |
| SE7-32   | BM1684X/yolo26s_fp32_1b.bmodel  | 134.93 |
| SE7-32   | BM1684X/yolo26s_fp16_1b.bmodel  | 45.96  |
| SE7-32   | BM1684X/yolo26s_int8_1b.bmodel  | 32.72  |
| SE9-16   | BM1688/yolo26s_fp32_1b.bmodel   | 588.74 |
| SE9-16   | BM1688/yolo26s_fp16_1b.bmodel   | 168.10 |
| SE9-16   | BM1688/yolo26s_int8_1b.bmodel   | 63.21  |
| SE9-16   | BM1688/yolo26s_int8_1b_2core.bmodel | 51.88 |
| SE9-8    | CV186X/yolo26s_fp32_1b.bmodel  | 585.05 |
| SE9-8    | CV186X/yolo26s_fp16_1b.bmodel  | 167.89 |
| SE9-8    | CV186X/yolo26s_int8_1b.bmodel  | 62.67  |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time` 已折算为平均每张图片的推理时间。

### 7.2 程序运行性能
参考 [C++例程](cpp/README.md) 或 [Python例程](python/README.md) 运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++ 例程打印的时间已经折算为单张图片的处理时间。

在 Cityscapes 验证集（val，500 张，原图 2048×1024 等比缩放至 1024×2048 输入，与官方 ultralytics 评测分辨率一致）上，各平台各程序的性能测试结果如下（单位均为毫秒 ms）：

| 测试平台 | 测试程序           | 测试模型                 | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | ------------------ | ------------------------ | ----------- | --------------- | -------------- | ---------------- |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |    95.00 |   126.51 |   156.34 |     5.70 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |   109.06 |    10.18 |   141.99 |     5.71 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |   107.77 |     7.00 |   137.37 |     5.54 |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |    92.00 |   122.44 |    67.32 |     5.67 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |   106.37 |    10.85 |    52.98 |     5.72 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |   109.93 |     6.92 |    48.38 |     5.55 |
| SE7-32    | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |    91.52 |   122.39 |    53.86 |     5.67 |
| SE7-32    | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |   106.53 |    10.18 |    39.66 |     5.71 |
| SE7-32    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |   105.04 |     6.99 |    35.11 |     5.53 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |   118.43 |   157.77 |   615.58 |     7.10 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |   121.51 |    19.82 |   597.48 |     7.35 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |   118.61 |    11.17 |   592.14 |     6.89 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |   118.28 |   162.63 |   194.83 |     7.02 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |   121.00 |    19.82 |   176.69 |     7.32 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |   118.50 |    11.16 |   171.39 |     6.86 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |   118.23 |   162.36 |    89.80 |     7.02 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |   120.91 |    19.82 |    72.35 |     7.03 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |   118.53 |    11.16 |    66.35 |     6.85 |
| SE9-16    | yolo26_sem_opencv.py   | yolo26s_int8_1b_2core.bmodel |   118.41 |   157.82 |    78.84 |     7.04 |
| SE9-16    | yolo26_sem_bmcv.py     | yolo26s_int8_1b_2core.bmodel |   120.95 |    19.85 |    61.03 |     7.03 |
| SE9-16    | yolo26_sem_bmcv.soc    | yolo26s_int8_1b_2core.bmodel |   118.56 |    11.15 |    55.10 |     6.88 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_fp32_1b.bmodel     |   144.21 |   203.58 |   612.18 |     7.04 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_fp32_1b.bmodel     |   148.90 |    19.87 |   594.51 |     7.08 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_fp32_1b.bmodel     |   126.09 |    11.17 |   588.50 |     6.91 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_fp16_1b.bmodel     |   145.38 |   187.29 |   194.86 |     7.04 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_fp16_1b.bmodel     |   150.42 |    19.87 |   177.22 |     7.05 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_fp16_1b.bmodel     |   129.40 |    11.16 |   171.29 |     6.91 |
| SE9-8     | yolo26_sem_opencv.py   | yolo26s_int8_1b.bmodel     |   145.90 |   196.64 |    89.70 |     7.03 |
| SE9-8     | yolo26_sem_bmcv.py     | yolo26s_int8_1b.bmodel     |   150.42 |    19.88 |    72.10 |     7.08 |
| SE9-8     | yolo26_sem_bmcv.soc    | yolo26s_int8_1b.bmodel     |   133.19 |    11.17 |    66.10 |     6.90 |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. 图片分辨率对解码时间影响较大（本例使用 Cityscapes 原图 2048×1024），推理时间为逐张图片的模型计算时间，与输入分辨率（固定 1024×2048）无关；
> 4. 性能差异主要来自 TPU 架构：BM1684X 算力强于 BM1688，BM1688(SE9-16) 与 CV186X(SE9-8) 同为 SGTPUV8 架构、性能接近；
> 5. 本例程把 bilinear 上采样 + argmax 烘焙进模型（模型直接输出类别图），因此 `postprocess_time` 从原来的 ~230–370ms 大幅降至 ~5–9ms，而 `inference_time` 相应略增（推理量增加了 8 倍上采样 + argmax，约 20–40ms）。

## 8. FAQ
请参考 [FAQ](../../docs/FAQ.md) 查看一些常见的问题与解答。