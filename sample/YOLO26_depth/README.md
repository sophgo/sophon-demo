# YOLO26_depth

## 目录

- [YOLO26\_depth](#yolo26_depth)
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
YOLO26_depth 将 [YOLO26 官方开源仓库](https://github.com/ultralytics/ultralytics)（v8.4.105）的**单目深度估计（depth estimation）**模型 `yolo26s-depth.pt` 移植到 SOPHON BM1684X/BM1688/CV186X 上进行推理测试。

模型输入为 768×768 的 RGB 图片（官方默认 `imgsz=768`），输出为 `[1, 1, 768, 768]` 的 float32 **度量深度图（绝对尺度，单位米）**，与官方 Depth head 的导出口径一致：log 深度经过 `cal_a/cal_b` 标定与 `exp` 还原、4× bilinear 上采样，全部**烘焙进模型**，推理端后处理只需取出深度图、按需伪彩色可视化，无需任何阈值/NMS。

## 2. 特性
* 支持 BM1688(SoC)、BM1684X(SoC)、CV186X(SoC)
* 支持 FP32、FP16、INT8 模型编译和推理
* 支持基于 BMCV 预处理的 C++ 推理
* 支持基于 OpenCV 和 BMCV 预处理的 Python 推理
* bmodel 直接输出绝对尺度深度图（单位米），无需 NMS/阈值/裁边
* 精度评测复刻 ultralytics 官方 `DepthMetrics` 口径
* 支持图片和视频测试

## 3. 数据准备与模型编译
### 3.1 数据准备
本例程在 `scripts` 目录下提供了相关模型和数据的下载脚本 `download.sh`。**如果您希望自己准备模型和数据集，可以跳过本小节，参考[4. 模型编译](#4-模型编译)进行模型转换。**

```bash
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
│   ├── yolo26s_depth_fp32_1b.bmodel
│   ├── yolo26s_depth_fp16_1b.bmodel
│   └── yolo26s_depth_int8_1b.bmodel
├── BM1688   # 在 BM1688 上运行的模型
│   ├── yolo26s_depth_fp32_1b.bmodel
│   ├── yolo26s_depth_fp16_1b.bmodel
│   ├── yolo26s_depth_int8_1b.bmodel
│   └── yolo26s_depth_int8_1b_2core.bmodel
├── CV186X   # 在 CV186X 上运行的模型
│   ├── yolo26s_depth_fp32_1b.bmodel
│   ├── yolo26s_depth_fp16_1b.bmodel
│   └── yolo26s_depth_int8_1b.bmodel
└── onnx
    └── yolo26s-depth.onnx   # 导出的 onnx 模型（输出 [1,1,768,768] 度量深度图）
```
下载的数据包括：
```
./datasets
├── test          # 测试图片（NYU Depth V2 抽样）
├── cali          # 量化校准图片（NYU Depth V2 train 集抽样，200 张）
├── nyu-depth     # NYU Depth V2 Eigen split（654 张 val 图片 + 480x640 深度 .npy，官方默认评测集）
└── test_car_person_1080P.mp4  # 测试视频（通用）
```

## 4. 模型编译
**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成 BModel 才能在 SOPHON TPU 上运行，源模型在编译前要导出成 onnx 模型，具体可参考 [YOLO26 模型导出](./docs/YOLO26_depth_Export_Guide.md)。

建议使用 TPU-MLIR 编译 BModel，具体可参考 [TPU-MLIR 环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在 TPU-MLIR 环境中进入例程目录，并使用本例程提供的脚本将 onnx 模型编译为 BModel。

### 4.1 生成FP32 BModel

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x   # bm1688 / cv186x
```

执行上述命令会在 `models/BM1684X`（或 `models/BM1688`、`models/CV186X`）文件夹下生成转换好的 FP32 BModel。

### 4.2 生成FP16 BModel

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x   # bm1688 / cv186x
```

执行上述命令会在 `models/BM1684X`（或 `models/BM1688`、`models/CV186X`）文件夹下生成转换好的 FP16 BModel。

### 4.3 生成INT8 BModel

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684x   # bm1688 / cv186x
```

上述脚本会在 `models/BM1684X`（或 `models/BM1688`、`models/CV186X`）文件夹下生成转换好的 INT8 BModel（BM1688 平台会额外生成 `yolo26s_depth_int8_1b_2core.bmodel`）。量化采用 `part_asymmetric` + `percentile9999` 校准。如果您需要量化自己微调过的模型，可以参考[量化指南](./docs/YOLO26_depth_Calibration_Guide.md)中的方法。

> **说明**：模型预处理为 **stretch resize 到 768×768**（官方 val 用 `LetterBox(scale_fill=True)` 无 padding）→ BGR→RGB → /255，因此编译脚本**未使用** `--keep_aspect_ratio`；量化校准数据采用相同预处理（stretch 到 768×768），与推理口径一致。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考 [C++例程](cpp/README.md) 或 [Python例程](python/README.md) 推理 NYU Depth V2 验证集，生成逐像素深度图（float32 `.npy`，保存在 `results/depth_maps/` 下）。

然后，使用 `tools` 目录下的 `eval_depth.py` 脚本，将预测深度与 NYU Depth V2 深度真值（480×640 `.npy`）比对，计算 delta1/delta2/delta3/abs_rel/rmse/silog。该脚本复刻 ultralytics 官方 `DepthMetrics` 口径（仅统计 gt∈(0.001,100) 的有效像素）。命令如下：

```bash
# 以下命令在例程根目录下执行（请根据实际情况修改 pred_dir / gt_dir 路径）
python3 tools/eval_depth.py \
    --pred_dir results/depth_maps \
    --gt_dir datasets/nyu-depth/depth/val
```

> **说明**：官方 val 协议把 GT 深度（480×640）也 `resize` 到 768×768（`cv2.INTER_NEAREST`）后与预测逐像素比对，`eval_depth.py` 内部已按此口径处理，无需提前缩放 GT。

### 6.2 测试结果
在 NYU Depth V2 验证集（val，654 张，官方默认评测集）上，精度测试结果如下：

| 测试平台 | 测试程序              | 测试模型                      | delta1 | delta2 | delta3 | abs_rel | rmse  | silog  |
| -------- | --------------------- | ----------------------------- | ------ | ------ | ------ | ------- | ----- | ------ |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   | 0.7874 | 0.9365 | 0.9785 | 0.1528  | 0.6601 | 22.7200 |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   | 0.7874 | 0.9351 | 0.9773 | 0.1533  | 0.6638 | 22.9538 |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   | 0.7877 | 0.9365 | 0.9784 | 0.1527  | 0.6600 | 22.7261 |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   | 0.7874 | 0.9365 | 0.9785 | 0.1528  | 0.6601 | 22.7203 |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   | 0.7874 | 0.9351 | 0.9773 | 0.1533  | 0.6639 | 22.9542 |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   | 0.7877 | 0.9365 | 0.9784 | 0.1527  | 0.6601 | 22.7270 |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   | 0.7893 | 0.9362 | 0.9764 | 0.1528  | 0.6600 | 22.9592 |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   | 0.7898 | 0.9358 | 0.9762 | 0.1529  | 0.6603 | 23.0062 |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   | 0.7893 | 0.9360 | 0.9764 | 0.1529  | 0.6603 | 22.9691 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   | 0.7874 | 0.9365 | 0.9785 | 0.1528  | 0.6601 | 22.7200 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   | 0.7875 | 0.9351 | 0.9772 | 0.1533  | 0.6638 | 22.9547 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   | 0.7878 | 0.9365 | 0.9783 | 0.1527  | 0.6601 | 22.7295 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   | 0.7874 | 0.9365 | 0.9785 | 0.1528  | 0.6601 | 22.7202 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   | 0.7875 | 0.9351 | 0.9772 | 0.1533  | 0.6638 | 22.9544 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   | 0.7878 | 0.9365 | 0.9783 | 0.1527  | 0.6601 | 22.7302 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   | 0.7893 | 0.9362 | 0.9764 | 0.1528  | 0.6600 | 22.9591 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   | 0.7892 | 0.9359 | 0.9761 | 0.1532  | 0.6609 | 23.0373 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   | 0.7896 | 0.9360 | 0.9763 | 0.1527  | 0.6603 | 22.9676 |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b_2core.bmodel | 0.7893 | 0.9362 | 0.9764 | 0.1528  | 0.6600 | 22.9591 |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b_2core.bmodel | 0.7892 | 0.9359 | 0.9761 | 0.1532  | 0.6609 | 23.0373 |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b_2core.bmodel | 0.7896 | 0.9360 | 0.9763 | 0.1527  | 0.6603 | 22.9676 |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   | 0.7874 | 0.9365 | 0.9785 | 0.1528  | 0.6601 | 22.7200 |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   | 0.7875 | 0.9351 | 0.9772 | 0.1533  | 0.6638 | 22.9547 |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   | 0.7878 | 0.9365 | 0.9783 | 0.1527  | 0.6601 | 22.7295 |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   | 0.7874 | 0.9365 | 0.9785 | 0.1528  | 0.6601 | 22.7202 |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   | 0.7875 | 0.9351 | 0.9772 | 0.1533  | 0.6638 | 22.9544 |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   | 0.7878 | 0.9365 | 0.9783 | 0.1527  | 0.6601 | 22.7302 |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   | 0.7893 | 0.9362 | 0.9764 | 0.1528  | 0.6600 | 22.9591 |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   | 0.7892 | 0.9359 | 0.9761 | 0.1532  | 0.6609 | 23.0373 |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   | 0.7896 | 0.9360 | 0.9763 | 0.1527  | 0.6603 | 22.9676 |

**PyTorch 官方基线**（ultralytics 原版 `model.val(data=..., imgsz=768)`，CPU）：delta1=0.7856，abs_rel=0.1540，rmse=0.6518，silog=22.28。

> **测试说明**：
> 1. `SE7-32` 为 BM1684X 平台结果，`SE9-16` 为 BM1688 平台结果，`SE9-8` 为 CV186X 平台结果；三个平台精度一致（差异 < 0.5%），均复刻官方 `DepthMetrics` 口径；
> 2. 由于 SDK 版本之间可能存在差异，实际运行结果与本表有微小误差是正常的；
> 3. 三个程序的 `delta1`/`delta2`/`delta3` 基本一致，`rmse`/`silog` 在 OpenCV 前处理（`opencv.py`）与 BMCV 前处理（`bmcv.py`、C++）之间存在约 1% 的微小差异，原因是 bmcv 硬件 resize 与 OpenCV 软件插值的实现差异；其中 `bmcv.py`（sail resize）与 C++（vpp_convert）的硬件插值细节亦略有不同，属正常现象；
> 4. 三个程序（opencv/bmcv/C++）输出保存的 `.npy` 深度图分辨率均为 768×768，与官方 val 逐像素比对口径一致。

## 7. 性能测试
### 7.1 bmrt_test
使用 bmrt_test 测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的 bmodel 路径和 devid 参数
bmrt_test --bmodel models/BM1688/yolo26s_depth_fp32_1b.bmodel --devid 0
```
测试结果中的 `calculate time` 就是模型推理的时间（本例程 batch size 为 1，无需再除以 batch size）。

| 测试平台 | 测试模型                              | calculate time(ms) |
| -------- | ------------------------------------- | ------------------ |
| SE7-32    | BM1684X/yolo26s_depth_fp32_1b.bmodel | 67.78              |
| SE7-32    | BM1684X/yolo26s_depth_fp16_1b.bmodel | 12.69              |
| SE7-32    | BM1684X/yolo26s_depth_int8_1b.bmodel | 6.84               |
| SE9-16    | BM1688/yolo26s_depth_fp32_1b.bmodel | 395.67             |
| SE9-16    | BM1688/yolo26s_depth_fp16_1b.bmodel | 74.59              |
| SE9-16    | BM1688/yolo26s_depth_int8_1b.bmodel | 18.32              |
| SE9-16    | BM1688/yolo26s_depth_int8_1b_2core.bmodel | 13.86              |
| SE9-8     | CV186X/yolo26s_depth_fp32_1b.bmodel | 395.49             |
| SE9-8     | CV186X/yolo26s_depth_fp16_1b.bmodel | 74.47              |
| SE9-8     | CV186X/yolo26s_depth_int8_1b.bmodel | 18.21              |

### 7.2 程序运行性能
参考 [C++例程](cpp/README.md) 或 [Python例程](python/README.md) 运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++ 例程打印的时间已经折算为单张图片的处理时间。

在 NYU Depth V2 验证集（val，654 张，原图 480×640 stretch resize 至 768×768 输入，与官方 ultralytics 评测分辨率一致）上，各平台各程序的性能测试结果如下（单位均为毫秒 ms）：

| 测试平台 | 测试程序               | 测试模型                      | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | ---------------------- | ----------------------------- | ----------- | --------------- | -------------- | ---------------- |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   | 33.25       | 11.11           | 74.17          | 1.35             |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   | 7.81        | 2.43            | 69.73          | 1.35             |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   | 1.50        | 2.00            | 68.36          | 1.88             |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   | 5.06        | 10.59           | 19.08          | 1.36             |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   | 1.70        | 2.43            | 14.65          | 1.36             |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   | 1.49        | 1.99            | 13.27          | 1.84             |
| SE7-32    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   | 5.06        | 11.48           | 13.21          | 1.35             |
| SE7-32    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   | 1.71        | 2.43            | 8.83           | 1.35             |
| SE7-32    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   | 1.49        | 1.99            | 7.43           | 1.83             |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   | 14.50       | 13.05           | 403.74         | 1.77             |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   | 9.61        | 4.39            | 398.34         | 1.69             |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   | 9.76        | 3.33            | 396.60         | 2.76             |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   | 6.35        | 14.31           | 82.65          | 2.61             |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   | 2.94        | 4.37            | 77.22          | 1.68             |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   | 2.62        | 3.33            | 75.48          | 2.73             |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   | 6.34        | 13.03           | 26.34          | 1.70             |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   | 2.95        | 4.38            | 20.94          | 1.68             |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   | 2.62        | 3.33            | 19.21          | 2.75             |
| SE9-16    | yolo26_depth_opencv.py | yolo26s_depth_int8_1b_2core.bmodel | 6.34        | 13.89           | 21.95          | 1.75             |
| SE9-16    | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b_2core.bmodel | 2.95        | 4.38            | 16.44          | 1.67             |
| SE9-16    | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b_2core.bmodel | 2.61        | 3.33            | 14.74          | 2.55             |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_fp32_1b.bmodel   | 8.27        | 14.21           | 403.79         | 2.07             |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_fp32_1b.bmodel   | 37.66       | 4.38            | 398.30         | 1.65             |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_fp32_1b.bmodel   | 27.65       | 3.34            | 396.52         | 2.60             |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_fp16_1b.bmodel   | 50.30       | 14.45           | 82.83          | 2.19             |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_fp16_1b.bmodel   | 49.41       | 4.40            | 77.21          | 1.67             |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_fp16_1b.bmodel   | 32.38       | 3.34            | 75.48          | 2.66             |
| SE9-8     | yolo26_depth_opencv.py | yolo26s_depth_int8_1b.bmodel   | 34.08       | 14.26           | 26.75          | 2.09             |
| SE9-8     | yolo26_depth_bmcv.py   | yolo26s_depth_int8_1b.bmodel   | 28.07       | 4.38            | 20.92          | 1.67             |
| SE9-8     | yolo26_depth_bmcv.soc  | yolo26s_depth_int8_1b.bmodel   | 27.26       | 3.33            | 19.20          | 2.66             |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. 图片分辨率为 480×640（NYU Depth V2 原图），预处理统一 stretch resize 至 768×768；
> 4. `decode_time` 与存储读写速度及系统负载密切相关，波动明显、仅供参考；三个程序口径统一的 `preprocess/inference/postprocess` 才是稳定指标。

## 8. FAQ
请参考 [FAQ](../../docs/FAQ.md) 查看一些常见的问题与解答。