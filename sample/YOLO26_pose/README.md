# YOLO26_pose

## 目录

- [YOLO26\_pose](#yolo26_pose)
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
YOLO26_pose 将 [YOLO26 官方开源仓库](https://github.com/ultralytics/ultralytics)（v8.4.105）的**人体关键点检测**模型 `yolo26s-pose.pt` 移植到 SOPHON BM1684X/BM1688/CV186X 上进行推理测试。

YOLO26-pose 默认使用端到端（end2end）输出，导出的 onnx 模型输入输出如下：

- 输入：`images`，shape 为 `[batch, 3, 640, 640]`（batch 为动态维度），float32，RGB、归一化到 `[0, 1]`；
- 输出：`output0`，shape 为 `[batch, 300, 57]`，float32，共 300 个人体候选框，已按置信度从高到低排序。

每一行的布局为 `[x1, y1, x2, y2, score, class_id, kpt0_x, kpt0_y, kpt0_score, ..., kpt16_x, kpt16_y, kpt16_score]`，即 4 个检测框坐标 + 1 个框置信度 + 1 个类别 id + 17 个人体关键点的 `(x, y, score)`。人体检测只有一个类别 `person`，所以 `class_id` 恒为 0；关键点与检测框坐标都是相对于 letterbox 后 640×640 图像的坐标。由于模型内部已经完成候选框筛选（等价于 NMS 的作用），**推理端不需要再做 NMS 后处理**，后处理只需按阈值过滤、将坐标反映射回原图并绘制骨架。

## 2. 特性
* 支持BM1684X(SoC)、BM1688(SoC)、CV186X(SoC)
* 支持FP32、FP16、INT8模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持端到端无 NMS 推理，模型直接输出人体框与 17 个人体关键点
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
│   ├── yolo26s-pose_fp32_1b.bmodel
│   ├── yolo26s-pose_fp16_1b.bmodel
│   ├── yolo26s-pose_int8_1b.bmodel
│   └── yolo26s-pose_int8_4b.bmodel   # batch_size=4
├── BM1688   # 在 BM1688 上运行的模型
│   ├── yolo26s-pose_fp32_1b.bmodel
│   ├── yolo26s-pose_fp16_1b.bmodel
│   ├── yolo26s-pose_int8_1b.bmodel
│   ├── yolo26s-pose_int8_1b_2core.bmodel
│   ├── yolo26s-pose_int8_4b.bmodel   # batch_size=4, num_core=1
│   └── yolo26s-pose_int8_4b_2core.bmodel   # batch_size=4, num_core=2
├── CV186X   # 在 CV186X 上运行的模型
│   ├── yolo26s-pose_fp32_1b.bmodel
│   ├── yolo26s-pose_fp16_1b.bmodel
│   ├── yolo26s-pose_int8_1b.bmodel
│   └── yolo26s-pose_int8_4b.bmodel   # batch_size=4
└── onnx
    ├── yolo26s-pose.onnx         # 导出的 onnx 模型（端到端输出 [batch, 300, 57]）
    ├── yolo26s-pose_qtable_f16   # 编译 F16 BModel 需要混合精度的层
    └── yolo26s-pose_qtable_int8  # 量化 INT8 BModel 需要混合精度的层
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
├── coco.names                                # coco类别名文件
├── coco128                                   # coco128数据集，用于模型量化
├── dance_1080P.mp4                           # 测试视频
└── coco
    ├── val2017_1000                               # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    ├── instances_val2017_1000.json                # coco val2017_1000数据集检测标签文件
    └── person_keypoints_val2017_1000.json         # coco val2017_1000数据集人体关键点标签文件，用于计算精度评价指标
```

## 4. 模型编译
**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成 BModel 才能在 SOPHON TPU 上运行，源模型在编译前要导出成 onnx 模型，具体可参考 [YOLO26-pose 模型导出](./docs/YOLO26_pose_Export_Guide.md)。同时，您需要准备用于测试的数据集；如果量化模型，还要准备用于量化的数据集。

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

执行上述命令会在 `models/BM1684X/`（或 `models/BM1688/`、`models/CV186X/`）文件夹下生成转换好的 FP16 BModel。脚本会额外传入 `yolo26s-pose_qtable_f16`，具体说明见 [4.3 生成INT8 BModel](#43-生成int8-bmodel) 中的提示。

### 4.3 生成INT8 BModel

```bash
./scripts/gen_int8bmodel_mlir.sh bm1684x   # bm1688 / cv186x
```

上述脚本会在 `models/BM1684X`（或 `models/BM1688`、`models/CV186X`）文件夹下生成转换好的 INT8 BModel：包含 `batch_size=1` 的 `yolo26s-pose_int8_1b.bmodel` 和 `batch_size=4` 的 `yolo26s-pose_int8_4b.bmodel`（BM1688 平台还会额外生成对应的 `_2core` 版本）。量化采用 `part_asymmetric` + `percentile9999` 校准，`batch_size=4` 复用 `batch_size=1` 的校准表。

> **提示**：YOLO26-pose 的端到端输出由模型内部的 TopK/GatherElements 后处理链产生，该链路对量化误差非常敏感。因此本例程提供了两个混合精度量化表 `yolo26s-pose_qtable_f16`、`yolo26s-pose_qtable_int8`（在 `download.sh` 下载的 `models/onnx` 文件夹里），通过 `model_deploy.py --quantize_table` 将这段后处理链固定在 F32 计算：
> * FP16：不加量化表时 AP@[0.5:0.95] 会从 0.576 掉到 0.290；
> * INT8：不加量化表时模型输出的候选框会被整体过滤为空，即所有图片都检测不到人体。
>
> 量化表由 `run_calibration.py` 自动生成的 `shape_pattern_qtable` 与 `fp_forward.py` 生成的敏感层量化表合并而来，与目标平台无关（BM1684X/BM1688/CV186X 通用），直接使用上述脚本即可，无需自行生成。如果您需要量化自己微调过的模型，可以参考[量化指南](./docs/YOLO26_pose_Calibration_Guide.md)中的方法。

## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考 [C++例程](cpp/README.md) 或 [Python例程](python/README.md) 对 coco val2017_1000 数据集进行推理，生成结果 json（保存在 `results/` 下，每张图片按人体记录检测框置信度与 17 个关键点坐标），推理时请把阈值放低到 0.001，避免漏检影响精度：

```bash
python3 python/yolo26_pose_bmcv.py --input datasets/coco/val2017_1000 \
    --bmodel models/BM1684X/yolo26s-pose_fp32_1b.bmodel --dev_id 0 --conf_thresh 0.001
```

然后，使用 `tools` 目录下的 `eval_coco.py` 脚本，调用 pycocotools 计算 COCO 人体关键点评价指标，命令如下：

```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 以下命令在例程根目录下执行（请根据实际情况修改 --gt_path / --result_json 路径）
python3 tools/eval_coco.py --ann_type keypoints \
    --gt_path datasets/coco/person_keypoints_val2017_1000.json \
    --result_json results/yolo26s-pose_fp32_1b.bmodel_val2017_1000_bmcv_python_result.json
```

> **说明**：指标口径与 COCO 官方一致——`AP@IoU=0.5:0.95`（表中简写为 AP，对应 `stats[0]`：IoU 阈值 0.5:0.95、area=all、maxDets=20）与 `AP@IoU=0.5`（`stats[1]`）。结果 json 中每个人的 `score` 保存的是**检测框置信度**，17 个关键点的置信度另存于 `keypoints` 字段；`eval_coco.py` 在计算指标时会按仓库中其他 pose 例程的同一口径，把检测框置信度与 17 个关键点置信度的均值相加。

### 6.2 测试结果
在 coco val2017_1000 数据集（1000 张）上，推理阈值取 0.001，精度测试结果如下：

| 测试平台 | 测试程序             | 测试模型                | AP@IoU=0.5:0.95 | AP@IoU=0.5 |
| -------- | -------------------- | ----------------------- | --------------- | ---------- |
|   SE7-32    |yolo26_pose_opencv.py |     yolo26s-pose_fp32_1b.bmodel     | 0.575 | 0.756 |
|   SE7-32    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp32_1b.bmodel     | 0.576 | 0.756 |
|   SE7-32    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp32_1b.bmodel     | 0.579 | 0.757 |
|   SE7-32    |yolo26_pose_opencv.py |     yolo26s-pose_fp16_1b.bmodel     | 0.574 | 0.747 |
|   SE7-32    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp16_1b.bmodel     | 0.576 | 0.756 |
|   SE7-32    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp16_1b.bmodel     | 0.579 | 0.756 |
|   SE7-32    |yolo26_pose_opencv.py |     yolo26s-pose_int8_1b.bmodel     | 0.565 | 0.749 |
|   SE7-32    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_1b.bmodel     | 0.564 | 0.749 |
|   SE7-32    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_1b.bmodel     | 0.565 | 0.749 |
|   SE7-32    |yolo26_pose_opencv.py |     yolo26s-pose_int8_4b.bmodel     | 0.565 | 0.749 |
|   SE7-32    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_4b.bmodel     | 0.564 | 0.749 |
|   SE7-32    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_4b.bmodel     | 0.565 | 0.749 |
|   SE9-16    |yolo26_pose_opencv.py |     yolo26s-pose_fp32_1b.bmodel     | 0.575 | 0.756 |
|   SE9-16    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp32_1b.bmodel     | 0.576 | 0.756 |
|   SE9-16    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp32_1b.bmodel     | 0.578 | 0.757 |
|   SE9-16    |yolo26_pose_opencv.py |     yolo26s-pose_fp16_1b.bmodel     | 0.575 | 0.756 |
|   SE9-16    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp16_1b.bmodel     | 0.576 | 0.756 |
|   SE9-16    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp16_1b.bmodel     | 0.579 | 0.756 |
|   SE9-16    |yolo26_pose_opencv.py |     yolo26s-pose_int8_1b.bmodel     | 0.565 | 0.749 |
|   SE9-16    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_1b.bmodel     | 0.562 | 0.749 |
|   SE9-16    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_1b.bmodel     | 0.563 | 0.749 |
|   SE9-16    |yolo26_pose_opencv.py |     yolo26s-pose_int8_4b.bmodel     | 0.565 | 0.749 |
|   SE9-16    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_4b.bmodel     | 0.562 | 0.749 |
|   SE9-16    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_4b.bmodel     | 0.563 | 0.749 |
|   SE9-16    |yolo26_pose_opencv.py |  yolo26s-pose_int8_1b_2core.bmodel  | 0.565 | 0.749 |
|   SE9-16    | yolo26_pose_bmcv.py  |  yolo26s-pose_int8_1b_2core.bmodel  | 0.562 | 0.749 |
|   SE9-16    | yolo26_pose_bmcv.soc |  yolo26s-pose_int8_1b_2core.bmodel  | 0.563 | 0.749 |
|   SE9-16    |yolo26_pose_opencv.py |  yolo26s-pose_int8_4b_2core.bmodel  | 0.565 | 0.749 |
|   SE9-16    | yolo26_pose_bmcv.py  |  yolo26s-pose_int8_4b_2core.bmodel  | 0.562 | 0.749 |
|   SE9-16    | yolo26_pose_bmcv.soc |  yolo26s-pose_int8_4b_2core.bmodel  | 0.563 | 0.749 |
|    SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_fp32_1b.bmodel     | 0.576 | 0.756 |
|    SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp32_1b.bmodel     | 0.576 | 0.756 |
|    SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp32_1b.bmodel     | 0.578 | 0.757 |
|    SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_fp16_1b.bmodel     | 0.576 | 0.756 |
|    SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp16_1b.bmodel     | 0.576 | 0.756 |
|    SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp16_1b.bmodel     | 0.579 | 0.756 |
|    SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_int8_1b.bmodel     | 0.561 | 0.748 |
|    SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_1b.bmodel     | 0.562 | 0.749 |
|    SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_1b.bmodel     | 0.563 | 0.749 |
|    SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_int8_4b.bmodel     | 0.561 | 0.748 |
|    SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_4b.bmodel     | 0.562 | 0.749 |
|    SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_4b.bmodel     | 0.563 | 0.749 |

> **测试说明**：
> 1. 由于 SDK 版本之间可能存在差异，实际运行结果与本表有 <0.01 的精度误差是正常的；
> 2. C++（bmcv）与 Python 在部分图片上存在微小差异，原因是 bmcv 硬件 resize 与 OpenCV 软件插值的实现差异，属正常现象；
> 3. 在搭载相同 TPU 和 SOPHONSDK 的平台上，相同程序的精度一致，SE7 系列对应 BM1684X，SE9-16 对应 BM1688，SE9-8 对应 CV186X；
> 4. FP16 与 FP32 精度完全一致（端到端后处理链按 [4.3 节](#43-生成int8-bmodel)的说明固定在 F32），INT8 相比 FP32 的衰减在 0.015 以内；
> 5. SE9-16 的 `int8_1b_2core` 为 BM1688 双核版本，精度与单核一致，推理时间约降低 10%~15%；
> 6. `int8_4b` 为 `batch_size=4` 的 INT8 模型（含 BM1688 的 `_2core` 版本），精度与 `int8_1b` 一致，可用于多路/多batch推理。

## 7. 性能测试
### 7.1 bmrt_test
使用 bmrt_test 测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的 bmodel 路径和 devid 参数
bmrt_test --bmodel models/BM1684X/yolo26s-pose_fp32_1b.bmodel --devid 0
```
测试结果中的 `calculate time` 就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间（本例程 bmrt_test 结果已换算为每张图片的理论推理时间）。测试各个模型的理论推理时间，结果如下：

| 测试平台 | 测试模型                            | calculate time(ms) |
| -------- | ----------------------------------- | ------------------ |
| SE7-32   | BM1684X/yolo26s-pose_fp32_1b.bmodel  | 30.50              |
| SE7-32   | BM1684X/yolo26s-pose_fp16_1b.bmodel  | 7.52               |
| SE7-32   | BM1684X/yolo26s-pose_int8_1b.bmodel  | 4.66               |
| SE7-32   | BM1684X/yolo26s-pose_int8_4b.bmodel  | 3.84               |
| SE9-16   | BM1688/yolo26s-pose_fp32_1b.bmodel   | 149.99             |
| SE9-16   | BM1688/yolo26s-pose_fp16_1b.bmodel   | 38.78              |
| SE9-16   | BM1688/yolo26s-pose_int8_1b.bmodel   | 10.41              |
| SE9-16   | BM1688/yolo26s-pose_int8_4b.bmodel   | 9.07               |
| SE9-16   | BM1688/yolo26s-pose_int8_1b_2core.bmodel | 9.04               |
| SE9-16   | BM1688/yolo26s-pose_int8_4b_2core.bmodel | 5.87               |
| SE9-8    | CV186X/yolo26s-pose_fp32_1b.bmodel   | 150.03             |
| SE9-8    | CV186X/yolo26s-pose_fp16_1b.bmodel   | 38.76              |
| SE9-8    | CV186X/yolo26s-pose_int8_1b.bmodel   | 10.41              |
| SE9-8    | CV186X/yolo26s-pose_int8_4b.bmodel   | 9.06               |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time` 已折算为平均每张图片的推理时间；
> 3. 本例程的 ONNX 采用 `dynamic=True` 导出（见[模型导出指南](docs/YOLO26_pose_Export_Guide.md)），以便编译 `batch_size=4` 的 BModel；动态 shape 会引入少量额外开销，因此上表数值略高于同结构的静态导出模型。

### 7.2 程序运行性能
参考 [C++例程](cpp/README.md) 或 [Python例程](python/README.md) 运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++ 例程打印的时间已经折算为单张图片的处理时间。

在 coco val2017_1000 数据集（1000 张，原图尺寸不一，letterbox 至 640×640 输入）上，各程序的性能测试结果如下（单位均为毫秒 ms）：

| 测试平台 | 测试程序             | 测试模型                | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | -------------------- | ----------------------- | ----------- | --------------- | -------------- | ---------------- |
|   SE7-32    |yolo26_pose_opencv.py |     yolo26s-pose_fp32_1b.bmodel     |      7.61      |     27.64      |     33.55      |      1.67      |
|   SE7-32    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp32_1b.bmodel     |      2.83      |      2.02      |     30.18      |      1.62      |
|   SE7-32    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp32_1b.bmodel     |      2.57      |      1.37      |     29.85      |      0.07      |
|   SE7-32    |yolo26_pose_opencv.py |     yolo26s-pose_fp16_1b.bmodel     |      7.20      |     27.36      |     10.61      |      1.65      |
|   SE7-32    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp16_1b.bmodel     |      2.84      |      2.02      |      7.21      |      1.62      |
|   SE7-32    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp16_1b.bmodel     |      2.57      |      1.37      |      6.87      |      0.07      |
|   SE7-32    |yolo26_pose_opencv.py |     yolo26s-pose_int8_1b.bmodel     |      7.18      |     26.86      |      7.70      |      1.63      |
|   SE7-32    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_1b.bmodel     |      2.82      |      2.02      |      4.31      |      1.60      |
|   SE7-32    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_1b.bmodel     |      2.56      |      1.36      |      3.97      |      0.07      |
|   SE7-32    |yolo26_pose_opencv.py |     yolo26s-pose_int8_4b.bmodel     |      6.95      |     25.53      |      6.95      |      1.49      |
|   SE7-32    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_4b.bmodel     |      2.67      |      1.92      |      3.81      |      1.48      |
|   SE7-32    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_4b.bmodel     |      2.49      |      1.32      |      3.72      |      0.06      |
|   SE9-16    |yolo26_pose_opencv.py |     yolo26s-pose_fp32_1b.bmodel     |      9.34      |     36.71      |     154.06     |      1.15      |
|   SE9-16    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp32_1b.bmodel     |      3.72      |      3.96      |     149.89     |      1.13      |
|   SE9-16    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp32_1b.bmodel     |      3.30      |      2.61      |     149.37     |      0.12      |
|   SE9-16    |yolo26_pose_opencv.py |     yolo26s-pose_fp16_1b.bmodel     |      9.30      |     35.39      |     42.81      |      1.14      |
|   SE9-16    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp16_1b.bmodel     |      3.73      |      3.96      |     38.58      |      1.13      |
|   SE9-16    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp16_1b.bmodel     |      3.33      |      2.59      |     38.12      |      0.12      |
|   SE9-16    |yolo26_pose_opencv.py |     yolo26s-pose_int8_1b.bmodel     |      9.29      |     35.83      |     14.46      |      1.13      |
|   SE9-16    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_1b.bmodel     |      3.73      |      3.97      |     10.20      |      1.11      |
|   SE9-16    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_1b.bmodel     |      3.30      |      2.59      |      9.74      |      0.11      |
|   SE9-16    |yolo26_pose_opencv.py |     yolo26s-pose_int8_4b.bmodel     |      9.28      |     34.09      |     13.05      |      0.98      |
|   SE9-16    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_4b.bmodel     |      3.47      |      3.76      |      9.09      |      0.97      |
|   SE9-16    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_4b.bmodel     |      3.12      |      2.14      |      8.92      |      0.09      |
|   SE9-16    |yolo26_pose_opencv.py |  yolo26s-pose_int8_1b_2core.bmodel  |      9.32      |     37.48      |     13.07      |      1.13      |
|   SE9-16    | yolo26_pose_bmcv.py  |  yolo26s-pose_int8_1b_2core.bmodel  |      3.72      |      3.97      |      8.82      |      1.10      |
|   SE9-16    | yolo26_pose_bmcv.soc |  yolo26s-pose_int8_1b_2core.bmodel  |      3.30      |      2.60      |      8.36      |      0.11      |
|   SE9-16    |yolo26_pose_opencv.py |  yolo26s-pose_int8_4b_2core.bmodel  |      9.29      |     33.43      |      9.83      |      0.98      |
|   SE9-16    | yolo26_pose_bmcv.py  |  yolo26s-pose_int8_4b_2core.bmodel  |      3.47      |      3.76      |      5.86      |      0.97      |
|   SE9-16    | yolo26_pose_bmcv.soc |  yolo26s-pose_int8_4b_2core.bmodel  |      3.11      |      2.14      |      5.71      |      0.09      |
|    SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_fp32_1b.bmodel     |      5.98      |     36.50      |     154.08     |      2.46      |
|    SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp32_1b.bmodel     |      3.80      |      3.97      |     149.86     |      2.42      |
|    SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp32_1b.bmodel     |      3.45      |      2.61      |     149.37     |      0.12      |
|    SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_fp16_1b.bmodel     |      4.50      |     36.57      |     42.82      |      2.44      |
|    SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_fp16_1b.bmodel     |      3.82      |      3.97      |     38.55      |      2.43      |
|    SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_fp16_1b.bmodel     |      3.44      |      2.60      |     38.12      |      0.11      |
|    SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_int8_1b.bmodel     |      4.47      |     36.60      |     14.47      |      2.42      |
|    SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_1b.bmodel     |      3.96      |      3.97      |     10.17      |      2.38      |
|    SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_1b.bmodel     |      3.38      |      2.60      |      9.74      |      0.11      |
|    SE9-8    |yolo26_pose_opencv.py |     yolo26s-pose_int8_4b.bmodel     |      4.26      |     34.53      |     13.06      |      2.25      |
|    SE9-8    | yolo26_pose_bmcv.py  |     yolo26s-pose_int8_4b.bmodel     |      3.50      |      3.75      |      9.07      |      2.20      |
|    SE9-8    | yolo26_pose_bmcv.soc |     yolo26s-pose_int8_4b.bmodel     |      3.12      |      2.15      |      8.90      |      0.09      |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. 图片分辨率对解码时间影响较大，推理时间为逐张图片的模型计算时间，与输入分辨率（固定 640×640）无关；
> 4. `decode_time` 与图片是否已进入系统页缓存有关：程序首次在 val2017_1000 上运行时缓存未命中，`decode_time` 偏高，缓存预热后同一程序会明显下降（本表为预热后的稳定值）。bmcv 走硬件解码，耗时低于 opencv 程序；opencv 程序的解码耗时 SE9-16 约 9.3 ms、SE7-32 约 7 ms、SE9-8 约 4~6 ms，属正常现象；
> 5. 本例程端到端输出无需 NMS，后处理只做阈值过滤与坐标反映射，因此 `postprocess_time` 很小（Python 约 1~2.5 ms，C++ 约 0.1 ms）。

## 8. FAQ
请参考 [FAQ](../../docs/FAQ.md) 查看一些常见的问题与解答。
