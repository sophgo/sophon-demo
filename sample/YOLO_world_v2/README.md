# YOLO_world_v2

## 目录

- [YOLO\_world\_v2](#yolo_world_v2)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)
    - [6.1 测试方法](#61-测试方法)
    - [6.2 测试结果](#62-测试结果)
  - [7. 性能测试](#7-性能测试)
    - [7.1 bmrt\_test](#71-bmrt_test)
    - [7.2 程序运行性能](#72-程序运行性能)
  - [8. FAQ](#8-faq)

## 1. 简介
YOLO-World 是腾讯人工智能实验室提出的实时开放词汇目标检测器，采用视觉语言建模和预训练的方法，能够在无需预先训练的情况下，实时识别图像中任何由描述性文本指定的物体。本例程对 [YOLO-World v2](https://github.com/AILab-CVC/YOLO-World) 官方开源仓库的模型和算法进行移植（ultralytics 导出的 `yolov8s-worldv2` 权重），使之能在SOPHON BM1684X上进行推理测试。

模型由两个子模型串联：`clip_text_vitb32` 把类别名编码为文本嵌入 `txt_feats[1,80,512]`，再送入 `yoloworld_v2` 主检测模型得到 `output[1,84,8400]`，经 NMS 后处理输出检测框。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持FP32、FP16模型编译和推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持开放词汇（open-vocabulary）：运行时通过 `--class_names` 指定任意类别
* 支持图片和视频测试

## 3. 准备模型与数据
建议使用TPU-MLIR编译BModel，在使用TPU-MLIR编译前需要导出ONNX模型。具体可参考[YOLO_world_v2模型导出](./docs/YOLO_World_v2_Export_Guide.md)。

同时，您需要准备用于测试的数据集。

本例程在`scripts`目录下提供了数据下载脚本`download.sh`（BModel已本地编译，仅下载测试数据集），您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行后，测试数据集下载并解压至`datasets/test/`，精度测试数据集下载并解压至`datasets/coco/val2017_1000/`（含`instances_val2017_1000.json`），测试视频下载至`datasets/test_car_person_1080P.mp4`，`coco.names`下载至`datasets/`。

```
下载的模型包括：
./models
├── BM1684X
│   ├── yoloworld_v2_fp32_1b.bmodel        # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yoloworld_v2_fp16_1b.bmodel        # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   └── clip_text_vitb32_bm1684x_f16_1b.bmodel  # CLIP文本编码部分FP16 BModel
├── onnx
│   ├── yoloworld_v2.onnx                     # 导出的主检测onnx模型
│   └── clip_text_vitb32.onnx              # 导出的CLIP文本编码onnx模型
├── bpe_simple_vocab_16e6.txt.gz           # CLIP BPE分词所需的合并规则和基础词汇表
├── text_projection_512_512.npy            # 导出CLIP onnx时保存的text_projection数据，推理时使用
├── coco80_txt_feats.npy                   # (可选)预计算的80类txt_feats，用--txt_feats_npy加载可加速
└── yolov8s-worldv2.pt                     # 官方v2权重
```
下载的数据包括：
```
./datasets
├── test                                   # 测试图片
├── test_car_person_1080P.mp4              # 测试视频
├── coco.names                             # coco类别名文件
└── coco
    ├── val2017_1000                        # coco val2017_1000数据集：coco val2017中随机抽取的1000张样本
    └── instances_val2017_1000.json         # coco val2017_1000数据集标签文件，用于计算精度评价指标
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载/已编译好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

本例程在 TPU-MLIR 容器中编译（环境搭建见 [Environment_Install_Guide.md](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)）。进入容器后，在例程目录下使用`scripts`目录下的脚本编译，执行时指定BModel运行的目标平台（**支持BM1684X**），如：

- 生成FP32 BModel

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

执行后会在`models/BM1684X`下生成`yoloworld_v2_fp32_1b.bmodel`和`clip_text_vitb32_bm1684x_f16_1b.bmodel`。

- 生成FP16 BModel

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

执行后会在`models/BM1684X/`下生成`yoloworld_v2_fp16_1b.bmodel`和`clip_text_vitb32_bm1684x_f16_1b.bmodel`。

> ℹ️ `--mean`/`--scale` 仅写入 mlir、用于 INT8 校准量化，**不烤入 bmodel 计算图**，对 FP32/FP16 推理无影响；推理侧 python 的 `/255` 是唯一归一化。脚本沿用 `--mean 0 --scale 1/255 --keep_aspect_ratio --pixel_format rgb --output_names output` 与 v1 一致。
> ⚠️ CLIP文本编码ONNX必须用**torch 1.13**导出（torch 2.0+的MHA走SDPA，TPU-MLIR会误编译），详见[模型导出](./docs/YOLO_World_v2_Export_Guide.md)。

## 5. 例程测试
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/coco/val2017_1000)和相关参数(class_names="all"、conf_thresh=0.001、nms_thresh=0.7)。  
然后，使用`tools`目录下的`eval_coco.py`脚本，将测试生成的json文件与测试集标签json文件进行对比，计算出目标检测的评价指标，命令如下：
```bash
# 安装pycocotools，若已安装请跳过
pip3 install pycocotools
# 请根据实际情况修改程序路径和json文件路径
python3 tools/eval_coco.py --gt_path datasets/coco/instances_val2017_1000.json --result_json results/yoloworld_v2_fp32_1b.bmodel_val2017_1000_opencv_python_result.json
```
### 6.2 测试结果
在coco2017 val数据集上，精度测试结果如下：
|   测试平台    |      测试程序     |      测试模型          |AP@IoU=0.5:0.95|AP@IoU=0.5|
| ------------ | ---------------- | ---------------------- | ------------- | -------- |
| SE7-32       | yoloworld_opencv.py | yoloworld_v2_fp32_1b.bmodel |    0.376 |    0.522 |
| SE7-32       | yoloworld_opencv.py | yoloworld_v2_fp16_1b.bmodel |    0.376 |    0.522 |
| SE7-32       | yoloworld_bmcv.py   | yoloworld_v2_fp16_1b.bmodel |    0.371 |    0.514 |
> **测试说明**：
> 1. FP32与FP16精度一致；opencv与bmcv精度基本一致；
> 2. 与旧版`sample/YOLO_world`（v1）的0.370一致，<0.01的精度误差源于ultralytics版本差异，属正常；
> 3. AP@IoU=0.5:0.95为area=all对应的指标；
> 4. 本例程使用v2权重（`yolov8s-worldv2.pt`），BModel文件名带`_v2_`以与v1区分，避免混用。


## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能（SE7-32 SoC上，`bmrt_test`位于`/opt/sophon/libsophon-0.5.1/bin/bmrt_test`）：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/yoloworld_v2_fp16_1b.bmodel --devid 0
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|    测试平台  |              测试模型               | calculate time(ms) |
| ----------- | ----------------------------------- | ----------------- |
|   SE7-32    | BM1684X/yoloworld_v2_fp32_1b.bmodel   |          35.26  |
|   SE7-32    | BM1684X/yoloworld_v2_fp16_1b.bmodel   |           6.87  |
|   SE7-32    | BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel|          4.37  |
> **测试说明**：
> 1. `clip_text`为一次性文本编码（按类集摊销，不计入每图耗时）；
> 2. 性能测试结果具有一定的波动性；
> 3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/coco/val2017_1000`，conf_thresh=0.25，nms_thresh=0.7，性能测试结果如下：
|    测试平台  |     测试程序      |        测试模型        |decode_time|preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ---------------------- | --------  | ---------    | ---------     | ---------      |
|   SE7-32    |yoloworld_opencv.py|yoloworld_v2_fp32_1b.bmodel |      6.90       |      22.87      |      40.74      |      5.39       |
|   SE7-32    |yoloworld_opencv.py|yoloworld_v2_fp16_1b.bmodel |      6.83       |      22.74      |      12.31      |      5.36       |
|   SE7-32    | yoloworld_bmcv.py |yoloworld_v2_fp32_1b.bmodel |      2.94       |       2.00      |      43.54      |      5.37       |
|   SE7-32    | yoloworld_bmcv.py |yoloworld_v2_fp16_1b.bmodel |      2.88       |       2.00      |      15.09      |      5.40       |

> **测试说明**：
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. BMCV预处理(2.0ms)远快于OpenCV(22.7ms)，bmcv例程端到端更快；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。另外本例程的常见问题：

- **AP=0 / 检测全是某一类且分数极低**：首选排查CLIP文本嵌入是否正确（clip bmodel必须用torch 1.13导出，嵌入与ultralytics/OpenAI的cos须>0.9999；错位会让score极低）。`--mean/--scale`不烤入bmodel，不是AP=0的原因。
- **clip bmodel输出与onnx不一致**：torch 2.0+导出的CLIP MHA会被TPU-MLIR误编译成退化图，需用torch 1.13 venv重新导出，详见[模型导出](./docs/YOLO_World_v2_Export_Guide.md)。
