# YOLOv5

- [YOLOv5](#yolov5)
  - [1. 简介](#1-简介)
  - [2. 数据集](#2-数据集)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
    - [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    - [4.2 生成INT8 BModel](#42-生成int8-bmodel)
  - [5. 例程测试](#5-例程测试)


## 1. 简介

​	YOLOv5是非常经典的基于anchor的One Stage目标检测算法YOLO的改进版本，因其优秀的精度和速度表现，在工程实践应用中获得了非常广泛的应用。

**文档:** [YOLOv5文档](https://docs.ultralytics.com/)

**参考repo:** [yolov5](https://github.com/ultralytics/yolov5)

**实现repo：**[yolov5_demo](https://github.com/xiaotan3664/yolov5_demo)

## 2. 数据集

​YOLOv5基于[COCO2017数据集](https://cocodataset.org/#home)，该数据集是一个可用于图像检测（image detection），语义分割（semantic segmentation）和图像标题生成（image captioning）的大规模数据集。它有超过330K张图像（其中220K张是有标注的图像），包含150万个目标，80个目标类别（object categories：行人、汽车、大象等），91种材料类别（stuff categoris：草、墙、天空等），每张图像包含五句图像的语句描述，且有250,000个带关键点标注的行人。

## 3. 准备模型与数据

​Pytorch的模型在编译前要经过`torch.jit.trace`，trace后的模型才能用于编译BModel，trace方法可以采用官方`export.py`。

​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​本例程在`scripts`目录下提供了相关模型和数据（后续demo会使用）的下载脚本`1_1_prepare_model_val.sh`和`1_2_prepare_test_data.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
cd scripts
chmod +x ./*
./1_1_prepare_model_val.sh
./1_2_prepare_test_data.sh
```

​	执行后，模型保存至`data/models`，数据集下载并解压至`data/images/`

```
下载的模型包括：
./models/
├── BM1684
│   ├── yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel
│   └── yolov5s_640_coco_v6.1_3output_int8_1b.bmodel
├── BM1684X
│   ├── yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel
│   └── yolov5s_640_coco_v6.1_3output_int8_1b.bmodel
└── torch
    └── yolov5s_640_coco_v6.1_3output.torchscript.pt 

下载的数据包括：
images
├── coco                                                  # coco2017val.zip解压缩文件
├── coco200                                               # 挑选的量化数据集
├── coco2017val.zip                                       # 官方数据集
├── dog.jpg                                               # 测试图片
└── zidane.jpg                                            # 测试图片

videos
└── dance.mp4                                             # 测试视频               
```

​模型信息：

| 模型名称 | [YOLOv5s v6.1](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt) |
| :------- | :----------------------------------------------------------- |
| 训练集   | MS COCO                                                      |
| 概述     | 80类通用目标检测                                             |
| 运算量   | 16.5 GFlops                                                  |
| 输入数据 | images, [batch_size, 3, 640, 640], FP32，NCHW，RGB planar    |
| 输出数据 | 339, [batch_size, 3, 80, 80, 85], FP32 <br />391, [batch_size, 3, 40, 40, 85], FP32  <br />443, [batch_size, 3, 20, 20, 85], FP32  <br />或<br />output, [batch_size, 25200, 85], FP32 |
| 其他信息 | YOLO_MASKS: [6, 7, 8, 3, 4, 5, 0, 1, 2]  <br />YOLO_ANCHORS: [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326] |
| 前处理   | BGR->RGB、/255.0                                             |
| 后处理   | nms等                                                        |

## 4. 模型编译

​trace后的pytorch模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-NNTC，具体可参考[tpu-nntc环境搭建](../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。安装好后需在tpu-nntc环境中进入例程目录。

### 4.1 生成FP32 BModel

​本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`2_1_gen_fp32bmodel.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
cd scripts
chmod +x ./*
./2_1_gen_fp32bmodel.sh BM1684X
```

​执行上述命令会在`data/models/BM1684X/`下生成`yolov5_float32_1b.bmodel`文件，即转换好的FP32 BModel。

### 4.2 生成INT8 BModel

​不量化模型可跳过本节。

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本。请注意修改`2_2_gen_int8bmodel.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684和BM1684X），如：

```shell
cd scripts
chmod +x ./*
./2_2_gen_int8bmodel.sh BM1684X
```

​上述脚本会在`data/models/int8bmodel/BM1684X`下生成`yolov5_int8_1b.bmodel`文件，即转换好的INT8 BModel。

> **YOLOv5模型量化建议：**
>
> 1. 制作lmdb量化数据集时，通过convert_imageset.py完成数据的预处理；
> 2. 尝试不同的iterations进行量化可能得到较明显的精度提升；
> 3. 最后一层conv到输出之间层之间设置为fp32，可能得到较明显的精度提升；
>
> 4. 尝试采用不同优化策略，比如：图优化、卷积优化，可能会得到较明显精度提升。

## 5. 例程测试

- [C++例程](./cpp/yolov5_bmcv/README.md)
- [Python例程](./python/README.md)