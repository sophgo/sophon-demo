# YOLOv34

- [YOLOv34](#yolov34)
  - [1. 简介](#1-简介)
  - [2. 数据集](#2-数据集)
  - [3. 准备模型与数据](#3-准备模型与数据)
  - [4. 模型编译](#4-模型编译)
    - [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    - [4.2 生成INT8 BModel](#42-生成int8-bmodel)
  - [5. 例程测试](#5-例程测试)


## 1. 简介

作为一种经典的单阶段目标检测框架，YOLO系列的目标检测算法得到了学术界与工业界们的广泛关注。由于YOLO系列属于单阶段目标检测，因而具有较快的推理速度，能够更好的满足现实场景的需求。随着YOLOv3算法的出现，使得YOLO系列的检测算达到了高潮。YOLOv4则是在YOLOv3算法的基础上增加了很多实用的技巧，使得它的速度与精度都得到了极大的提升。

**文档:** [YOLOv4文档](https://arxiv.org/pdf/2004.10934.pdf)

**参考repo:** [yolov34]([GitHub - AlexeyAB/darknet: YOLOv4 / Scaled-YOLOv4 / YOLO - Neural Networks for Object Detection (Windows and Linux version of Darknet )](https://github.com/AlexeyAB/darknet))

## 2. 数据集

​	YOLOv34基于[COCO2017数据集](https://cocodataset.org/#home)，该数据集是一个可用于图像检测（image detection），语义分割（semantic segmentation）和图像标题生成（image captioning）的大规模数据集。它有超过330K张图像（其中220K张是有标注的图像），包含150万个目标，80个目标类别（object categories：行人、汽车、大象等），91种材料类别（stuff categoris：草、墙、天空等），每张图像包含五句图像的语句描述，且有250,000个带关键点标注的行人。

## 3. 准备模型与数据

​	同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

​	本例程在`scripts`目录下提供了相关模型和测试数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

- Remember to use your own anchors, mask and classes number config values in `cpp/yolov3.hpp` and `python/configs/*.yml`

```bash
sudo chmod +x -R ./scripts
cd scripts
sudo ./download.sh
```

​	执行后，模型保存至`data/models`，数据集下载并解压至`data/images/`

```
下载的模型包括：
./models/
├── BM1684
│   ├──yolov4_416_coco_fp32_1b.bmodel      # BM1684 fp32bmodel，[1 3 416 416]
│   └── yolov4_416_coco_int8_1b.bmodel                                                
└── darknet
    ├── yolov4.cfg                                        # 官方darknet模型
    └── yolov4.weights 


下载的数据包括：
images
├── val2017                                                  # coco2017val.zip解压缩文件
├── coco200                                               # 挑选的量化数据集
├── coco2017val.zip                                       # 官方数据集
├── dog.jpg                                               # 测试图片
├── bus.jpg                                               # 测试图片
├── horse.jpg                                             # 测试图片
├── person.jpg                                            # 测试图片
└── zidane.jpg                                            # 测试图片

videos
└── dance.mp4                                             # 测试视频               
```

可以点击连接下载bmodel模型 [here](http://219.142.246.77:65000/sharing/SJ4gY5Nzz) 并将其放入 `data/models/BM1684` 路径下

​	模型信息：

| 模型文件                       | 输入                                                | 输出                                                         | anchors and masks                                            |
| ------------------------------ | --------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| yolov3_416_coco_fp32_1b.bmodel | input: data, [1, 3, 416, 416], float32, scale: 1    | output: Yolo0, [1, 255, 13, 13], float32, scale: 1<br/>output: Yolo1, [1, 255, 26, 26], float32, scale: 1<br/>output: Yolo2, [1, 255, 52, 52], float32, scale: 1 | YOLO_MASKS: [6, 7, 8, 3, 4, 5, 0, 1, 2]<br />YOLO_ANCHORS: [10, 13, 16, 30, 33, 23, 30, 61, 62, 45,59, 119, 116, 90, 156, 198, 373, 326] |
| yolov3_608_coco_fp32_1b.bmodel | input: data, [1, 3, 608, 608], float32, scale: 1    | output: Yolo0, [1, 255, 19, 19], float32, scale: 1<br/>output: Yolo1, [1, 255, 38, 38], float32, scale: 1<br/>output: Yolo2, [1, 255, 76, 76], float32, scale: 1 | YOLO_MASKS: [6, 7, 8, 3, 4, 5, 0, 1, 2]<br />YOLO_ANCHORS: [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326] |
| yolov4_416_coco_fp32_1b.bmodel | input: data, [1, 3, 416, 416], float32, scale: 1    | output: Yolo0, [1, 255, 52, 52], float32, scale: 1<br/>output: Yolo1, [1, 255, 26, 26], float32, scale: 1<br/>output: Yolo2, [1, 255, 13, 13], float32, scale: 1 | YOLO_MASKS: [0, 1, 2, 3, 4, 5, 6, 7, 8]<br />YOLO_ANCHORS: [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401] |
| yolov4_608_coco_fp32_1b.bmodel | input: data, [1, 3, 608, 608], float32, scale: 1    | output: Yolo0, [1, 255, 76, 76], float32, scale: 1<br/>output: Yolo1, [1, 255, 38, 38], float32, scale: 1<br/>output: Yolo2, [1, 255, 19, 19], float32, scale: 1 | YOLO_MASKS: [0, 1, 2, 3, 4, 5, 6, 7, 8]<br /> YOLO_ANCHORS: [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401] |
| yolov4_608_coco_int8_1b.bmodel | input: data, [1, 3, 608, 608], int8, scale: 127.986 | output: Yolo0, [1, 255, 76, 76], float32, scale: 0.0078125<br/>output: Yolo1, [1, 255, 38, 38], float32, scale: 0.0078125<br/>output: Yolo2, [1, 255, 19, 19], float32, scale: 0.0078125 | YOLO_MASKS: [0, 1, 2, 3, 4, 5, 6, 7, 8]<br /> YOLO_ANCHORS: [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401] |



## 4. 模型编译

​	darknet模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-NNTC(>=3.1.0)，具体可参考[tpu-nntc环境搭建](../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。

### 4.1 生成FP32 BModel

​	darknet模型编译为FP32 BModel，具体方法可参考[BMNETD 使用 — NNToolChain 3.0.0 文档 (sophgo.com)](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/nntc/html/usage/bmnetd.html)

​	本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`2_1_gen_fp32bmodel.sh`中的模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
cd scripts
./gen_fp32bmodel.sh BM1684
```

​	执行上述命令会在`data/models/BM1684/`下生成`yolov4_coco_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

### 4.2 生成INT8 BModel

​	不量化模型可跳过本节。

​	darknet模型的量化方法可参考[Quantization-Tools User Guide](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/calibration-tools/html/index.html)，主要步骤如下：

- 使用 `ufwio.io` 从图片生成LMDB数据集 
- 使用 `bmnetd --mode=GenUmodel`将`.cfg` 和`.weights`转换为 fp32 umodel 
- 使用 `calibration_use_pb quantize` 将 fp32 umodel 转换为 int8 umodel 
- 使用 `bmnetu` 将 int8 umodel 转换为 int8 bmodel 

​	本例程在`scripts`目录下提供了量化INT8 BModel的脚本。请注意修改`2_2_gen_int8bmodel.sh`中的darknet模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684和BM1684X），如：

```shell
cd scripts
./gen_int8bmodel.sh BM1684X
```

​	上述脚本会在`data/models/BM1684X`下生成`yolov4_coco_int8_1b.bmodel`文件，即转换好的INT8 BModel。

> **YOLOv34模型量化建议（也可参考[官方量化手册指导](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/calibration-tools/html/module/chapter7.html)）：**
>
> 1. 制作lmdb量化数据集时，通过convert_imageset.py完成数据的预处理；
> 2. 尝试不同的iterations进行量化可能得到较明显的精度提升；
> 3. 最后一层conv到输出之间层之间设置为fp32，可能得到较明显的精度提升；
>
> 4. 尝试采用不同优化策略，比如：图优化、卷积优化，可能会得到较明显精度提升。

## 5. 例程测试

- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)