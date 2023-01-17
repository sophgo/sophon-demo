# yolov34 Python例程

> Notes：For Python codes,  create your own config file *.yml in `configs` based on the values of `ENGINE_FILE`, `LABEL_FILE `, `YOLO_MASKS`, `YOLO_ANCHORS`, `OUTPUT_TENSOR_CHANNELS`,`DEV_ID` for your model.

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | yolov34_opencv.py | 使用OpenCV解码、OpenCV前处理、SAIL推理 |

## 1. x86 PCIe平台
## 1.1 环境准备
如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)、sophon-ffmpeg(>=0.2.4)和sophon-sail(>=3.1.0),具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)。
此外您可能还需要安装其他第三方库：

```bash
pip3 install -r requirements.txt
```

请确保在测试开始之前已经执行download.sh

## 1.2 测试命令

python例程不需要编译，可以直接运行。以yolov34_opencv.py的推理为例，参数说明如下：

```bash
usage:yolov34_opencv.py [--config file CONFIG_PATH] [--input IMAGE_PATH]
# default: --cfgfile=configs/yolov4_416.yml --input=../data/images/person.jpg
--cfgfile=<config file> config文件目录（请根据实际情况修改yml文件中的ENGINE_FILE的参数值，从而更改默认bmodel路径,若有其余需要也请修改yml中相关参数）
--input=<image file path>推理图片路径，仅单张图
#$ python3 yolov34_opencv.py --help # show help info
```
confige文件修改如下：
```yml
DETECTOR: # 检测模型
    TYPE: "coco_detector"
    NAME: 'yolov34'
    PRODUCER_NAME: 'sophon'
    ENGINE_FILE: '../data/models/BM1684X/yolov4_416_coco_fp32_1b.bmodel'    #更改ENGINE_FILE更换模型文件
    LABEL_FILE: '../data/coco.names'                                        #更改LABLE_FILE更换标签文件
    DEV_ID: 0                                                               #更改DEV_ID选择芯片

    YOLO_MASKS: [0, 1, 2, 3, 4, 5, 6, 7, 8]
    YOLO_ANCHORS: [12, 16, 19, 36, 40, 28, 36, 75,
                    76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    
    OUTPUT_TENSOR_CHANNELS: [52, 26, 13]
    MIN_CONFIDENCE: 0.5
    NMS_MAX_OVERLAP: 0.45

```

测试实例如下：
```bash
# 测试单张图
python yolov34_opencv.py
$ cd python
$ python3 yolov34_opencv.py --input=../data/images/person.jpg
```

执行完成后，会将预测结果保存在`results_images`下，同时会打印预测结果、推理时间等信息。

```bash
INFO 09/16/2022 15:27:14.482 139952929467264 model.py:86] detect - yolov3 cost: 0.027872 seconds
DEBUG 09/16/2022 15:27:14.482 139952929467264 model.py:100] <listcomp> - (1, 255, 52, 52)
DEBUG 09/16/2022 15:27:14.482 139952929467264 model.py:100] <listcomp> - (1, 255, 26, 26)
DEBUG 09/16/2022 15:27:14.482 139952929467264 model.py:100] <listcomp> - (1, 255, 13, 13)
classID=0, classLabel=person, conf=0.995000, bbox=(194,126,269,344)
classID=17, classLabel=horse, conf=0.985000, bbox=(418,152,590,333)
classID=16, classLabel=dog, conf=0.985000, bbox=(52,270,214,339)
INFO 09/16/2022 15:27:14.495 139952929467264 yolov34_opencv.py:70] <module> - [('person', 0.995, [194, 126, 269, 344]), ('horse', 0.985, [418, 152, 590, 333]), ('dog', 0.985, [52, 270, 214, 339])]
```

可通过改变模型进行int8的推理测试。


## 2. arm SoC平台
## 2.1 环境准备
如果您使用SoC平台测试本例程，您需要交叉编译安装sophon-sail(>=3.1.0)，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。
此外您可能还需要安装其他第三方库：

```bash
# for Debian 9, please specify numpy version 1.17.2
pip3 install -r requirements.txt
```
请确保在测试开始之前已经执行download.sh

## 2.2 测试命令

SoC平台的测试方法与x86 PCIe平台相同，请参考[1.2 测试命令](#12-测试命令)。

## 3. 精度与性能测试
### 3.1性能测试
可以使用bmrt_test测试模型的理论性能：
```bash
bmrt_test --bmodel {path_of_bmodel}此处为bmodel文件路径
```
也可以参考[1.2 测试命令](#12-测试命令)打印程序运行中的实际性能指标。  
测试中性能指标存在一定的波动属正常现象。

> **测试说明**：  
1. 性能测试的结果具有一定的波动性。
2. 请根据自身具体模型数据`ENGINE_FILE`, `LABEL_FILE `, `YOLO_MASKS`, `YOLO_ANCHORS`, `OUTPUT_TENSOR_CHANNELS`等参数创建 config file *.yml 