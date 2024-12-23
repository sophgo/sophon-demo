# YOLOv8-obb

## 目录

- [YOLOv8-obb](#yolov8-obb)
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
​YOLOv8是YOLO系列的的一个重大更新版本，它抛弃了以往的YOLO系类模型使用的Anchor-Base，采用了Anchor-Free的思想。YOLOv8建立在YOLO系列成功的基础上，通过对网络结构的改造，进一步提升其性能和灵活性。本例程对[​YOLOv8官方开源仓库](https://github.com/ultralytics/ultralytics)中的yolov8s-obb模型和算法进行移植，使之能在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性

### 2.1 目录结构说明
```bash
├── cpp                   # 存放C++例程及其README
|   ├──README_EN.md     
|   ├──README.md      
|   ├──yolov8_bmcv        # 使用SOPHON-OpenCV解码、BMCV前处理、BMRT推理的C++例程
├── docs                  # 存放本例程专用文档，如ONNX导出、移植常见问题等
├── pics                  # 存放README等说明文档中用到的图片
├── python                # 存放Python例程及其README
|   ├──README_EN.md 
|   ├──README.md 
|   ├──yolov8_opencv.py   # 使用OpenCV解码、OpenCV前处理、SAIL推理的Python例程
|   └──...                # Python例程共用功能的封装。
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译、数据下载、自动测试等shell脚本
└── tools                 # 存放精度测试、性能比对等python脚本
```

### 2.2 SDK特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC、riscv PCIe)
* 支持FP32、FP16模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV预处理的Python推理
* 支持DOTA数据集精度测试
 
## 3. 数据准备与模型编译

### 3.1 数据准备

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**

```bash
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684X
│   ├── yolov8s-obb_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov8s-obb_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
├── BM1688
│   ├── yolov8s-obb_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=1
│   ├── yolov8s-obb_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=1
│   ├── yolov8s-obb_fp32_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP32 BModel，batch_size=1, num_core=2
│   ├── yolov8s-obb_fp16_1b_2core.bmodel   # 使用TPU-MLIR编译，用于BM1688的FP16 BModel，batch_size=1, num_core=2
├── CV186X
│   ├── yolov8s-obb_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP32 BModel，batch_size=1
│   ├── yolov8s-obb_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于CV186X的FP16 BModel，batch_size=1
└── onnx
    └── yolov8s-obb.onnx      # 导出的动态onnx模型
```

下载的数据包括：
```
./datasets
├── test                                      # 测试图片
└── DOTAv1                                      
    ├── images                                # DOTAv1数据集图片
    └── labels                                # DOTAv1数据集标签  
```

### 3.2 模型编译

**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，如果您使用的TPU-MLIR版本>=v1.3.0（即官网v23.07.01），也可以直接使用torchscript模型。具体可参考[YOLOv8_obb模型导出](./docs/YOLOv8_Export_Guide.md)。​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`等文件夹下生成`yolov8s-obb_fp32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`yolov8s-obb_fp16_1b.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 BModel

​本例程在`scripts`目录下提供了量化INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```shell
./scripts/gen_int8bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​上述脚本会在`models/BM1684X`等文件夹下生成`yolov8s-obb_int8_1b.bmodel`等文件，即转换好的INT8 BModel。

## 4. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 5. 精度测试
### 5.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/DOTAv1/images/val)和相关参数(conf_thresh=0.25、nms_thresh=0.7)。  

使用`tools`目录下的`eval_DOTA.py`脚本，修改其中的result_file变量内容为上述程序运行后生成的json路径，如：
```python
result_file = "../python/results/yolov8s-obb_fp16_1b.bmodel_val_opencv_python_result.json"
```

运行上述脚本，将测试生成的json文件转化为DOTA数据集精度计算所需的数据格式，命令如下：
```bash
# 在tools目录下
python3 eval_DOTA.py
```
上述命令执行后，会在tools下创建TASK1目录。目录中的内容是多个Task1_{classname}.txt个文件和一个valset.txt。其中Task1_{classname}.txt只包含对应类别的检测结果，valset.txt则包含所有图片的名称。

下载[DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)到tools中，并编译得到该仓库中python程序需要的c++扩展，命令如下：
```bash
# 在tools目录下载DOTA_devkit
git clone https://github.com/CAPTAIN-WHU/DOTA_devkit.git
# 安装swig
sudo apt-get install swig
# 进入DOTA_devkit目录，编译得到c++扩展
cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

需要修改DOTA_devkit仓库下，dota_evaluation_task1.py文件的main()中的detpath、annopath、imagesetfile、classnames这四个变量的值。其中detpath和imagesetfile所对应的是上述eval_DOTA.py生成的TASK1目录下的内容，annopath则是DOTAv1中的ground truth。由于yolov8官方检测结果classnames顺序与DOTA默认的classnames顺序不同，因此classnames需要修改为python/utils.py中的DOTA_CLASSES的值。修改示例如下：
```python
# 这里的执行目录是DOTA_devkit目录
detpath = r'../TASK1/Task1_{:s}.txt'
annopath = r'../../datasets/DOTAv1/labels/val_original/{:s}.txt' 
imagesetfile = r'../TASK1/valset.txt'

classnames = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']

```
运行dota_evaluation_task1.py以及对应的示例结果如下所示：
```bash
# 在DOTA_devkit目录下执行
python3 dota_evaluation_task1.py
```
```
# ...省略中间过程输出
map: 0.5979079160733564
classaps:  [72.2210693  61.68566016 61.76413838 72.79584505 90.74538823 48.917518
 56.12096637 68.53773973 25.76861659 79.92904457 57.03861268 54.54545455
 33.79482754 62.87837086 50.11862211]
```

### 5.2 测试结果
在`datasets/DOTAv1/images/val`数据集上，参数设置为`nms_thresh=0.7,conf_thresh=0.25`,精度测试结果如下：
|   测试平台    |      测试程序     |               测试模型               | AP@IoU=0.5 |
| ------------ | ---------------- | ---------------------------------- | ---------- |
| SE7-32       | yolov8_opencv.py   | yolov8s-obb_fp32_1b.bmodel               |    0.562 |
| SE7-32       | yolov8_opencv.py   | yolov8s-obb_fp16_1b.bmodel               |    0.562 |
| SE7-32       | yolov8_bmcv.soc    | yolov8s-obb_fp32_1b.bmodel               |    0.550 |
| SE7-32       | yolov8_bmcv.soc    | yolov8s-obb_fp16_1b.bmodel               |    0.550 |
| SE9-16       | yolov8_opencv.py   | yolov8s-obb_fp32_1b.bmodel               |    0.562 |
| SE9-16       | yolov8_opencv.py   | yolov8s-obb_fp16_1b.bmodel               |    0.562 |
| SE9-16       | yolov8_bmcv.soc    | yolov8s-obb_fp32_1b.bmodel               |    0.551 |
| SE9-16       | yolov8_bmcv.soc    | yolov8s-obb_fp16_1b.bmodel               |    0.551 |
| SE9-16       | yolov8_opencv.py   | yolov8s-obb_fp32_1b_2core.bmodel         |    0.562 |
| SE9-16       | yolov8_opencv.py   | yolov8s-obb_fp16_1b_2core.bmodel         |    0.562 |
| SE9-16       | yolov8_bmcv.soc    | yolov8s-obb_fp32_1b_2core.bmodel         |    0.551 |
| SE9-16       | yolov8_bmcv.soc    | yolov8s-obb_fp16_1b_2core.bmodel         |    0.551 |
| SE9-8        | yolov8_opencv.py   | yolov8s-obb_fp32_1b.bmodel               |    0.562 |
| SE9-8        | yolov8_opencv.py   | yolov8s-obb_fp16_1b.bmodel               |    0.562 |
| SE9-8        | yolov8_bmcv.soc    | yolov8s-obb_fp32_1b.bmodel               |    0.551 |
| SE9-8        | yolov8_bmcv.soc    | yolov8s-obb_fp16_1b.bmodel               |    0.551 |
| SRM1-20      | yolov8_opencv.py   | yolov8s-obb_fp32_1b.bmodel               |    0.562 |
| SRM1-20      | yolov8_opencv.py   | yolov8s-obb_fp16_1b.bmodel               |    0.562 |
| SRM1-20      | yolov8_bmcv.pcie   | yolov8s-obb_fp32_1b.bmodel               |    0.550 |
| SRM1-20      | yolov8_bmcv.pcie   | yolov8s-obb_fp16_1b.bmodel               |    0.550 |


> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE5系列对应BM1684，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；
> 3. 源模型在该数据集的精度结果为：0.512，对应的命令`yolo val obb data=DOTAv1.yaml device=cpu split=val save_json=True conf=0.25 iou=0.7`；

## 6. 性能测试
### 6.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684/yolov8s-obb_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                测试模型                  | calculate time(ms) |
| ----------------------------------      | --------------- |
| BM1684X/yolov8s-obb_fp32_1b.bmodel      | 76.6            |
| BM1684X/yolov8s-obb_fp16_1b.bmodel      | 15.0            |
| BM1688/yolov8s-obb_fp32_1b.bmodel  |         426.83  |
| BM1688/yolov8s-obb_fp16_1b.bmodel  |          88.11  |
| BM1688/yolov8s-obb_fp32_1b_2core.bmodel|         224.39  |
| BM1688/yolov8s-obb_fp16_1b_2core.bmodel|          49.41  |
| CV186X/yolov8s-obb_fp32_1b.bmodel  |         436.44  |
| CV186X/yolov8s-obb_fp16_1b.bmodel  |          91.95  |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致；

### 6.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/DOTAv1/images/val`，`conf_thresh=0.25, nms_thresh=0.7`，性能测试结果如下：

|    测试平台  |      测试程序       |               测试模型           |   decode_time   | preprocess_time | inference_time  |postprocess_time | 
| ----------- | ----------------- | ------------------------------- | --------------  | ------------    | -----------     | --------------  |
|   SE7-32    | yolov8_opencv.py  |yolov8s-obb_fp32_1b.bmodel|     129.87      |      65.81      |      86.56      |      40.35      |
|   SE7-32    | yolov8_opencv.py  |yolov8s-obb_fp16_1b.bmodel|     142.89      |      65.00      |      24.94      |      40.66      |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s-obb_fp32_1b.bmodel|      50.94      |      10.12      |      76.71      |      8.66       |
|   SE7-32    |  yolov8_bmcv.soc  |yolov8s-obb_fp16_1b.bmodel|      36.93      |      10.12      |      15.10      |      8.65       |
|   SE9-16    | yolov8_opencv.py  |yolov8s-obb_fp32_1b.bmodel|     164.73      |      87.15      |     439.43      |      44.46      |
|   SE9-16    | yolov8_opencv.py  |yolov8s-obb_fp16_1b.bmodel|     163.24      |      86.07      |     100.31      |      43.88      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-obb_fp32_1b.bmodel|      40.23      |      28.32      |     427.11      |      12.21      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-obb_fp16_1b.bmodel|      30.13      |      28.32      |      88.33      |      12.05      |
|   SE9-16    | yolov8_opencv.py  |yolov8s-obb_fp32_1b_2core.bmodel|     159.57      |      86.44      |     236.79      |      45.33      |
|   SE9-16    | yolov8_opencv.py  |yolov8s-obb_fp16_1b_2core.bmodel|     156.49      |      84.28      |      61.93      |      43.15      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-obb_fp32_1b_2core.bmodel|      52.33      |      28.32      |     224.74      |      12.07      |
|   SE9-16    |  yolov8_bmcv.soc  |yolov8s-obb_fp16_1b_2core.bmodel|      30.08      |      28.32      |      49.68      |      12.17      |
|    SE9-8    | yolov8_opencv.py  |yolov8s-obb_fp32_1b.bmodel|     169.42      |      86.45      |     448.55      |      48.90      |
|    SE9-8    | yolov8_opencv.py  |yolov8s-obb_fp16_1b.bmodel|     169.59      |      86.64      |     103.97      |      49.06      |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s-obb_fp32_1b.bmodel|      46.39      |      29.67      |     436.68      |      12.16      |
|    SE9-8    |  yolov8_bmcv.soc  |yolov8s-obb_fp16_1b.bmodel|      57.61      |      29.67      |      92.13      |      12.02      |
|   SRM1-20   | yolov8_opencv.py  |yolov8s-obb_fp32_1b.bmodel|     193.16      |      72.35      |     128.59      |      181.55     |
|   SRM1-20   | yolov8_opencv.py  |yolov8s-obb_fp16_1b.bmodel|     200.58      |      68.67      |      61.97      |      166.76     |
|   SRM1-20   |  yolov8_bmcv.pcie |yolov8s-obb_fp32_1b.bmodel|      132.98     |       7.75      |      90.56      |      59.50      |
|   SRM1-20   |  yolov8_bmcv.pcie |yolov8s-obb_fp16_1b.bmodel|      127.94     |       7.54      |      17.41      |      60.45      |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 

## 7. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。

