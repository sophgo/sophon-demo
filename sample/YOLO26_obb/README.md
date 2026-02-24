# YOLO26_obb

## 目录

- [YOLO26\_obb](#yolo26_obb)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 数据准备与模型编译](#3-数据准备与模型编译)
    - [3.1 数据准备](#31-数据准备)
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
​YOLO26_obb对[​YOLO26官方开源仓库](https://github.com/ultralytics/ultralytics)v8.4.14的yolo26s-obb模型和算法进行移植，使之能在SOPHON BM1684X/BM1688/CV186X上进行推理测试。

## 2. 特性
* 支持BM1688/CV186X(SoC)、BM1684X(x86 PCIe、SoC)
* 支持FP32、FP16模型编译和推理
* 支持基于BMCV预处理的C++推理
* 支持基于OpenCV和BMCV预处理的Python推理
* 支持端到端无 NMS 推理
* 支持DOTA数据集精度测试

## 3. 数据准备与模型编译
### 3.1 数据准备
​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，**如果您希望自己准备模型和数据集，可以跳过本小节，参考[3.2 模型编译](#32-模型编译)进行模型转换。**


```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

`download.sh`默认只下载`datasets`，`models`可以通过指定参数分平台下载，参数如下：
```bash
--all     # 下载所有模型
--BM1684X # 下载BM1684X的bmodel
--BM1688  # 下载BM1688的bmodel
--CV186X  # 下载CV186X的bmodel
--onnx    # 下载onnx
```

```
下载的模型包括：
./models
├── BM1684X # 在BM1684X上运行的模型
│   ├── yolo26s_fp32_1b.bmodel
│   └── yolo26s_fp16_1b.bmodel 
├── BM1688 # 在BM1688上运行的模型
│   ├── yolo26s_fp32_1b.bmodel
│   ├── yolo26s_fp16_1b.bmodel
│   └── yolo26s_fp16_1b_2core.bmodel
├── CV186X # 在CV186X上运行的模型
│   ├── yolo26s_fp32_1b.bmodel
│   └── yolo26s_fp16_1b.bmodel
└── onnx
    ├── yolo26s-obb.onnx         # 导出的onnx模型
    └── yolo26s_qtable_f16       # 编译F16 BModel需要混合精度的层
            
```
下载的数据包括：
```
./datasets
├── test                                      # 测试图片
└── DOTAv1                                      
    ├── images                                # DOTAv1数据集图片
    └── labels                                # DOTAv1数据集标签 
```

## 4. 模型编译
**如果您不编译模型，只想直接使用下载的数据集和模型，可以跳过本小节。**

源模型需要编译成BModel才能在SOPHON TPU上运行，源模型在编译前要导出成onnx模型，具体可参考[YOLO26模型导出](./docs/YOLO26_Export_Guide.md)。​同时，您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X`文件夹下生成转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X/BM1688/CV186X**），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x #bm1688/cv186x
```

​执行上述命令会在`models/BM1684X/`文件夹下生成转换好的FP16 BModel。


## 5. 例程测试
- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法

首先，参考[C++例程](cpp/README.md#32-测试图片)或[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的json文件，注意修改数据集(datasets/DOTAv1/images/val)和相关参数(conf_thresh=0.001)。  
使用`tools`目录下的`eval_DOTA.py`脚本，修改其中的result_file变量内容为上述程序运行后生成的json路径，如：
```python
result_file = "../python/results/yolo26s_fp32_1b.bmodel_val_opencv_python_result.json"
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
python3 setup.py build_ext --inplace
```
注：soc平台（Python版本为3.8.2）可直接使用`scripts/download.sh`下载的`DOTA_devkit_soc`。

需要修改DOTA_devkit仓库下，dota_evaluation_task1.py文件的main()中的detpath、annopath、imagesetfile、classnames这四个变量的值。其中detpath和imagesetfile所对应的是上述eval_DOTA.py生成的TASK1目录下的内容，annopath则是DOTAv1中的ground truth。由于官方检测结果classnames顺序与DOTA默认的classnames顺序不同，因此classnames需要修改为python/utils.py中的DOTA_CLASSES的值。修改示例如下：
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
map: 0.5978870067718212
classaps:  [55.59535284 61.21494625 60.74666923 75.77703974 90.78787879 53.55143066
 59.212333   70.14384665 25.49977581 79.15267052 56.97262351 52.0661157
 42.45299889 61.07331006 52.5835185 ]
```
### 6.2 测试结果
在`datasets/DOTAv1/images/val`数据集上，精度测试结果如下：
| 测试平台 | 测试程序          | 测试模型                | map |
| -------- | ----------------- | ----------------------- | --------------- |
|   SE7-32    | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       | 0.610 |
|   SE7-32    | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       | 0.610 |
|   SE7-32    |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       | 0.592 |
|   SE7-32    |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       | 0.593 |
|   SE7-32    |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       | 0.597 |
|   SE7-32    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       | 0.598 |
|   SE9-16    | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       | 0.610 |
|   SE9-16    | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       | 0.610 |
|   SE9-16    | yolo26_opencv.py  |      yolo26s_fp16_1b_2core.bmodel | 0.610 |
|   SE9-16    |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       | 0.592 |
|   SE9-16    |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       | 0.592 |
|   SE9-16    |  yolo26_bmcv.py   |      yolo26s_fp16_1b_2core.bmodel | 0.592 |
|   SE9-16    |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       | 0.597 |
|   SE9-16    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       | 0.597 |
|   SE9-16    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b_2core.bmodel | 0.597 |
|   SE9-8     | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       | 0.610 |
|   SE9-8     | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       | 0.610 |
|   SE9-8     |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       | 0.592 |
|   SE9-8     |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       | 0.592 |
|   SE9-8     |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       | 0.597 |
|   SE9-8     |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       | 0.597 |

> **测试说明**：  
> 1. 由于sdk版本之间可能存在差异，实际运行结果与本表有<0.01的精度误差是正常的；
> 2. 在搭载了相同TPU和SOPHONSDK的PCIe或SoC平台上，相同程序的精度一致，SE7系列对应BM1684X，SE9系列中，SE9-16对应BM1688，SE9-8对应CV186X；


## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/yolo26s_fp32_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|    测试平台  | 测试模型                        | calculate time(ms) |
| ----------- | ------------------------------- | ------------------ |
| SE7-32          | BM1684X/yolo26s_fp32_1b.bmodel     |          69.69  |
| SE7-32          | BM1684X/yolo26s_fp16_1b.bmodel     |          17.24  |
| SE9-16          | BM1688/yolo26s_fp32_1b.bmodel      |         354.76  |
| SE9-16          | BM1688/yolo26s_fp16_1b.bmodel      |          89.71  |
| SE9-16          | BM1688/yolo26s_fp16_1b_2core.bmodel|          51.63  |
| SE9-8           | CV186X/yolo26s_fp32_1b.bmodel      |         352.09  |
| SE9-8           | CV186X/yolo26s_fp16_1b.bmodel      |          89.57  |


> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为平均每张图片的推理时间；
3. SoC和PCIe的测试结果基本一致。


### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++和Python例程打印的时间已经折算为单张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/DOTAv1/images/val`，conf_thresh=0.001，性能测试结果如下：
| 测试平台 | 测试程序          | 测试模型                | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | ----------------- | ----------------------- | ----------- | --------------- | -------------- | ---------------- |
|   SE7-32    | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       |      95.03      |      65.83      |      77.48      |      0.71                          |
|   SE7-32    | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       |      93.17      |      66.96      |      25.18      |      0.71                          |
|   SE7-32    |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       |      21.85      |      10.95      |      69.45      |      0.69                          |
|   SE7-32    |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       |      21.91      |      10.98      |      16.97      |      0.69                          |
|   SE7-32    |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       |      21.30      |      10.20      |      69.11      |      0.15                          |
|   SE7-32    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       |      21.31      |      10.21      |      16.65      |      0.15                          |
|   SE9-16    | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       |     140.91      |      85.57      |     364.98      |      0.90       |
|   SE9-16    | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       |     139.66      |      84.53      |      99.98      |      0.90       |
|   SE9-16    | yolo26_opencv.py  |   yolo26s_fp16_1b_2core.bmodel    |     130.23      |      84.15      |      61.99      |      0.90       |
|   SE9-16    |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       |      36.88      |      30.82      |     354.51      |      0.88       |
|   SE9-16    |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       |      39.12      |      30.84      |      89.77      |      0.87       |
|   SE9-16    |  yolo26_bmcv.py   |   yolo26s_fp16_1b_2core.bmodel    |      31.32      |      30.83      |      51.68      |      0.87       |
|   SE9-16    |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       |      31.72      |      28.39      |     353.86      |      0.22       |
|   SE9-16    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       |      30.38      |      28.38      |      89.33      |      0.22       |
|   SE9-16    |  yolo26_bmcv.soc  |   yolo26s_fp16_1b_2core.bmodel    |      31.46      |      28.38      |      51.24      |      0.22       |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_fp32_1b.bmodel       |     154.26      |      89.15      |     362.32      |      0.90       |
|    SE9-8    | yolo26_opencv.py  |      yolo26s_fp16_1b.bmodel       |     160.87      |      87.05      |      99.77      |      0.89       |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_fp32_1b.bmodel       |      40.90      |      30.87      |     352.19      |      0.87       |
|    SE9-8    |  yolo26_bmcv.py   |      yolo26s_fp16_1b.bmodel       |      63.90      |      30.82      |      89.54      |      0.95       |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_fp32_1b.bmodel       |      32.19      |      28.40      |     351.68      |      0.22       |
|    SE9-8    |  yolo26_bmcv.soc  |      yolo26s_fp16_1b.bmodel       |      30.41      |      28.38      |      89.08      |      0.22       |

> **测试说明**：  
> 1. 时间单位均为毫秒(ms)，统计的时间均为平均每张图片处理的时间；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE7-32的主控处理器均为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 4. 图片分辨率对解码时间影响较大，推理结果对后处理时间影响较大，不同的测试图片可能存在较大差异，不同的阈值对后处理时间影响较大。 


## 8. FAQ
请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。