# SSD

## 目录

* [SSD](#SSD)
  * [目录](#目录)
  * [1. 简介](#1-简介)
  * [2. 数据集](#2-数据集)
  * [3. 准备模型与数据](#3-准备模型与数据)
  * [4. 模型编译](#4-模型编译)
    * [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
    * [4.2 生成INT8 BModel](#42-生成int8-bmodel)
  * [5. 例程测试](#5-例程测试)
    


## 1. 简介
SSD300 Object Detect Demos

## 2. 数据集
VOC0712

## 3. 准备模型与数据

您需要准备用于测试的数据集，如果量化模型，还要准备用于量化的数据集。

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。
```bash
pip3 install dfn pycocotools
chmod -R +x scripts/
./scripts/download.sh
```
执行后，模型保存在`data/models`，数据集在`data/`
```
下载的模型包括：

BM1684/ssd300_fp32_1b.bmodel: 用于BM1684的FP32 BModel，batch_size=1
BM1684/ssd300_fp32_4b.bmodel: 用于BM1684的FP32 BModel，batch_size=4
BM1684/ssd300_int8_1b.bmodel: 用于BM1684的INT8 BModel，batch_size=1
BM1684/ssd300_int8_4b.bmodel: 用于BM1684的INT8 BModel，batch_size=4
BM1684X/ssd300_fp32_1b.bmodel: 用于BM1684X的FP32 BModel，batch_size=1
BM1684X/ssd300_fp32_4b.bmodel: 用于BM1684X的FP32 BModel，batch_size=4
BM1684X/ssd300_int8_1b.bmodel: 用于BM1684X的INT8 BModel，batch_size=1
BM1684X/ssd300_int8_4b.bmodel: 用于BM1684X的INT8 BModel，batch_size=4

下载的数据集包括：
images/lmdb: 用于量化的lmdb数据集
videos.mp4: 测试视频
VOC2007-test-images: VOC2007测试集
pascal_test2007.json: VOC2007测试集的ground truth
```


## 4. 模型编译

模型编译前需要安装tpu-nntc，具体可参考[tpu-nntc环境搭建](../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。

### 4.1 生成FP32 BModel

caffe模型编译为FP32 BModel，具体方法可参考[BMNETC 使用](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/nntc/html/usage/bmnetc.html)。

本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`gen_fp32bmodel.sh`中的模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
./scripts/gen_fp32bmodel.sh BM1684X
```

执行上述命令会在`data/models/BM1684X/`下生成`ssd_fp32_1b.bmodel、ssd_fp32_4b.bmodel、`文件，即转换好的FP32 BModel。

### 4.2 生成INT8 BModel

不量化模型可跳过本节。

模型的量化方法可参考[Quantization-Tools User Guide](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/calibration-tools/html/index.html)

本例程在`scripts`目录下提供了编译INT8 BModel的脚本。请注意修改`gen_int8bmodel.sh`中的模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
./scripts/gen_int8bmodel.sh BM1684X
```

执行上述命令会在`data/models/BM1684X/`下生成`ssd_int8_1b.bmodel、ssd_int8_4b.bmodel、`文件，即转换好的INT8 BModel。


## 5. 例程测试
* [C++例程](cpp/README.md)
* [python例程](python/README.md)


