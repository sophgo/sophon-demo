# C3D

## 目录

* [C3D](#C3D)
  * [目录](#目录)
  * [1. 简介](#1-简介)
  * [2. 数据集](#2-数据集)
  * [3. 准备模型与数据](#3-准备模型与数据)
  * [4. 模型编译](#4-模型编译)
    * [4.1 生成FP32 BModel](#41-生成fp32-bmodel)
  * [5. 例程测试](#5-例程测试)
    


## 1. 简介
Learning Spatiotemporal Features with 3D Convolutional Networks
https://arxiv.org/abs/1412.0767v4
## 2. 数据集
UCF-101 Action Recognition Dataset

## 3. 准备模型与数据

本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型转换](#4-模型转换)进行模型转换。
```bash
chmod -R +x scripts/
./scripts/download.sh
```
执行后，模型保存在`data/models`，数据集在`data/`
```
下载的模型包括：

BM1684/c3d_fp32_1b.bmodel: 用于BM1684的FP32 BModel，batch_size=1
BM1684/c3d_fp32_4b.bmodel: 用于BM1684的FP32 BModel，batch_size=4
BM1684/c3d_int8_1b.bmodel: 用于BM1684的INT8 BModel，batch_size=1
BM1684/c3d_int8_4b.bmodel: 用于BM1684的INT8 BModel，batch_size=4
BM1684X/c3d_fp32_1b.bmodel: 用于BM1684X的FP32 BModel，batch_size=1
BM1684X/c3d_fp32_4b.bmodel: 用于BM1684X的FP32 BModel，batch_size=4
BM1684/c3d_int8_1b.bmodel: 用于BM1684X的INT8 BModel，batch_size=1
BM1684/c3d_int8_4b.bmodel: 用于BM1684X的INT8 BModel，batch_size=4

下载的数据集包括：
UCF_test_01: UCF101的一个测试子集。
```
本例程在`tools`目录下提供了准备lmdb数据的python脚本，用户可以根据脚本自己准备lmdb量化数据集。
```bash
cd tools
python3 c3d_lmdb.py --input_path ../data/UCF_test_01 
```
执行后，会在data目录下产生c3d_lmdb文件夹，可以作为量化模型使用的数据集。

## 4. 模型编译

trace后的pytorch模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

模型编译前需要安装TPU-NNTC，具体可参考[tpu-nntc环境搭建](../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。安装好后需在tpu-nntc环境中进入例程目录。

### 4.1 生成FP32 BModel

pytorch模型编译为FP32 BModel，具体方法可参考TPU-NNTC开发参考手册(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

本例程在`scripts`目录下提供了编译FP32 BModel的脚本。请注意修改`gen_fp32bmodel.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（支持BM1684和BM1684X），如：

```bash
./scripts/gen_fp32bmodel.sh BM1684X
```

执行上述命令会在`data/models/BM1684X/`下生成`c3d_fp32_1b.bmodel、c3d_fp32_4b.bmodel、`文件，即转换好的FP32 BModel。


### 4.2 生成INT8 BModel

不量化模型可跳过本节。

pytorch模型的量化方法可参考TPU-NNTC开发参考手册(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

本例程在`scripts`目录下提供了量化INT8 BModel的脚本。请注意修改`gen_int8bmodel.sh`中的JIT模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台（支持BM1684和BM1684X），如：

```shell
./scripts/gen_int8bmodel.sh BM1684X
```

上述脚本会在`data/models/BM1684X`下生成`c3d_int8_4b.bmodel、c3d_int8_1b.bmodel`文件，即转换好的INT8 BModel。


## 5. 例程测试
* [C++例程](cpp/README.md)
* [python例程](python/README.md)

**自动化测试：**
确保环境依赖安装完毕，可以使用自动测试脚本进行C++和python例程自动化测试，请注意在执行时指定BModel运行的目标平台（支持BM1684和BM1684X）和TPU的id，**该脚本仅限pcie环境，soc环境需自行修改脚本内命令。** 例如：
```
./scripts/auto_test.sh BM1684X 0
```

执行完毕后，会在cpp/c3d_opencv/build/和python/下生成fp32_1b.log、fp32_4b.log日志文件，可以查看其中的结果信息。