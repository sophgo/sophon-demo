# CenterNet C++例程

## 目录

[TOC]

## 1. 目录说明

 cpp目录下提供了C++例程以供参考使用，目录结构如下：

```bash
.
centernet_bmcv
├── CMakeLists.txt # cmake编译脚本
├── main.cpp	   # 主程序
├── processor.cpp
├── processor.h
└── README.md
```

## 2. 程序编译

### 2.1 PCIe模式

#### 2.1.1 环境配置

硬件环境：x86平台，并安装了含有1684或1684X的PCIE加速卡

软件环境：libsophon、sophon-mw、sophon-sail，可以通过[算能官网](https://developer.sophgo.com/site/index/material/21/all.html)下载安装对应版本

用户需要前往sophon-sail github仓库拉取sail，在当前平台进行编译，记录编译后目标文件路径，一般为`/opt/sophon/sophon-sail`

#### 2.1.2 程序编译

c++程序运行前需要编译可执行文件，命令如下：

```bash
cd centernet_bmcv
mkdir build && cd build
rm -rf ./*
cmake -DTARGET_ARCH=x86 -DSAIL_PATH=/opt/sophon/sophon-sail ..
make
cd ..
```

运行成功后，会在build上级目录下生成可执行文件，如下：

```bash
centernet_bmcv
├── ......
└── centernet_bmcv.pcie    # 可执行程序
```

#### 2.1.3 测试

可执行程序默认有一套参数，也可以根据具体情况传入对应参数，参数说明如下：

```bash
usage: centernet_bmcv.pcie [params] 

        --bmodel (value:../../data/models/BM1684/centernet_fp32_1b.bmodel)
                bmodel file path
        --conf (value:0.35)
                confidence threshold for filter boxes
        --help (value:true)
                Print help information.
        --image (value:../../data/ctdet_test.jpg)
                input stream file path
        --tpu_id (value:0)
                TPU device id
```

 demo中支持图片测试，按照实际情况传入路径参数即可。另外，模型支持fp32bmodel、int8bmodel，可以通过传入模型路径参数进行测试：

```bash
# 测试图片目标检测,PCIe mode,x86环境下运行
# 1batch
./centernet_bmcv.pcie --bmodel=../../data/models/BM1684/centernet_fp32_1b.bmodel --image=../../data/ctdet_test.jpg
# 执行完毕后，在./results成centernet_result_20xx-xx-xx-xx-xx-xx.jpg格式的图片
# 图片上检测出11个目标

# 4batch
./centernet_bmcv.pcie --bmodel=../../data/models/BM1684/centernet_int8_4b.bmodel --image=../../data/ctdet_test.jpg
# 执行完毕后，在./results生成centernet_result_20xx-xx-xx-xx-xx-xx_bx.jpg格式的图片
# 按照量化结果差异，图片上检测出11-13个目标，均属正常范围
```

### 2.2 SoC模式

#### 2.2.1 环境配置

硬件环境：x86平台（交叉编译）

软件环境：libsophon、sophon-mw、sophon-sail，可以通过[算能官网](https://developer.sophgo.com/site/index/material/21/all.html)下载安装对应版本

编译工具链：aarch64-linux-gnu

以上环境与编译器的配置与安装，请参考[环境配置指南](../../../docs/Environment_Install_Guide.md#4-SoC平台的开发和运行环境搭建)进行。
> 请注意，在进行交叉编译时使用的sail需要按照sophon-sail仓库中的指导编译为arm兼容的格式

#### 2.2.2 程序编译

C++程序运行前需要编译可执行文件，命令如下：

```bash
mkdir build && cd build
rm -rf ./*
cmake -DTARGET_ARCH=arm -DSDK=${path_to_socsdk}/soc-sdk -DSAIL_PATH=${path_to_sail}/build_soc/sophon-sail ..
make
```

运行成功后，会在build上级目录下生成可执行文件，如下：

```bash
centernet_bmcv
├── ......
└── centernet_bmcv.arm    # 可执行程序
```

#### 2.2.3  测试

 demo中支持图片测试，将编译好的可执行文件、bmodel、测试图片传输到soc模式的算能设备中，按照实际情况传入路径参数即可。另外，模型支持fp32bmodel、int8bmodel，可以通过传入模型路径参数进行测试。

- 将以下文件拷贝到盒子中同一个目录中，进行测试
> 若在BM1684设备上运行程序，拷贝路径为：
1. `centernet_bmcv.arm`
2. `sophon-demo/sample/CenterNet/data/models/BM1684/centernet_fp32_1b.bmodel`
3. `sophon-demo/sample/CenterNet/data/models/BM1684/centernet_int8_4b.bmodel`
4. `sophon-demo/sample/CenterNet/data/ctdet_test.jpg`
5. `sophon-demo/sample/CenterNet/data/coco_classes.txt`
> 若在BM1684X设备上运行程序，拷贝路径为：
1. `centernet_bmcv.arm`
2. `sophon-demo/sample/CenterNet/data/models/BM1684X/centernet_fp32_1b.bmodel`
3. `sophon-demo/sample/CenterNet/data/models/BM1684X/centernet_int8_4b.bmodel`
4. `sophon-demo/sample/CenterNet/data/ctdet_test.jpg`
5. `sophon-demo/sample/CenterNet/data/coco_classes.txt`

```bash
# 测试图片目标检测,SoC mode,SoC环境下运行
mkdir results
# 1batch
./centernet_bmcv.arm --bmodel=./centernet_fp32_1b.bmodel --image=./ctdet_test.jpg
# 执行完毕后，在./results生成centernet_result_20xx-xx-xx-xx-xx-xx.jpg格式的图片
# 图片上检测出11个目标

# 4batch
./centernet_bmcv.arm --bmodel=./centernet_int8_4b.bmodel --image=./ctdet_test.jpg
# 执行完毕后，在./results生成centernet_result_20xx-xx-xx-xx-xx-xx-bx.jpg格式的图片
# 按照量化结果差异，图片上检测出11-13个目标，均属正常范围
```
若在SoC执行报错找不到libsail.so，请设置如下环境变量
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib
```