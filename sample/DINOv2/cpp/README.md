# C++例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 程序编译](#2-程序编译)
    * [2.1 x86/arm/riscv PCIe平台](#21-x86armriscv-pcie平台)
    * [2.2 SoC平台](#22-soc平台)
* [3. 推理测试](#3-推理测试)
    * [3.1 参数说明](#31-参数说明)
    * [3.2 测试图片](#32-测试图片)

cpp目录下提供了一系列C++例程以供参考使用，具体情况如下：
| 序号  | C++例程      | 说明                                 |
| ---- | ------------- | -----------------------------------  |
| 1    | dinov2_bmcv   | 使用OpenCV解码、BMCV前处理、BMRT推理   |

> **注：** 本 C++ 例程直接使用 `bm_wrapper.hpp` 暴露的 bmruntime 底层接口（`bmrt_*`/`bm_*`/`bm_tensor_t`），不依赖 `bmnn_utils.h` 的 `BMNNContext/BMNNNetwork/BMNNTensor` 封装。

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon和sophon-opencv，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon和sophon-opencv运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

## 2. 程序编译
C++程序运行前需要编译可执行文件。

### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd cpp/dinov2_bmcv
mkdir build && cd build
cmake .. && make # 生成dinov2_bmcv.pcie
cd ..
```
编译完成后，会在dinov2_bmcv目录下生成dinov2_bmcv.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon和sophon-opencv运行库包（解码用 sophon-opencv 的设备侧 `imread`，不依赖 sophon-ffmpeg）。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/dinov2_bmcv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make
cd ..
```
编译完成后，会在dinov2_bmcv目录下生成dinov2_bmcv.soc。

> **必须按目标盒子分开编译（SDK 与编译环境的 glibc 都要匹配）**：SE7-32（BM1684X，libsophon 0.5.3）与 SE9-16 / SE9-8（libsophon 0.4.14）的 opencv soname 和编译环境 glibc 都不同，**一个 `dinov2_bmcv.soc` 不能同时在三台盒子上运行**，请分别为两类平台编译：
>
> | 目标平台 | 交叉编译环境 | 配套 SDK |
> | --- | --- | --- |
> | SE7-32（BM1684X） | Ubuntu 20.04（glibc 2.31，gcc 9.4） | SDK-26.03.01 `soc-sdk`（libsophon 0.5.3） |
> | SE9-16（BM1688）/ SE9-8（CV186X） | Ubuntu 22.04（glibc 2.35，gcc 11.4） | SE9 `se9-soc-sdk`（libsophon 0.4.14） |
>
> 两者混用的典型报错：用 SE9 SDK 编出的程序在 SE7-32 上启动会报 `libyuv.so.1: cannot open shared object file`（opencv soname 不同）；用 Ubuntu 22.04 环境编出的程序在 SE7-32 上会报 `GLIBC_2.34 not found`（编译环境 glibc 高于目标系统）。本仓验证时两类平台分别在 Ubuntu 20.04 / 22.04 交叉编译容器内编译，容器与宿主机挂载路径一致，同一份源码在两个容器里各编一次即可。

## 3. 推理测试
本版本在 SoC 上实测：交叉编译得到 `dinov2_bmcv.soc` 后，将可执行文件连同 `models/`、`datasets/` 拷贝到盒子运行。PCIe 平台的编译/运行方式相同，把 `.soc` 换成 `.pcie` 即可（无需拷贝到盒子）；本版本的 PCIe 路径未实测。下面以 SoC 为例。

模型输出的是官方线性探针特征 `concat([CLS], mean(patch_tokens))`（维度 1536），例程在CPU侧再接官方线性头 `dinov2_vitb14_reg4_linear_head.bin`（`weight[1000,1536]` + `bias[1000]`，二进制按行主序拼接）得到 1000 类 logits，最后 softmax 取 Top-1。

解码使用 sophon-opencv 的设备侧 `cv::imread(path, cv::IMREAD_COLOR, dev_id)`（JPU 硬解），再用 `cv::bmcv::toBMI()` 转成 `bm_image`，因此不依赖 sophon-ffmpeg；`cv::Mat` 会在整批推理结束（`bm_image_destroy`）后才释放。

前处理对齐官方 `make_classification_eval_transform()` 的默认参数：`Resize(短边=256)` -> `CenterCrop(224)` -> `(x/255 - mean) / std`，全部走 BMCV（`bmcv_image_vpp_convert` + `bmcv_image_crop` + `bmcv_image_convert_to`）；VPP 不支持 BICUBIC，故 resize 用 `BMCV_INTER_LINEAR` 替代。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
usage:./dinov2_bmcv.soc [params]

        --bmodel (value:../../models/BM1684X/dinov2_vitb14_reg_fp32_1b.bmodel)
                bmodel file path
        --head (value:../../datasets/dinov2_vitb14_reg4_linear_head.bin)
                linear head bin file path
        --dev_id (value:0)
                TPU device id
        --help (value:0)
                print help information.
        --input (value:../../datasets/imagenet_val_1k/img)
                input path, image file or image directory
```
**注意：** CPP传参与python不同，需要用等于号，例如`./dinov2_bmcv.soc --bmodel=xxx`。

### 3.2 测试图片
图片测试实例如下，支持单张图片或整个图片文件夹。示例按 BM1684X 编写，在 SE9-16 / SE9-8 上请把 `models/BM1684X/` 换成 `models/BM1688/` 或 `models/CV186X/`（文件名相同）。
```bash
# 测试整个文件夹
./dinov2_bmcv.soc --input=../../datasets/imagenet_val_1k/img --bmodel=../../models/BM1684X/dinov2_vitb14_reg_fp32_1b.bmodel --head=../../datasets/dinov2_vitb14_reg4_linear_head.bin --dev_id=0

# 测试单张图片
./dinov2_bmcv.soc --input=../../datasets/imagenet_val_1k/img/ILSVRC2012_val_00000001.JPEG --bmodel=../../models/BM1684X/dinov2_vitb14_reg_fp32_1b.bmodel --head=../../datasets/dinov2_vitb14_reg4_linear_head.bin --dev_id=0
```
**测试整个文件夹**时，会把每张图的预测结果保存到**当前工作目录**的 `results/dinov2_vitb14_reg_fp32_1b.bmodel_img_bmcv_cpp_result.json`（按上面的命令在 `cpp/dinov2_bmcv` 里运行时，即 `cpp/dinov2_bmcv/results/`）；**测试单张图片**时只在终端打印预测类别与分数，**不写 json 文件**。两种方式都会打印推理/解码/预处理耗时统计。
