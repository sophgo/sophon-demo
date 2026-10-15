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

cpp目录下提供C++例程以供参考使用，说明如下：
| 序号  | C++例程          | 说明                                                       |
| ---- | ---------------- | ---------------------------------------------------------- |
| 1    | sam3_opencv      | 使用OpenCV前处理、BMRT推理、OpenCV后处理，支持文本提示检测+分割 |

> 与Python例程类似，C++例程需要预先准备好9个bmodel（ViT part0-4、neck、text encoder、grounding encoder/decoder）以及后处理权重文件，运行时分别加载。模型准备方式见[准备模型与数据](../README.md#3-数据准备与模型编译)。


## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

## 2. 程序编译
### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd sam3_opencv
mkdir build && cd build
cmake .. && make # 生成sam3_opencv.pcie
cd ..
```

编译完成后，会在sam3_opencv目录下生成sam3_opencv.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/sam3_opencv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make
cd ..
```

编译完成后，会在sam3_opencv目录下生成sam3_opencv.soc。

SoC平台也可以直接在板子上原生编译：

```bash
cd cpp/sam3_opencv
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/opt/sophon/libsophon-current .. && make
cd ..
```

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以SoC模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
usage:./sam3_opencv.soc [params]

        --model_dir (value:../../models/BM1684X_504)
                bmodel directory (vit/ neck/ grounding/ subdirs)
        --bpe (value:./tokenizer/bpe_simple_vocab_16e6.txt.gz)
                BPE vocab file (gzip)
        --post_npz (value:../../models/post_process_weights.npz)
                post-process weights (scoring + box refinement)
        --seg_npz (value:../../models/seg_head_weights.npz)
                mask decoder weights
        --input_image (value:../../datasets/truck.jpg)
                input image path
        --prompt (value:a truck)
                text prompt(s), comma-separated
        --score_thresh (value:0.3)
                score threshold
        --output (value:./sam3_output.jpg)
                output image path
        --dev_id (value:0)
                TPU device id
        --streaming (value:false)
                stream ViT part1-4 bmodels per frame (lower device memory,
                slower; all-resident needs ~2.8G device memory)
        --help (value:0)
                print help information.
```
**注意：** CPP传参与python不同，需要用等于号，例如`./sam3_opencv.soc --prompt="a dog,a truck"`。

### 3.2 测试图片
图片测试实例如下，通过`--prompt`指定文本提示（多个提示用逗号分隔，每个提示独立检测/分割一次）：

```bash
./sam3_opencv.soc --model_dir=../../models/BM1684X_504 --input_image=../../datasets/truck.jpg --prompt="a truck" --output=results/cpp_output.jpg --dev_id=0
```

运行后会在终端打印每个提示的最佳置信度、检测框和mask像素数，并将mask叠加在原图上保存为输出图片：

```
prompt: a truck, score: 0.71, query: 157, box(xyxy): [146,276,1684,830], mask pixels: 510124
```

结果与Python例程一致（truck.jpg最佳score 0.7135 vs Python 0.7102，检测框完全对齐；dog.jpg score 0.6858 vs Python 0.6881）。

设备内存不足（`bm_malloc_device_mem`失败或默认npu heap约2.4G的SE7/SE9-16G等）时，加`--streaming=true`以逐帧加载ViT part1-4，峰值设备内存降至约1.4G：

```bash
./sam3_opencv.soc --model_dir=../../models/BM1684X_504 --input_image=../../datasets/truck.jpg --prompt="a truck" --streaming=true --dev_id=0
```
