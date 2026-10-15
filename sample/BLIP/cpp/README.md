# C++例程

## 目录

- [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
- [2. 程序编译](#2-程序编译)
    - [2.1 x86/arm/riscv PCIe平台](#21-x86armriscv-pcie平台)
    - [2.2 SoC平台](#22-soc平台)
- [3. 推理测试](#3-推理测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 测试图片](#32-测试图片)

cpp目录下提供C++例程以供参考使用，说明如下：
| 序号  | C++例程     | 说明                                                              |
| ---- | ---------- | ----------------------------------------------------------------- |
| 1    | blip_opencv | 使用opencv预处理（resize+归一化）、BMRT推理，支持图文匹配(itm)、图像字幕(cap)、图文问答(vqa) |

> 与Python例程类似，C++例程需要预先准备好bmodel，运行时加载。模型准备方式见[准备模型与数据](../README.md#4-准备数据与模型)。C++例程自带Bert WordPiece分词器（基于bert-base-uncased词表），无需额外安装transformers。

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台
如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv和sophon-ffmpeg，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

## 2. 程序编译
### 2.1 x86/arm/riscv PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd cpp/blip_opencv
mkdir build && cd build
cmake .. && make # 生成blip_opencv.pcie
cd ..
```
编译完成后，会在blip_opencv目录下生成blip_opencv.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/blip_opencv
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make
cd ..
```
编译完成后，会在blip_opencv目录下生成blip_opencv.soc。

SoC平台也可以直接在板子上原生编译：

```bash
cd cpp/blip_opencv
mkdir build && cd build
cmake -DTARGET_ARCH=soc -DSDK=/opt/sophon/libsophon-current .. && make
cd ..
```

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以SoC模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：

```bash
usage:./blip_opencv.soc [params]

        --task (value:itm)
                sub-task: itm | cap | vqa
        --input (value:<none>)
                input image path
        --text (value:a woman sitting on the beach with a dog,a woman sitting on the beach with a cat)
                comma-separated texts (itm)
        --bmodel (value:../../models/BM1684X/blip_itm_bm1684x_f32_1b.bmodel)
                bmodel path (itm/cap)
        --venc_bmodel (value:../../models/BM1684X/blip_vqa_venc_bm1684x_f32_1b.bmodel)
                vqa image-encoder bmodel
        --tenc_bmodel (value:../../models/BM1684X/blip_vqa_tenc_bm1684x_f32_1b.bmodel)
                vqa text-encoder bmodel
        --tdec_bmodel (value:../../models/BM1684X/blip_vqa_tdec_bm1684x_f32_1b.bmodel)
                vqa text-decoder bmodel
        --vocab (value:../../models/bert-base-uncased/vocab.txt)
                bert-base-uncased vocab.txt
        --dev_id (value:0)
                TPU device id
        --help
                print help information.
```

> **注意：**
> 1. 程序自带Bert WordPiece分词器，`--vocab`指向bert-base-uncased词表目录下的`vocab.txt`文件（非Python例程中的整个tokenizer目录）。
> 2. CV84X6平台使用FP16 BModel，可指定`CV84X6`目录下的bmodel。
> 3. vqa任务为交互式，程序先编码图像，随后在命令行输入问题（输入`exit`退出）。

### 3.2 测试图片

图文匹配（itm）测试实例如下：
```bash
./blip_opencv.soc --task itm --input ../../datasets/test/demo.jpg \
  --text "a woman sitting on the beach with a dog,a woman sitting on the beach with a cat" \
  --bmodel ../../models/BM1684X/blip_itm_bm1684x_f32_1b.bmodel \
  --vocab ../../models/bert-base-uncased/vocab.txt --dev_id 0
```
运行后会在命令行打印每个文本与图像的匹配度：
```
Text: a woman sitting on the beach with a dog, Similarity: 0.99xx
Text: a woman sitting on the beach with a cat, Similarity: 0.00xx
```

图像字幕（cap）测试实例如下：
```bash
./blip_opencv.soc --task cap --input ../../datasets/test/demo.jpg \
  --bmodel ../../models/BM1684X/blip_cap_bm1684x_f32_1b.bmodel \
  --vocab ../../models/bert-base-uncased/vocab.txt --dev_id 0
```
运行后会在命令行打印生成的字幕：
```
Caption: <生成的字幕>
```

图文问答（vqa）测试实例如下：
```bash
./blip_opencv.soc --task vqa --input ../../datasets/test/demo.jpg \
  --venc_bmodel ../../models/BM1684X/blip_vqa_venc_bm1684x_f32_1b.bmodel \
  --tenc_bmodel ../../models/BM1684X/blip_vqa_tenc_bm1684x_f32_1b.bmodel \
  --tdec_bmodel ../../models/BM1684X/blip_vqa_tdec_bm1684x_f32_1b.bmodel \
  --vocab ../../models/bert-base-uncased/vocab.txt --dev_id 0
```
运行后会提示输入问题，可以输入"where is the woman"、"what are they doing"、"what's the color of the clothes"来测试，输入`exit`退出：
```
Image encoded. Enter your question (or 'exit' to quit):
where is the woman
Answer: <生成的回答>
Enter your question (or 'exit' to quit):
exit
```

与Python例程结果对比（测试模型均为BM1684X fp32模型）：

| 测试平台 | 测试程序 | itm匹配结果 | cap字幕 | vqa问答 |
| ------- | ------- | ----------- | ------- | ------- |
| SE7-32  | blip_itm.py / blip_cap.py / blip_vqa.py | 与C++一致 | 与C++一致 | 与C++一致 |
| SE7-32  | blip_opencv (C++) | 与Python一致 | 与Python一致 | 与Python一致 |
