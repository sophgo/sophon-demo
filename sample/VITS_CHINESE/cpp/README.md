# C++例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 程序编译](#2-程序编译)
    * [2.1 x86/arm PCIe平台](#21-x86arm-pcie平台)
    * [2.2 SoC平台](#22-soc平台)
* [3. 推理测试](#3-推理测试)
    * [3.1 参数说明](#31-参数说明)
    * [3.2 使用方式](#32-使用方式)

cpp目录下提供了C++例程以供参考使用，与Python例程不同，C++例程将文本前处理（拼音转换、BERT推理）也一并使用C++实现，无需依赖Python环境：

| 序号  | C++例程          | 说明                                    |
| ---- | ---------------- | --------------------------------------- |
| 1    | vits_infer_bmnn  | 使用BMRT推理，原生WAV输出，纯C++全流程 |

## 1. 环境准备
### 1.1 x86/arm PCIe平台
如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），可以直接使用它作为开发环境和运行环境。您需要安装libsophon、sophon-opencv，具体步骤可参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv运行库包，可直接使用它作为运行环境。通常还需要一台x86主机作为开发环境，用于交叉编译C++程序。

**数据文件准备**：本例程使用`data/pinyin_map.txt`作为汉字到拼音的映射表。该文件由`pypinyin`库导出，已随例程提供（约2万条映射），无需额外准备。


## 2. 程序编译
C++程序运行前需要编译可执行文件。
### 2.1 x86/arm PCIe平台
可以直接在PCIe平台上编译程序：

```bash
cd cpp/vits_infer_bmnn
mkdir build && cd build
cmake .. -DTARGET_ARCH=pcie
make
cd ..
```

编译完成后，会在vits_infer_bmnn目录下生成vits_infer_bmnn.pcie。

### 2.2 SoC平台
通常在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：

```bash
cd cpp/vits_infer_bmnn
mkdir build && cd build
#请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..
make
```
编译完成后，会在vits_infer_bmnn目录下生成vits_infer_bmnn.soc。

## 3. 推理测试
对于PCIe平台，可以直接在PCIe平台上推理测试；对于SoC平台，需将交叉编译生成的可执行文件及所需的模型、测试数据拷贝到SoC平台中测试。测试的参数及运行方式是一致的，下面主要以PCIe模式进行介绍。

### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参。**注意**：C++传参与python不同，需要用等于号，例如`./vits_infer_bmnn.pcie --bmodel=xxx`。

```bash
Usage: vits_infer_bmnn.pcie [params]

        --bmodel (value:../../models/BM1684X/vits_chinese_f16.bmodel)
                VITS bmodel file path
        --bert_model (value:../../models/BM1684X/bert_f16_1core.bmodel)
                BERT bmodel file path
        --text_file (value:../../datasets/vits_infer_item.txt)
                input text file
        --pinyin_map (value:./data/pinyin_map.txt)
                pinyin map file
        --vocab_file (value:../../python/bert/vocab.txt)
                BERT vocab file
        --dev_id (value:0)
                TPU device id
        --help (value:0)
                print help information.
```

### 3.2 使用方式

- 准备文本数据

您可以运行下载脚本(`scripts/download.sh`)获得数据集。您也可以自行新建`./datasets/vits_infer_item.txt`，并在该txt文件写入您所希望转为语音的文字。

- 运行例程

在本例程顶层目录VITS_CHINESE/执行：

```bash
cd cpp/vits_infer_bmnn
./vits_infer_bmnn.pcie --bmodel=../../models/BM1684X/vits_chinese_f16.bmodel --bert_model=../../models/BM1684X/bert_f16_1core.bmodel --text_file=../../datasets/vits_infer_item.txt --dev_id=0
```

测试结束后，会将推理得到的音频文件保存在`results/`下，文件命名格式为`ts_{line_id}_{seg_id}.wav`。

- 程序流程

本C++例程实现端到端的中文语音合成：
1. 读取中文文本，按标点符号切分为短句
2. 汉字转拼音（TONE3格式），拼音转音素序列
3. BERT模型推理，生成字符嵌入（char embeddings）
4. 字符嵌入按音素数量扩展
5. VITS模型推理，生成音频波形
6. 尾端静音去除，输出16-bit PCM WAV文件

程序运行时会输出各阶段耗时统计：
- **preprocess**: 文本前处理 + BERT TPU推理耗时
- **vits inference**: VITS TPU推理耗时
- **postprocess**: CPU侧音频后处理耗时（截断+去静音）