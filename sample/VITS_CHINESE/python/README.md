# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 使用方式](#22-使用方式)

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon以及sophon-mw，除此之外，您还需要编译并安装sophon-sail，以上安装步骤具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您还需要安装其他第三方库，在本例程顶层目录VITS_CHINESE/执行
```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

cd python/monotonic_align
mkdir monotonic_align
python3 setup.py build_ext --inplace
```


### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

此外您还需要安装其他第三方库：
```bash
pip3 install -r python/requirements_soc.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

cd python/monotonic_align
mkdir monotonic_align
python3 setup.py build_ext --inplace

sudo apt update
sudo apt install libsndfile1
```

您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。


## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明

```bash
usage: vits_infer_sail.py [--vits_model VITS_BMODEL] [--bert_model BERT_BMODEL] [--text_file *.txt] [--dev_id DEV_ID]
--vits_model: 用于vits推理的bmodel路径；
--bert_model: 用于bert推理的bmodel路径；
--text_file: 用于推理的文本路径;
--dev_id: 用于推理的tpu设备id；
```

### 2.2 使用方式
- 准备文本数据

您可以运行下载脚本(scripts/download.sh)获得数据集。您也可以自行新建./datasets/vits_infer_item.txt，并在该txt文件写入您所希望转为语音的文字。
- 运行例程

在本例程顶层目录VITS_CHINESE/执行：
```bash
python3 python/vits_infer_sail.py --vits_model ./models/BM1684X/vits_chinese_f16.bmodel --bert_model  ./models/BM1684X/bert_f16_1core.bmodel --text_file ./datasets/vits_infer_item.txt --dev_id 0
```
测试结束后，会将推理得到的音频文件保存在results/下

## 3. 流程图
![alt text](image.png)
