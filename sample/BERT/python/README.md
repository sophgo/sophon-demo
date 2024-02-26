# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 测试文本](#22-测试文本)
    - [2.3 命令行测试](#23-命令行测试)
    - [2.4 测试数据集](#24-测试数据集)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | bert_sail.py   | 使用SAIL推理 |

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install bert4torch==0.3.0 packaging==23.2 seqeval==1.2.2
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install bert4torch==0.3.0 packaging==23.2 seqeval==1.2.2
```

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明
```bash
usage: bert_sail.py [--input INPUT] [--bmodel BMODEL] [--dev_id DEV_ID]
                     
--input: 测试数据，可输入文本或者整个文本文件；
--bmodel: 用于推理的bmodel路径，默认使用stage 0的网络进行推理；
--dev_id: 用于推理的tpu设备id；
--if_crf: 是否启用crf层
--dict_path: 预训练模型词典


```
### 2.2 测试文本
文本测试实例如下
```bash
cd python
python3 bert_sail.py --input ../datasets/china-people-daily-ner-corpus/test.txt --bmodel ../models/BM1684/bert4torch_output_fp32_1b.bmodel --dev_id 0 
```
测试结束后，输出结果

### 2.3 测试数据集
文本数据集测试实例如下，支持对整个文本数据集进行测试。
数据集格式应采取以行为分隔，每行包括一个汉字和他的实体类型，以空格分隔
```bash
python3 bert_sail.py --input ../datasets/china-people-daily-ner-corpus/example.test --bmodel ../models/BM1684/bert4torch_output_fp32_1b.bmodel --dev_id 0 
```
测试结束后，预测的结果保存在`results/bert4torch_output_fp32_1b.bmodel_test_sail_python_result.txt`下，同时会打印预测结果、推理时间等信息。


