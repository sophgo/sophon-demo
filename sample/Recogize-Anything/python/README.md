# Python例程
- [Python例程](#python例程)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 测试图片](#22-测试图片)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程      | 说明                                |
| ---- | ---------------- | -----------------------------------  |
| 1    | ram_pillow.py   | 使用pillow解码、torch前处理、SAIL推理 |

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台

如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)。或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install -r python/requirements.txt
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install -r python/requirements.txt
```

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明
```bash
usage: ram_pillow.py
       [-h] [--input INPUT] [--bmodel BMODEL] [--dev_id DEV_ID] [--tag_list TAG_LIST] [--tag_list_chinese TAG_LIST_CHINESE]
       [--tag_list_threshold TAG_LIST_THRESHOLD]

options:
  -h, --help            show this help message and exit
  --input INPUT         path of input, must be image directory
  --bmodel BMODEL       path of bmodel
  --dev_id DEV_ID       tpu id
  --tag_list TAG_LIST   path of tag_list
  --tag_list_chinese TAG_LIST_CHINESE
                        path of tag_list_chinese
  --tag_list_threshold TAG_LIST_THRESHOLD
                        path of tag_list_threshold
```

### 2.2 测试图片
图片测试实例如下，支持对整个图片文件夹进行测试。
```bash
cd python
python3 ram_pillow.py --input .../datasets/test --bmodel ../models/BM1684X/ram_fp16_1b.bmodel --dev_id 0
```
测试结束后，会打印预测结果、推理时间等信息。