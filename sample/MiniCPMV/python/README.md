# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 使用方式](#22-使用方式)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程           | 说明                                 |
| ---- | ---------------------| -----------------------------------  |
| 1    | minicpmv.py          | 使用SAIL推理                          |


## 1. 环境准备
> **注意：**
> 无论哪个环境，都要求transformers>=4.49.0，该版本要求python版本大于3.10。若不满足，请参考[python3.10安装](../../../docs/FAQ.md#13-se7安装python310)安装。

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon，但是不需要sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

此外您还需要安装其他第三方库：
```bash
pip3 install -r python/requirements.txt
pip3 install decord
```

您还需要安装sophon-sail，可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade #安装dfss依赖
python3 -m dfss --install sail
```

亦可参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#)编译不包含bmcv,sophon-ffmpeg,sophon-opencv的可被Python3接口调用的Wheel文件。

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

此外您还需要安装其他第三方库：
```bash
pip3 install -r python/requirements.txt
```
您还需要安装sophon-sail，可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade #安装dfss依赖
python3 -m dfss --install sail
```
如果whl包无法使用，也可以参考上一小节，下载源码自己编译。

您还需要手动编译安装decord库，具体请参考[手动编译decord](../docs/FAQ.md##手动编译decord)

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明
minicpmv.py使用config/minicpmv.yaml配置文件进行参数配置。

minicpmv.yaml内容如下
```yaml
bmodel_path: ../models/BM1684X/minicpm-v-4-awq_w4bf16_seq2048_bm1684x_1dev_20250915_204204.bmodel   ## 用于推理的bmodel路径
token_path: ./token_config    ## tokenizer目录路径；
dev_ids: 0   ## 用于推理的tpu设备id；
```

### 2.2 使用方式

请注意，在进行脚本运行之前请使用 `ulimit -a` 检查您的 `open files` 选项值是否>=65536，可以通过下面的指令来进行设置，如果是PCIE主机可以设置更大，如：1048576
```bash
ulimit -n 65536
```

确认运行环境准备完毕，通过[download脚本](../scripts/download.sh)下载模型文件后，使用如下命令运行：
```bash
cd python
python3 minicpmv.py --config ./config/minicpmv.yaml
```
在读入模型后会显示"Question:"，然后输入就可以了。模型的回答会出现在"Answer"中。结束对话请输入"exit"或者"q"或者"quit"。
![diagram](../pics/demo.png)

如果需要**web服务或者Openai API接口服务**，请参考[MiniCPM3例程代码实现](../../MiniCPM3/README.md)

