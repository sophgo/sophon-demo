# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 测试音频](#22-测试音频)

## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install torch==1.13.1
pip3 install torchaudio==0.13.1
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，请使用发布版本v23.07.01之后的刷机包，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install torch==1.13.1
pip3 install torchaudio==0.13.1
```

## 2. 推理测试
python例程不需要编译，可以直接运行，除了所需的模型、数据集之外，还需要确保`python`目录下存在目标平台对应架构的swig decoder模块，这些文件一般在[准备模型与数据](../README.md#3-准备模型与数据)这一步就会准备完毕。PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明
```
运行wenet.py文件，请注意修改相应的参数：
```bash
usage: wenet.py [--input INPUT_PATH] [--encoder_bmodel ENCODER_BMODEL] [--decoder_bmodel DECODER_BMODEL][--dev_id DEV_ID] [--result_file RESULT_FILE_PATH] [--mode MODE]

--input: 测试数据路径，必须是符合格式要求的数据列表；
--encoder_bmodel: 用于推理的encoder bmodel路径，默认使用stage 0的网络进行推理；
--decoder_bmodel: 用于推理的decoder bmodel路径，默认使用stage 0的网络进行推理；
--dev_id: 用于推理的tpu设备id；
--result_file: 用于保存结果的文件路径；
--mode: 对整句进行解码采用的方式。
```
### 2.2 测试音频
音频测试实例如下，通过传入相应的模型路径参数进行测试即可。
```bash
python3 wenet.py --input ../datasets/aishell_S0764/aishell_S0764.list --encoder_bmodel ../models/BM1684/wenet_encoder_fp32.bmodel --dev_id 0 --result_file ./result.txt --mode ctc_prefix_beam_search
```
默认情况下decoder不开启，如果想要开启decoder重打分，请指定mode和decoder_bmodel参数如下：
```bash
python3 wenet.py --input ../datasets/aishell_S0764/aishell_S0764.list --encoder_bmodel ../models/BM1684/wenet_encoder_fp32.bmodel --decoder_bmodel ../models/BM1684/wenet_decoder_fp32.bmodel --dev_id 0 --result_file ./result.txt --mode attention_rescoring
```
测试结束后，会将预测的文本结果保存在`results.txt`下，同时会打印预测结果、推理时间等信息。
