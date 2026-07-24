# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
      - [一张图片占多少Token ?](#一张图片占多少token-)
      - [视频占多少Token ?](#视频占多少token-)
    - [2.2 使用方式](#22-使用方式)

Qwen3.5能够输入单一图片/视频进行对话，python目录下提供了例程，具体情况如下：

| 序号  |  Python例程       |            说明                 |
| ---- | ---------------- | ------------------------------ |
|   1  | qwen3_5.py       | 使用SAIL推理                     |

## 1. 环境准备
> **注意：**
> 无论哪个环境，都要求transformers==5.7.0，该版本要求python版本大于3.10。若不满足，请参考[python3.10安装](../../../docs/FAQ.md#13-se7安装python310)安装。

### 1.1 x86/arm PCIe平台

- 需要**SDK v24.04.01及其以上版本**

- 如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

- 此外您可能还需要安装其他库：

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- 您还需要安装sophon-sail，由于本例程需要的sophon-sail版本较新，可以用如下命令安装sophon-sail。

```bash
python3 -m dfss --install sail
```

### 1.2 SoC平台

- BM1684X 需要**SDK v24.04.01及其以上版本**

  如果您使用BM1684X的SoC平台（如SE7、SM7系列边缘设备），并使用它测试本例程，请使用**SDK V24.04.01及其以上版本**对应的刷机包进行刷机，刷机成功后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

- BM1688 需要**SDK V2.2及其以上版本**

  如果您使用BM1688的SoC平台（如SE9、SM9系列边缘设备），并使用它测试本例程，请使用**SDK V2.2及其以上版本**对应的刷机包进行刷机，刷机成功后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。


- 此外您可能还需要安装其他库：

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
``` 
- 本例程依赖sophon-sail，可直接安装sophon-sail，执行如下命令：

```bash
python3 -m dfss --install sail
```

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

```bash
usage: qwen3_5.py [-h] -m MODEL_PATH [-c CONFIG_PATH] [-vr VIDEO_RATIO] [-d DEVID] [-ll {DEBUG,INFO,WARNING,ERROR}] 

options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        path to the bmodel file
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        path to the processor file
  -vr VIDEO_RATIO, --video_ratio VIDEO_RATIO
                        Set video ratio, default is 0.25
  -d DEVID, --devid DEVID
                        device ID to use
  -ll {DEBUG,INFO,WARNING,ERROR}, --log_level {DEBUG,INFO,WARNING,ERROR}
                        log level, default: INFO, option[DEBUG, INFO, WARNING, ERROR]
```


#### 一张图片占多少Token ?

计算公式 $ token数 = 长 \times 宽 \div 32 \div 32 $
比如768x768尺寸图片占token数为576 token

#### 视频占多少Token ?

本例中视频尺寸默认为图片的1/4，比如768x768情况下取尺寸384x384，也就是每两帧(`temporal_patch_size`)占144个token。

默认每秒1帧。

20秒视频取20帧，总token数为 $ 144 \times 20 \div 2 = 1440 $


### 2.2 使用方式


输入`../datasets/test.jpg`测试图片，测试问题为："请描述图片中的内容"，测试命令如下:
```bash
python3 qwen3_5.py -m ../models/BM1684X/qwen3.5-2b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260415_111517.bmodel -c config/ -d 0
```

```
在Question: 处输入问题，在Image or Video Path: 处输入图片路径（如`test.jpg`）。如果图片路径为空，则进入对话模式。

终端将打印FTL、TPS性能数据，并输出回答结果，接着可进一步对图片或者视频进行提问，输入q即可退出。

> **测试说明**：  
> 1. 图片或者视频尺寸越大，一般精度越高，直到达到一定尺寸，较大输入需要上下文较长的模型。
