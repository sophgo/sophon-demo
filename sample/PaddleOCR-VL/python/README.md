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

PaddleOCR-VL能够进行通用OCR、表格识别、公式识别和图表识别，python目录下提供了例程，具体情况如下：

| 序号  |  Python例程       |            说明                 |
| ---- | ---------------- | ------------------------------ |
|   1  | paddleocr_vl.py  | 使用SAIL推理（命令行交互，支持ocr/table/formula/chart四种模式） |

## 1. 环境准备

### 1.1 x86/arm PCIe平台

- 需要**SDK v24.04.01及其以上版本**

- 如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

- 此外您可能还需要安装其他库：

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- 您还需要安装sophon-sail，可以用如下命令安装sophon-sail。

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

运行前请先执行[下载脚本](../scripts/download.sh)下载模型、测试图片和configs（tokenizer等）。

### 2.1 参数说明

```bash
usage: paddleocr_vl.py [-h] -m MODEL_PATH [-c CONFIG_PATH] [-d DEVID] [-s]

options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        path to the bmodel file
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        path to the config directory, default: ../config
  -d DEVID, --devid DEVID
                        device ID to use, default: 0
  -s, --do_sample       if set, generate tokens by sample parameters (requires
                        bmodel compiled with --do_sample)
```

### 2.2 使用方式

输入`../datasets/ocrtest.jpg`测试图片，测试模式为`ocr`，测试命令如下:

```bash
python3 paddleocr_vl.py -m ../models/BM1688/paddleocr-vl_bf16_seq2048_bm1688_1core_static_20260221_195626.bmodel -c ../config -d 0
```

```
在Mode处输入模式（ocr/table/formula/chart），在Image Path处输入图片路径，终端将打印FTL、TPS性能数据和识别结果，输入q即可退出。
```

> **测试说明：**
> 1. 图片在推理前会缩放到 784x784（编译模型时`--max_pixels 784,784`的固定分辨率）；
> 2. decode阶段各层的中间hidden与KV cache均使用应用侧分配的device tensor（同`Qwen2_5-VL`例程的接线方式），每层launch后用`sync_d2d`把新token的K/V写入cache（设备侧拷贝，异步排在launch之后），不做主机侧读回；
> 3. 命令行 `input()` 一次只读取一行，粘贴多行长文本会串行。
