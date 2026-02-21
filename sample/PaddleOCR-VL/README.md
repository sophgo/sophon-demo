# PaddleOCR-VL

## 目录
- [PaddleOCR-VL](#paddleocr-vl)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型](#3-准备模型)
    - [4.1 使用提供的模型](#41-使用提供的模型)
    - [4.2 自行编译BModel模型](#42-自行编译bmodel模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介
本例程对PaddleOCR-VL-0.9B在SOPHON BM1684X和BM1688处理器上进行移植。源仓库：[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)。
PCIE模式下，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行。在1684X SoC设备（如SE7、SM7、Airbox等）以及16G版本的1688设备（例如SE9-16）上，支持在V24.04.01(libsophon_0.5.1)SDK上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性

* 支持BM1684X和BM1688(x86 PCIe、SoC)
* 支持cpp例程

## 3. 准备模型

该模型目前支持在1684X以及1688上运行，已提供编译好的bmodel。其中编译好的BModel上下文长度为2k，若需要自行编译其他上下文长度模型，需要参考[4.2 自行编译BModel模型](#42-自行编译BModel模型)

### 4.1 使用提供的模型

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本
```bash
└── scripts
    ├── download.sh                                        # 通过该脚本下载PaddleOCR-VL的BModel、测试数据、动态链接库文件
```

> **注意：**
> 1. 下载BModel之前，应该保证存储空间大于7G (bmodel文件大小)

```bash
chmod -R +x scripts/
./scripts/download.sh
```

执行下载脚本，将所有的模型都下载后，目录结构如下：

```bash
├── models #测试模型
|   ├── BM1684X
|   |   ├── paddleocr-vl_bf16_seq2048_bm1684x_1dev_20260206_230325.bmodel
|   └── BM1688
|       └── paddleocr-vl_bf16_seq2048_bm1688_1core_20260207_125123.bmodel
└── datasets  #测试图片
```

### 4.2 自行编译BModel模型

PaddleOCR-VL模型编译需要依赖[transformers官方仓库](https://github.com/huggingface/transformers)和[TPU-MLIR工具包](https://github.com/sophgo/tpu-mlir)，目前只支持在x86主机进行模型编译。  

> **注意:**
> 
>1.编译模型需要保证CPU运行内存至少15G以上，编译的bmodel模型需要存储空间10G以上，请确保有足够的内存和磁盘空间完成此操作。  
>2.由于本例程使用的transformers版本需要Python版本大于等于3.10.0。使用TPU-MLIR工具链提供的docker环境可满足此要求。

- 模型编译前需要安装最新版本TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)创建并进入docker环境。

- 进入docker环境后需要安装TPU-MLIR。本例程需要的TPU-MLIR版本较新，这里提供一个whl包供下载安装：
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/PaddleOCR-VL/paddleocr-vl-0.9B/tpu_mlir-1.26b0-py3-none-any.whl
pip3 install tpu_mlir-1.26b0-py3-none-any.whl --force-reinstall
```

- 安装依赖
```bash
pip3 install accelerate transformers==4.51.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
``` 

- 从ModelScope下载`PaddleOCR-VL`

(比较大，会花费较长时间)

``` shell
git clone https://huggingface.co/PaddlePaddle/PaddleOCR-VL
```

- 编译模型生成bmodel

``` shell
# 如果有提示transformers/torch版本问题，pip3 install transformers torch torchvision -U
llm_convert.py --model_path /workspace/open-source/PaddleOCR-VL/ -q bf16 -g 64 -c bm1688 --num_core 1 --do_sample -s 2048 --max_pixels 784,784
```
编译完成后，在`tmp/`生成`paddleocr-vl-xxx.bmodel`和`config`


## 5. 例程测试

- [cpp例程](./cpp/README.md)

## 6. 程序性能测试

使用`datasets/ocrtest.jpg`作为测试图片，测试模式为`ocr`:


|    测试平台   |               测试模型                                          | first token latency(s) |token per second(tokens/s)|
| -----------  | ---------------------------------------------------------------| ---------------------   | -----------------------  |
|    SE9-16    | ../../paddleocr-vl_bf16_seq2048_bm1688_1core_static_20260221_195626.bmodel  |        6.499          |          22.7494         |