# Qwen2.5-VL

## 目录
- [Qwen2.5-VL](#qwen2.5-vl)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型](#4-准备模型)
    - [4.1 使用提供的模型](#41-使用提供的模型)
    - [4.2 自行编译BModel模型](#42-自行编译bmodel模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介
Qwen2.5-VL 是阿里巴巴推出的新一代多模态大语言模型（Multimodal Large Language Model, MLLM），属于通义千问（Qwen）系列的最新成员。支持图像、文本、视频等多种输入模态，具备跨模态理解、推理、生成能力。适用于图像描述、视觉问答（VQA）、文档分析、多模态交互等任务。相比前代模型（Qwen-VL），在推理速度、准确性、多语言支持等方面均有显著提升。Qwen2.5-VL仓库可见[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)。

本例程对Qwen2.5-VL进行移植，使其可在Sophon BM1684X以及BM1688芯片上运行。PCIE模式下，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行。在1684X SoC设备（如SE7、SM7、Airbox等）以及16G版本的1688设备（例如SE9-16）上，支持在V24.04.01(libsophon_0.5.1)SDK上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性

* 支持BM1684X和BM1688(x86 PCIe、SoC)
* 支持INT4模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持连续对话
* 支持单一图片、视频输入
* 支持纯文本对话
* 支持图像Resize
* 支持视频抽帧
* 支持历史信息存储与清理

## 3. 运行环境准备

在PCIe上无需修改内存，以下为soc模式相关：
对于1684X系列设备（如SE7/SM7），都可以通过这种方式完成环境准备，使其满足Qwen2.5-VL运行条件。首先，确保使用V24.04.01 SDK，可以通过bm_version命令检查SDK版本，如需要升级，可从sophgo.com获取v24.04.01版本SDK，刷机包位于sophon-img-xxx/sdcard.tgz中，参考对应的产品手册进行刷机。

确保SDK版本后，在1684x SoC环境上，参考如下命令修改设备内存。

```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz #vx.x是版本号
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息
./memory_edit.sh -c -npu 7615 -vpu 2048 -vpp 2048 #如果是在1688平台上请修改为：./memory_edit.sh -c -npu 10240 -vpu 0 -vpp 3072
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```

> **注意：**
> 1. tpu总内存为npu/vpu/vpp三者之和。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html#)

## 4. 准备模型

该模型目前支持在1684X以及1688上运行，已提供编译好的bmodel。其中编译好的BModel上下文长度为2k，若需要自行编译其他上下文长度模型，需要参考[4.2 自行编译BModel模型](#42-自行编译BModel模型)

### 4.1 使用提供的模型

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本
```bash
└── scripts
    ├── download_bm1684x_bmodel.sh                                        # 通过该脚本下载BM1684X平台的Qwen2.5-VL的BModel
    ├── download_bm1688_bmodel.sh                                         # 通过该脚本下载BM1688平台的Qwen2.5-VL的BModel
    ├── download_datasets.sh                                              # 通过该脚本下载Qwen2.5-VL的测试数据
```

> **注意：**
> 1. 下载BModel之前，应该保证存储空间大于7G (bmodel文件大小)

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download_bm1684x_bmodel.sh
./scripts/download_datasets.sh
```

执行下载脚本后，目录结构如下：

```bash
├── models
|   └── BM1684X                                        
|       └── qwen2.5-vl-3b-instruct-awq_w4bf16_seq2048_bm1684x_1dev_20250428_143625.bmodel            # 使用TPU-MLIR编译，用于BM1684X的Qwen2.5-VL BModel，上下文长度为2k
└── datasets
    ├── images                                                               # 测试图片目录
    └── videos                                                               # 测试视频目录
```
此外我们也提供了编译好的7B模型，可以使用以下命令下载：
```bash
# 用于BM1684X的Qwen2.5-VL-7B的BModel，上下文长度为2k
python3 -m dfss --url=open@sophgo.com:ext_model_information/LLM/LLM-TPU/qwen2.5-vl-7b-instruct-awq_w4bf16_seq2048_bm1684x_1dev_20250428_150810.bmodel
# 用于BM1688的Qwen2.5-VL-7B的BModel，上下文长度为2k
python3 -m dfss --url=open@sophgo.com:ext_model_information/LLM/LLM-TPU/qwen2.5-vl-7b-instruct-awq_w4bf16_seq2048_bm1688_2core_20250428_152052.bmodel
```
### 4.2 自行编译BModel模型

Qwen2.5-VL模型编译需要依赖[transformers官方仓库](https://github.com/huggingface/transformers)和[TPU-MLIR工具包](https://github.com/sophgo/tpu-mlir)，目前只支持在x86主机进行模型编译。  

> **注意:**
> 
>1.编译模型需要保证CPU运行内存至少15G以上，编译的bmodel模型需要存储空间30G以上，请确保有足够的内存和磁盘空间完成此操作。  
>2.由于transformers是在v4.49.0版本开始支持的Qwen2.5-VL，该版本需要Python版本大于等于3.10.0。使用TPU-MLIR工具链提供的docker环境可满足此要求。
>3.推荐源模型使用Huggingface上的AWQ量化版本，编译成bmodel的过程基本没有精度损失。

- 模型编译前需要安装最新版本TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)创建并进入docker环境。

- 进入docker环境后需要安装TPU-MLIR。本例程需要的TPU-MLIR版本较新，这里提供一个whl包供下载安装：
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen2_5_VL/tpu_mlir-1.18b0-py3-none-any.whl
pip3 install tpu_mlir-1.18b0-py3-none-any.whl --force-reinstall
```

- 安装依赖
```bash
pip3 install qwen-vl-utils accelerate torch==2.6.0 transformers==4.49.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
``` 

- 从Huggingface下载源模型`Qwen2.5-VL-3B-Instruct-AWQ`。源模型文件比较大，会花费较长时间，请耐心等待。
```bash
git lfs install
# 下载3B模型：
git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ
# 下载7B模型：
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ
```

- 使用以下命令编译模型生成bmodel。您可以根据自己的需要来配置参数，参数详情可参考[tpu-mlir/README](https://github.com/sophgo/tpu-mlir/blob/master/README_cn.md)。
```bash
llm_convert.py --model_path Qwen2.5-VL-3B-Instruct-AWQ --seq_length 2048 --quantize w4bf16 --chip bm1684x --num_device 1 --out_dir ./models --max_pixels 672,896
```
其中，视觉部分的长度`vision_length = max_pixels // 28 ** 2`，应保证其不超过`seq_length`。编译完成后生成的bmodel模型在当前目录的`models`文件夹。


## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

输入`datasets/images/test_frames`测试图片集，原始图片尺寸为1920x1080，缩放到模型能够接受的最大尺寸进行测试，即max_side参数为-1，测试问题为："请描述图片中的内容"，测试命令如下

```bash
cd tools
python3 performance_test.py
```

|    测试平台   |               测试模型                                          |preprocess + tokenize(s)|  vision inference(s)     | first token latency(s) |token per second(tokens/s)|
| -----------  | ---------------------------------------------------------------| ---------------------   | -----------------------  | -----------------------| -----------------------|
|    SE7-32    | qwen2.5-vl-3b-instruct-awq_w4bf16_seq2048_bm1684x_1dev.bmodel  |          0.192          |        1.128             |        3.791           |        15.467          |
|    SE7-32    | qwen2.5-vl-7b-instruct-awq_w4bf16_seq2048_bm1684x_1dev.bmodel  |          0.195          |        1.200             |        5.343           |        8.016           |
|    SE9-16    | qwen2.5-vl-3b-instruct-awq_w4bf16_seq2048_bm1688_2core.bmodel  |          0.266          |        3.298             |        11.254          |        7.412           |
|    SE9-16    | qwen2.5-vl-7b-instruct-awq_w4bf16_seq2048_bm1688_2core.bmodel  |          0.248          |        3.370             |        26.407          |        4.443           |
> **测试说明**：  
>1. 性能测试结果具有一定的波动性，且与输入也有关，此处结果是对12张照片测试后取的平均值；
>2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
>3. 图片或者视频尺寸越大，一般精度越高，直到达到一定尺寸，较大输入需要上下文较长的模型；