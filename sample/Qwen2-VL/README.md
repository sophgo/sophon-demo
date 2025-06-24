# Qwen2-VL

## 目录
- [Qwen2-VL](#qwen2-vl)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型](#4-准备模型)
    - [4.1 使用提供的模型](#41-使用提供的模型)
    - [4.2 自行导出ONNX模型](#42-自行导出onnx模型)
    - [4.3 自行编译BModel模型](#43-自行编译bmodel模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介
Qwen2-VL是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM）。Qwen2-VL可以以图像、文本、视频作为输入，并以文本作为输出。Qwen2-VL系列模型性能强大，具备多语言对话、多图交错对话、视频理解并对话等能力，Qwen2-VL仓库可见[Qwen2-VL-Chat](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)。

本例程对Qwen2-VL进行移植，使其可在Sophon BM1684X芯片上运行。PCIE模式下，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行。在1684X SoC设备（如SE7、SM7、Airbox等）上，支持在V24.04.01(libsophon_0.5.1)SDK上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性

* 支持BM1684X(x86 PCIe、SoC)
* 支持ONNX导出
* 支持FP16、INT4模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持连续对话
* 支持单一图片、多图片、视频输入
* 支持纯文本对话
* 支持图像Resize
* 支持视频抽帧
* 支持历史信息存储与清理

## 3. 运行环境准备

在PCIe上无需修改内存，以下为soc模式相关：
对于1684X系列设备（如SE7/SM7），都可以通过这种方式完成环境准备，使其满足Qwen2-VL运行条件。首先，确保使用V24.04.01 SDK，可以通过bm_version命令检查SDK版本，如需要升级，可从sophgo.com获取v24.04.01版本SDK，刷机包位于sophon-img-xxx/sdcard.tgz中，参考对应的产品手册进行刷机。

确保SDK版本后，在1684x SoC环境上，参考如下命令修改设备内存。

```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://github.com/sophgo/sophon-tools/releases/download/v24.09.21/memory_edit_v2.10.tar.xz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz #vx.x是版本号
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息
./memory_edit.sh -c -npu 7615 -vpu 2048 -vpp 2048 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```

> **注意：**
> 1. tpu总内存为npu/vpu/vpp三者之和。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html#)

## 4. 准备模型

该模型目前只支持在1684X上运行，已提供编译好的bmodel，LLM为int4, ViT为fp16。其中编译好的BModel上下文长度为1.5k，ONNX为0.5k，若需要自行导出其他上下文长度模型，需要参考[4.2 自行导出ONNX模型](#42-自行导出ONNX模型)和[4.3 自行编译BModel模型](#43-自行编译BModel模型)

### 4.1 使用提供的模型

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本
```bash
└── scripts
    ├── download_bm1684x_bmodel.sh                                           # 通过该脚本下载BM1684X平台的Qwen2-VL的BModel
    ├── download_datasets.sh                                                 # 通过该脚本下载Qwen2-VL的测试数据
    └── download_onnx.sh                                                     # 通过该脚本下载Qwen2-VL的ONNX模型
```

> **注意：**
> 1. 下载BModel之前，应该保证存储空间大于7G (bmodel文件大小)
> 2. 下载ONNX模型之前，应该保证存储空间大于100G（ONNX文件大小）

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
|       └── qwen2-vl-7b_int4_seq1536_1dev.bmodel                              # 使用TPU-MLIR编译，用于BM1684X的Qwen2-VL BModel，上下文长度为1.5k
└── datasets
    ├── images                                                               # 测试图片目录
    └── videos                                                               # 测试视频目录
```

### 4.2 自行导出ONNX模型

Qwen2-VL模型导出需要依赖[transformers官方仓库](https://github.com/huggingface/transformers)，目前只支持在x86主机进行模型导出。  

> **注意:** 
> 1. 导出模型需要保证CPU运行内存至少55G以上，导出的onnx模型需要存储空间68G以上，请确有足够的内存和磁盘空间完成此操作。  

- 首先安装依赖

```bash
pip3 install qwen-vl-utils accelerate torch==2.5.1 transformers==4.45.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- 查看需要修改的`transformers`源码文件路径`{transformers_path}`，执行下面命令会输出`Location`字段

```bash
pip3 show transformers
```

- 修改`transformers`中的部分源码文件以便导出ONNX模型

```bash
cp tools/modeling_qwen2_vl.py {transformers_path}/transformers/models/qwen2_vl/modeling_qwen2_vl.py
```

- 导出ONNX模型需要使用脚本`tools/export_onnx_video.py`，该脚本支持输入如下参数以配置导出的ONNX模型

```bash
usage: tools/export_onnx_video.py [-h] [-m MODEL_PATH] [-d {cpu,cuda}] [-s SEQ_LENGTH] [-vs VISION_SEQ_LENGTH] [-n NUM_THREADS]

--model_path: 模型路径或模型名，默认为Qwen/Qwen2-VL-7B-Instruct
--device: pt模型加载的位置，仅支持cpu、cuda两类，默认为cpu
--seq_length: LLM的上下文最大长度，默认为512
--vision_seq_length: 视觉输入的最大长度，默认为1024，**根据经验，该参数需要被设置为--seq_length数值的两倍**
--num_threads: PyTorch的CPU线程数
```

- 按照默认配置导出ONNX模型，可执行如下命令，参数`seq_length`和`vision_seq_length`可根据实际情况修改

```bash
python3 tools/export_onnx_video.py
```

**注意：根据经验，参数`vision_seq_length`往往是`seq_length`两倍**

最终会在`models/onnx/llm`和`models/onnx/vit`两个目录下分别生成LLM和VIT的ONNX模型。

### 4.3 自行编译BModel模型

Qwen2-VL模型编译需要依赖[TPU-MLIR工具包](https://github.com/sophgo/tpu-mlir)，目前只支持在x86主机进行模型编译。  

> **注意:** 
> 1. 编译模型需要保证运行内存至少15G以上，需要存储空间100G以上，请确有足够的内存完成对应的操作。  

- 模型编译前需要安装最新版本TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)创建并进入docker环境

- 安装好后需在TPU-MLIR环境中进入本例程目录，执行如下命令使用TPU-MLIR将onnx模型编译为BModel。详情可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)相应版本的SDK中获取)。

```bash
cd scripts
./compile.sh --seq_length 512 --vision_seq_length 1024
```

**注意：其中参数`seq_length`和`vision_seq_length`与ONNX参数对应，需保持一样，请根据实际情况修改。**
执行完以后最终的BModel会存储在`models/BM1684X/`。

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

输入`datasets/images/test_frames`测试图片集，原始图片尺寸为1920x1080，缩放到模型能够接受的最大尺寸进行测试，即`max_side`参数为-1，测试问题为："请描述图片中的内容"，测试命令如下

```bash
cd scripts
python3 performance_test.py
```

|    测试平台   |               测试模型                  |    preprocess + tokenize(s)     |   vision inference(s)  | first token latency(s)  |token per second(tokens/s)| 
| -----------  | -------------------------------------- | -------------------------------- | ----------------------- | ---------------------- | ------------------------ |
|    SE7-32    | qwen2-vl-7b_int4_seq1536_1dev.bmodel   |   0.252                          |     3.086               |      2.152             |       9.186              |
 
> **测试说明**：  
> 1. 性能测试结果为多张图片的平均值，具有一定的波动性，且与输入也有关，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 3. 图片或者视频尺寸越大，一般精度越高，直到达到一定尺寸，较大输入需要上下文较长的模型；

