# Qwen3-VL

## 目录
- [Qwen3-VL](#Qwen3-VL)
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
Qwen3-VL 是阿里巴巴推出的新一代多模态大语言模型（Multimodal Large Language Model, MLLM），属于通义千问（Qwen）系列的最新成员。支持图像、文本、视频等多种输入模态，具备跨模态理解、推理、生成能力。适用于图像描述、视觉问答（VQA）、文档分析、多模态交互等任务。相比前代模型（Qwen-VL），在推理速度、准确性、多语言支持等方面均有显著提升。Qwen3-VL仓库可见[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)。

本例程对Qwen3-VL进行移植，使其可在Sophon BM1684X以及BM1688芯片上运行。PCIE模式下，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行。在1684X SoC设备（如SE7、SM7、Airbox等）以及16G版本的1688设备（例如SE9-16）上，支持在V24.04.01(libsophon_0.5.1)SDK上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

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
* 支持动态模型

## 3. 运行环境准备

在PCIe上无需修改内存，以下为soc模式相关：
对于1684X系列设备（如SE7/SM7），都可以通过这种方式完成环境准备，使其满足Qwen3-VL运行条件。首先，确保使用V24.04.01 SDK，可以通过bm_version命令检查SDK版本，如需要升级，可从sophgo.com获取v24.04.01版本SDK，刷机包位于sophon-img-xxx/sdcard.tgz中，参考对应的产品手册进行刷机。

确保SDK版本后，在1684x SoC环境上，参考如下命令修改设备内存。

```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://github.com/sophgo/sophon-tools/releases/download/v24.09.21/memory_edit_v2.10.tar.xz
tar xvf memory_edit_v2.10.tar.xz
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
    ├── download_bmodel.sh                                        # 通过该脚本下载Qwen3-VL的BModel
```

> **注意：**
> 1. 下载BModel之前，应该保证存储空间大于7G (bmodel文件大小)

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download_bmodel.sh all # 提供了四种all|bm1684x_4b|bm1684x_8b|bm1688模型的下载
```

执行下载脚本，将所有的模型都下载后，目录结构如下：

```bash
├── models
|   ├── BM1684X
|   |   ├── qwen3-vl-8b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_145323.bmodel
|   |   └── qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_141347.bmodel            # 使用TPU-MLIR编译，用于BM1684X的Qwen3-VL BModel，上下文长度为2k
|   └── BM1688
|       └── qwen3-vl-4b-instruct_w4bf16_seq2048_bm1688_2core_20251026_141708.bmodel
└── datasets
    ├── images                                                               # 测试图片目录
    └── videos                                                               # 测试视频目录
```

### 4.2 自行编译BModel模型

Qwen3-VL模型编译需要依赖[transformers官方仓库](https://github.com/huggingface/transformers)和[TPU-MLIR工具包](https://github.com/sophgo/tpu-mlir)，目前只支持在x86主机进行模型编译。  

> **注意:**
> 
>1.编译模型需要保证CPU运行内存至少15G以上，编译的bmodel模型需要存储空间30G以上，请确保有足够的内存和磁盘空间完成此操作。  
>2.由于transformers是在v4.49.0版本开始支持的Qwen3-VL，该版本需要Python版本大于等于3.10.0。使用TPU-MLIR工具链提供的docker环境可满足此要求。
>3.推荐源模型使用Huggingface上的AWQ量化版本，编译成bmodel的过程基本没有精度损失。

- 模型编译前需要安装最新版本TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)创建并进入docker环境。

- 进入docker环境后需要安装TPU-MLIR。本例程需要的TPU-MLIR版本较新，这里提供一个whl包供下载安装：
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen3_VL/tpu_mlir-1.18b0-py3-none-any.whl
pip3 install tpu_mlir-1.18b0-py3-none-any.whl --force-reinstall
```

- 安装依赖
```bash
pip3 install qwen-vl-utils accelerate torch==2.6.0 transformers==4.57.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
``` 

- 从ModelScope下载`Qwen3-VL-4B-Instruct`

(比较大，会花费较长时间)

``` shell
# 下载4B模型
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir Qwen3-VL-4B-Instruct

# 如果想用8B模型，如下：
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir Qwen3-VL-8B-Instruct
```

- 编译模型生成bmodel

``` shell
# 如果有提示transformers/torch版本问题，pip3 install transformers torchvision -U
# 这里max_input_length指定最大输入长度，如果不指定则为-s指定的长度
llm_convert.py -m /workspace/Qwen3-VL-4B-Instruct  -s 2048 --max_input_length 1024  --quantize w4bf16  -c bm1684x --out_dir qwen3vl_4b  --max_pixels 768,768
```
编译完成后，在指定目录`qwen3vl_4b`生成`qwen3-vl-xxx.bmodel`和`config`


## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

输入`../datasets/images/test.jpg`测试图片，，测试问题为："请描述图片中的内容"，测试命令如下:

```bash
python qwen3_vl.py -m /models/BM1684X/qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_141347.bmodel -c config/ -d 0
```


|    测试平台   |               测试模型                                          | first token latency(s) |token per second(tokens/s)|
| -----------  | ---------------------------------------------------------------| ---------------------   | -----------------------  |
|    SE7-32    | qwen3-vl-8b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_145323.bmodel  |        1.711           |        6.643          |
|    SE7-32    | qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_141347.bmodel  |        1.645           |        10.718           |
|    SE9-16    | qwen3-vl-4b-instruct_w4bf16_seq2048_bm1688_2core_20251026_141708.bmodel  |        5.411          |          5.486         |

> **测试说明**：  
>1. 性能测试结果具有一定的波动性，且与输入也有关，此处结果是对12张照片测试后取的平均值；
>2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
>3. 图片或者视频尺寸越大，一般精度越高，直到达到一定尺寸，较大输入需要上下文较长的模型；