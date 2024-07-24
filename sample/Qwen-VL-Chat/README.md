# Qwen-VL

## 目录
- [Qwen-VL](#qwen-vl)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型](#4-准备模型)
    - [4.1 使用提供的模型](#41-使用提供的模型)
    - [4.2 自行编译模型](#42-自行编译模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介
Qwen-VL是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM）。Qwen-VL可以以图像、文本、检测框作为输入，并以文本和检测框作为输出。Qwen-VL系列模型性能强大，具备多语言对话、多图交错对话等能力，Qwen-VL仓库可见[Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)。

本例程对Qwen-VL进行移植，使其可在Sophon BM1684X芯片上运行。PCIE模式下，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行。在1684X SoC设备（如SE7、SM7、Airbox等）上，支持在V24.04.01(libsophon_0.5.1)SDK上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性

* 支持BM1684X(x86 PCIe、SoC)
* 支持FP16、INT8模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持多轮对话

## 3. 运行环境准备

在PCIe上无需修改内存，以下为soc模式相关：
对于1684X系列设备（如SE7/SM7），都可以通过这种方式完成环境准备，使其满足Qwen-VL运行条件。首先，确保使用V24.04.01刷机包，刷机包可由如下命令获取：

```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-sdk/release/v24.04.01/sophon-img/sdcard.tgz
```

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
./memory_edit.sh -c -npu 7615 -vpu 3072 -vpp 3072 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```

> **注意：**
> 1. tpu总内存为npu/vpu/vpp三者之和。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html#)

SE7默认python版本为3.8.2，可通过如下命令安装本例程依赖的sophon-sail(与python3.8.2对应):

```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen-VL/sophon_arm-3.8.0-py3-none-any.whl
pip3 install sophon_arm-3.8.0-py3-none-any.whl --force-reinstall
```

## 4. 准备模型

该模型目前只支持在1684X上运行，已提供编译好的bmodel，LLM为int8, 1k上下文，ViT为fp16。

### 4.1 使用提供的模型

​本例程在`scripts`目录下提供了下载脚本`download.sh`

**注意：**在运行前，应该保证存储空间大于12G (bmodel文件大小)

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行下载脚本后，`models`目录下的文件如下：

```bash
.
├── BM1684X                                        
│   └── qwen-vl-chat-int8-vit-fp16-1dev.bmodel    # 使用TPU-MLIR编译，用于BM1684X的qwen-vl BModel，上下文长度为1024
└── token_config
    ├── qwen.tiktoken                             # token对照表
    ├── tokenization_qwen.py                      # tokenizer相关函数
    └── tokenizer_config.json                     # tokenizer配置
```

### 4.2 自行编译模型

Qwen-VL模型导出需要依赖[Qwen-VL-Chat官方仓库](https://huggingface.co/Qwen/Qwen-VL-Chat)，目前只支持在x86主机进行模型编译。  

**注意:** 用cpu转模型需要保证运行内存至少52G以上，导出的onnx模型需要存储空间61G以上，请确有足够的内存完成对应的操作。  

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
git lfs install
git clone https://huggingface.co/Qwen/Qwen-VL-Chat
```

如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。  

完成官方仓库下载后，使用本例程的`tools`目录下的`config.json`和`modeling_qwen2.py`直接替换掉原仓库的文件，`config.json`中`seq_length`关键字为上下文长度。

```bash
# /path/to/Qwen-VL-Chat/ 为下载的Qwen-VL官方库的路径
cp tools/config.json /path/to/Qwen-VL-Chat/
cp tools/modeling_qwen.py /path/to/Qwen-VL-Chat/
```

在`tools`文件夹下，运行`export_onnx.py`脚本即可导出onnx/pt模型，并存放在`models`文件夹下，指令如下：

```bash
# 一颗1684X目前最大支持1024上下文长度 + ViT
python3 tools/export_onnx.py --model_path /path/to/Qwen-VL-Chat/ --seq_length 1024
```

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)创建并进入docker环境，安装好后需在TPU-MLIR环境中进入本例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)相应版本的SDK中获取)。

最后参考TPU-MLIR工具的使用方式激活对应的环境，并在`scripts`路径下执行导出脚本，会将`models/onnx/`下的文件转换为bmodel，并将bmodel移入`models/BM1684X`文件夹下。

```bash
cd scripts
./gen_bmodel.sh --mode int8 --name qwen-vl-chat --addr_mode io_alone --seq_length 1024
```

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

这里的测试输入为："请描述图中的内容"
|    测试平台   |               测试模型                   |first token latency(s)|token per second(tokens/s)| 
| -----------  | -------------------------------------- | --------------------- | ----------------------- | 
|    SE7-32    | qwen-vl-chat-int8-vit-fp16-1dev.bmodel |    2.498              |        5.023            | 
 
> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
