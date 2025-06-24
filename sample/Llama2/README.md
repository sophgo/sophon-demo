# Llama2

## 目录

- [Llama2](#llama2)
  - [目录](#目录)
  - [1.简介](#1简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型](#4-准备模型)
    - [4.1 使用提供的模型](#41-使用提供的模型)
    - [4.2 自行编译模型](#42-自行编译模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 性能测试](#6-性能测试)

## 1.简介

本例程对[Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)中的模型进行移植，支持在BM1684X上运行。

对于BM1684X，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86/riscv主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性

* 支持BM1684X(x86 PCIe、SoC、riscv PCIe)
* 支持INT4模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持多轮对话

## 3. 运行环境准备
在PCIe上无需修改内存，以下为soc模式相关，如果您遇到了设备内存不足问题如`alloc mem failed`等报错，可以参考如下方式完成内存布局修改：
对于1684X系列设备（如SE7/SM7）和1688/cv186系列设备（SE9-16的8G/16G版本和SE9-8的8G版本）都可以通过这种方式完成环境准备，使得满足Qwen运行条件。参考如下命令修改设备内存。
```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://github.com/sophgo/sophon-tools/releases/download/v24.09.21/memory_edit_v2.10.tar.xz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz #vx.x是版本号
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息

#如果是1684x系列设备，执行以下命令
./memory_edit.sh -c -npu 4096 -vpu 3072 -vpp 3072 #npu也可以访问vpu和vpp的内存，可以自己调整各个heap的大小。
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```
> **注意：**
> 1. tpu总内存为npu/vpu/vpp三者之和。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)


## 4. 准备模型
已提供编译好的bmodel。
### 4.1 使用提供的模型

​本例程在`scripts`目录下提供了下载脚本`download.sh`

```bash
./scripts/download.sh
```

执行后，当前目录下文件如下：
```bash
├── docs
│   └── Llama2_Export_Guide.md
├── models
│   └── BM1684X
│       └── llama_w4f16_seq512_20250121_171104
│           ├── config.json                               #导出onnx时的配置文件。
│           ├── embedding.bin                             #从模型中拆出来的embedding模块，推理时使用cpu加载、计算。
│           ├── llama_w4f16_seq512_20250121_171104.bmodel #在bm1684x上运行的bmodel。
│           └── tokenizer                                 #该模型对应的tokenizer
├── python
│   ├── config
│   ├── llama2.py
│   ├── README.md
│   ├── requirements.txt
│   └── web_demo.py
├── README.md
├── scripts
│   └── download.sh
└── tools
    ├── files
    |   └── llama-2-7b-chat-hf
    └── export_onnx.py
```

### 4.2 自行编译模型

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需**在TPU-MLIR环境中**进入例程目录。

Llama2模型导出需要依赖[Llama2官方仓库](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)，目前只支持在x86主机进行模型编译。  

**注意:** 用cpu转模型需要保证运行内存至少32G以上，导出的onnx模型需要存储空间20G以上，请确有足够的内存完成对应的操作。 

```bash
pip3 install dfss
# llama2-7B
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama2/llama2-7b-torch.zip
unzip llama2-7b-torch.zip
```

完成官方仓库下载后，使用本例程的`tools`目录下的`files/llama-2-7b-chat-hf/*`等文件，直接替换掉原仓库的文件。

```bash
pip install -r python/requirements.txt
cp tools/files/llama-2-7b-chat-hf/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
```

在`tools`文件夹下，运行`export_onnx.py`脚本即可导出onnx模型，指令如下：

```bash
python3 tools/export_onnx.py --model_path torch2onnx/llama-2-7b-chat-hf --seq_length 512 #在models/onnx/llama2-7b下生成onnx文件和embedding.bin文件。
```

在tpu-mlir环境中执行编译脚本，会将`models/onnx/`下的文件转换为bmodel。

```bash
./scripts/gen_bmodel.sh --mode int4 --name llama2-7b --chip bm1684x
mkdir -p models/BM1684X/llama2-7b
mv llama2-7b_bm1684x_int4_1core.bmodel models/BM1684X/llama2-7b
mv models/onnx/llama2-7b/embedding.bin models/BM1684X/llama2-7b #确保embedding.bin和bmodel在同一个文件夹下。
```

编译脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 性能测试

|   测试平台   |     测试程序  |           测试模型                        |first token latency(s) |token per second(tokens/s)| 
| ----------- | ------------ | ----------------------------------------  | --------------------- | ------------------------ | 
| SE7-32      | llama2.py    | llama_w4f16_seq512_20250121_171104.bmodel |  0.731                | 10.250                   |