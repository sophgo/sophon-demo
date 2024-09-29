# Intern-VL2

## 目录
- [Intern-VL2](#intern-vl2)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 准备模型](#3-准备模型)
    - [3.1 使用提供的模型](#31-使用提供的模型)
    - [4.2 自行编译模型](#42-自行编译模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介
Intern-VL2是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，原始仓库见[InternVL2-4B](https://huggingface.co/OpenGVLab/InternVL2-4B)，[InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B)。

本例程对Intern-VL2 4B/2B模型进行移植，使其可在Sophon BM1684X/BM1688芯片上运行。PCIE模式下，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行。在1684X SoC设备（如SE7、SM7、Airbox等）上，支持在V24.04.01(libsophon_0.5.1)SDK上运行。在1688 SoC设备上（如SE9-16、SM9-16等），支持在v1.7.0及以上SDK上运行。

## 2. 特性

### 2.1 目录结构说明

```bash
├── pics                  # 存放README等说明文档中用到的图片
├── python                # 存放Python例程及其README
|   ├──README.md 
|   ├──internvl2_sail.py  # 使用SAIL推理的Python例程
|   ├──image1.jpg         # 测试图片
|   ├──requirements.txt   # 运行环境上需要安装的第三方依赖
|   ├──token_config_2b    # 2b模型的tokenizer
|   └──token_config_4b    # 4b模型的tokenizer
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译等shell脚本
└── tools                 # 存放onnx导出等python脚本
```

### 2.2 SDK特性

* 支持BM1684X(x86 PCIe、SoC)、BM1688(SoC)
* LLM语言部分支持INT4，ViT视觉部分支持FP16
* 支持基于SAIL推理的Python例程
  
## 3. 准备模型

### 3.1 使用提供的模型

可以通过以下命令下载我们编译好的模型。

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internvl2-4b_bm1684x_int4.bmodel #1684x 4b
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/internvl2-2b_bm1688_int4_2core.bmodel #1688 2b
```

### 4.2 自行编译模型

Intern-VL2模型导出需要依赖[InternVL2官方仓库](https://huggingface.co/OpenGVLab/InternVL2-4B)，目前只支持在x86主机进行模型编译。  

**注意:** 用cpu转模型需要保证运行内存至少32G以上，导出的onnx模型需要存储空间20G以上，请确有足够的内存完成对应的操作。  

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL2-4B
git clone https://huggingface.co/OpenGVLab/InternVL2-2B
```

如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。  

完成官方仓库下载后，使用本例程的`tools`目录下的`files/InternVL2-4B/*`等文件，直接替换掉原仓库的文件。

```bash
# /path/to/InternVL2-4B/ 为下载的InternVL2-4B官方仓库的路径
cp tools/files/InternVL2-4B/* /path/to/InternVL2-4B/
cp tools/files/InternVL2-2B/* /path/to/InternVL2-2B/
```

在`tools`文件夹下，运行`export_onnx.py`脚本即可导出onnx模型，并存放在`models/onnx`中的`internvl2-4b`和`internvl2-2b`文件夹下，指令如下：

```bash
python3 tools/export_onnx.py --model_path /path/to/InternVL2-4B
python3 tools/export_onnx.py --model_path /path/to/InternVL2-2B
```

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

在tpu-mlir环境中执行编译脚本，会将`models/onnx/`下的文件转换为bmodel。

```bash
cd scripts
./gen_bmodel.sh --mode int4 --name internvl2-4b --chip bm1684x #会在当前目录下生成internvl2-4b_bm1684x_int4_1core.bmodel
./gen_bmodel.sh --mode int4 --name internvl2-2b --chip bm1688  #会在当前目录下生成internvl2-2b_bm1688_int4_2core.bmodel
```

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

这里的测试输入为："please describe this image in detail."
|    测试平台   |               测试模型                   |first token latency(s)|token per second(tokens/s)| 
| -----------  | -------------------------------------- | --------------------- | ----------------------- | 
|    SE7-32    | internvl2-4b_bm1684x_int4.bmodel       |   0.365               |       27.092            | 
 
> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. 加载模型和第一次问答可能会比较慢，后续几次就会恢复正常。