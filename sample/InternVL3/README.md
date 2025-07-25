# Intern-VL3

## 目录
- [Intern-VL3](#intern-vl3)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
    - [2.1 目录结构说明](#21-目录结构说明)
    - [2.2 SDK特性](#22-sdk特性)
  - [3. 准备模型](#3-准备模型)
    - [3.1 使用提供的模型](#31-使用提供的模型)
    - [3.2 自行编译模型](#32-自行编译模型)
  - [4. 例程测试](#4-例程测试)
  - [5. 程序性能测试](#5-程序性能测试)

## 1. 简介
OpenGVLab推出了InternVL3，一款先进的多模态大语言模型（MLLM）系列，展现出卓越的整体性能。相比InternVL2.5，InternVL3在多模态感知和推理能力上表现更优，同时进一步扩展了多模态能力，涵盖工具使用、图形用户界面（GUI）代理、工业图像分析、三维视觉感知等更多领域。此外，OpenGVLab将InternVL3与Qwen2.5 Chat模型进行了对比，后者对应的预训练基础模型被用作InternVL3语言组件的初始化。得益于原生多模态预训练，InternVL3系列在整体文本性能上甚至优于Qwen2.5系列，原始仓库见[InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B)，[InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B)。

本例程对Intern-VL3 8B/2B模型进行移植，使其可在Sophon BM1684X/BM1688芯片上运行。PCIE模式下，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行。在1684X SoC设备（如SE7、SM7、Airbox等）上，支持在V24.04.01(libsophon_0.5.1)SDK上运行。在1688 SoC设备上（如SE9-16、SM9-16等），支持在v1.7.0及以上SDK上运行。

## 2. 特性

### 2.1 目录结构说明

```bash
├── pics                  # 存放README等说明文档中用到的图片
├── examples              # 测试图片和视频
│   ├── image1.jpg
│   ├── image2.jpg
│   └── red-panda.mp4
├── python                # 存放Python例程及其README
│   ├── config            # 模型的tokenizer
│   ├── internvl3_sail.py # 使用SAIL推理的Python例程
│   ├── preprocess.py
│   ├── README.md
│   └── requirements.txt
└── README.md
```

### 2.2 SDK特性

* 支持BM1684X(x86 PCIe、SoC)、BM1688(SoC)
* LLM语言部分支持INT4，ViT视觉部分支持FP16
* 支持基于SAIL推理的Python例程
  
## 3. 准备模型

### 3.1 使用提供的模型

可以通过以下命令下载我们编译好的模型

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
# 1684x
python3 -m dfss --url=open@sophgo.com:sophon-demo/InternVL3/internvl3-2b-awq_w4bf16_seq2048_bm1684x_1dev_20250616_163646.bmodel #1684x 2b seq_len=2048
python3 -m dfss --url=open@sophgo.com:sophon-demo/InternVL3/internvl3-2b-awq_w4bf16_seq4096_bm1684x_1dev_20250611_185254.bmodel #1684x 2b seq_len=4096
python3 -m dfss --url=open@sophgo.com:sophon-demo/InternVL3/internvl3-8b-awq_w4bf16_seq4096_bm1684x_1dev_20250612_140917.bmodel #1684x 8b seq_len=4096
# 1688
python3 -m dfss --url=open@sophgo.com:sophon-demo/InternVL3/internvl3-2b-awq_w4bf16_seq2048_bm1688_2core_20250617_102708.bmodel #1688 2b seq_len=4096
python3 -m dfss --url=open@sophgo.com:sophon-demo/InternVL3/internvl3-2b-awq_w4bf16_seq4096_bm1688_2core_20250612_141932.bmodel #1688 2b seq_len=4096
```

同时​本例程在`scripts`目录下提供了相关测试图片和视频的下载脚本`download.sh`
```bash
chmod -R +x scripts/
./scripts/download.sh
```

### 3.2 自行编译模型

Intern-VL3模型导出需要依赖[InternVL3官方仓库](https://huggingface.co/OpenGVLab/InternVL3-2B)，目前只支持在x86主机进行模型编译。  

```bash
modelscope download --model OpenGVLab/InternVL3-2B-AWQ --local_dir ./InternVL3-2B-AWQ
modelscope download --model OpenGVLab/InternVL3-8B-AWQ --local_dir ./InternVL3-8B-AWQ
```

完成官方仓库下载后，建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

在tpu-mlir环境中执行一键编译脚本：

``` shell
llm_convert.py -m /workspace/InternVL3-2B-AWQ -s 4096 -q w4bf16 -c bm1684x --out_dir internvl3-2b
```
编译完成后，在指定目录`internvl3-2b`生成`internvl3-2b-xxx.bmodel`和`config`，其中config包含tokenizer和其他原始config。

## 4. 例程测试

- [Python例程](./python/README.md)

## 5. 程序性能测试

这里的测试输入为："介绍图片内容"，测试图片为`examples/image1.jpg`。
|    测试平台   |               测试模型                   |first token latency(s)|token per second(tokens/s)| 
| -----------  | -------------------------------------- | --------------------- | ----------------------- | 
|    SE7-32    | internvl3-2b-awq_w4bf16_seq2048_bm1684x.bmodel  |   0.983               |       25.022             | 
|    SE7-32    | internvl3-2b-awq_w4bf16_seq4096_bm1684x.bmodel  |   2.403               |       17.923             | 
|    SE7-32    | internvl3-8b-awq_w4bf16_seq4096_bm1684x.bmodel  |   6.859               |       7.031             | 
|    SE9-16    | internvl3-2b-awq_w4bf16_seq2048_bm1688.bmodel  |   3.875               |       12.894            | 
|    SE9-16    | internvl3-2b-awq_w4bf16_seq4096_bm1688.bmodel  |   9.646               |       10.793            | 
 
> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. 加载模型和第一次问答可能会比较慢，后续几次就会恢复正常。