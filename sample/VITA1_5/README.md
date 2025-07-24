# VITA1.5

## 目录
- [VITA1.5](#vita15)
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
源仓库：[VITA](https://github.com/VITA-MLLM/VITA)。

本例程对VITA-1.5中的模型进行移植，使其可在Sophon BM1684X芯片上运行。PCIE模式下，该例程支持在V24.04.01(libsophon_0.5.1)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行。在1684X SoC设备（如SE7、SM7、Airbox等）上，支持在V24.04.01(libsophon_0.5.1)SDK上运行。

## 2. 特性

### 2.1 目录结构说明

```bash
├── python                # 存放Python例程及其README
|   ├──README.md 
|   ├──vita.py          # 使用SAIL推理的Python例程
|   ├──train.yaml       # asr模型的配置文件
|   ├──vita_tts         # vita的tts模块代码
|   ├──requirements.txt   # 运行环境上需要安装的第三方依赖
├── README.md             # 本例程的中文指南
├── scripts               # 存放模型编译等shell脚本
└── tools                 # 存放onnx导出等python脚本
```

### 2.2 SDK特性

* 支持BM1684X(x86 PCIe、SoC)
* LLM语言部分使用INT4，ASR、VIT、TTS部分使用FP16。
* 支持基于SAIL推理的Python例程
* 支持文本、语音、图片输入，支持输出保存为音频
  
## 3. 准备模型

### 3.1 使用提供的模型

可以通过以下命令下载我们编译好的模型和对应的processor，以及测试图片和测试音频。

```bash
./scripts/download.sh 

#该脚本会下载模型、processor、数据集等文件，会直接放在python/目录下：
python
├── ...
├── datasets
│   ├── q1.wav
│   ├── q2.wav
│   └── vita_newlog.jpg
├── codec_bm1684x_fp16_1core.bmodel # vita_tts的codec部分模型
├── vita-Qwen2_bm1684x_int4_1core.bmodel # vita llm部分模型，包括vit和asr
└── vqvae_fp16_1b.bmodel # vita_tts的vqvae部分模型
```

### 4.2 自行编译模型

VITA模型导出onnx需要依赖[VITA官方仓库](https://modelscope.cn/models/modelscope/NJU_VITA-1.5)，目前只支持在x86主机进行模型编译。  

**注意:** 用cpu转模型需要保证运行内存至少64G以上，需要存储空间60G以上，请确有足够的内存完成对应的操作。  

```bash
pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple #安装导出模型所需依赖
pip3 install modelscope
modelscope download --model modelscope/NJU_VITA-1.5 --local_dir ./NJU_VITA-1.5
```

在`tools`文件夹下，运行`model_export.py`脚本即可导出onnx模型，指令如下：

```bash
cd tools
python3 model_export.py --model_path /path/to/NJU_VITA-1.5 --seq_length 512 --out_dir ./tmp
```

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录，并使用本例程提供的脚本将onnx模型编译为BModel。脚本中命令的详细说明可参考《TPU-MLIR开发手册》(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

在tpu-mlir环境中执行编译脚本，会将`models/onnx/`下的文件转换为bmodel。

```bash
cd scripts
./gen_llm_bmodel.sh --mode int4 --chip bm1684x --onnx_dir ../tools/tmp/onnx_seq512 #会在当前目录下生成vita-Qwen2_bm1684x_int4_1core.bmodel
./gen_codec_bmodel.sh --mode fp16 --chip bm1684x --onnx_dir ../tools/tmp/onnx_tts
./gen_vqvae_bmodel.sh --mode fp16 --chip bm1684x --onnx_dir ../tools/tmp/onnx_tts
```

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

使用默认下载的模型，测试输入为："请详细描述这张图片。"，测试图片为`datasets/vita_newlog.jpg`，测试音频为空，保存音频为`test.wav`：
|    测试平台   |first token latency(s)|token per second(tokens/s)| RTF |
| -----------  | --------------------- | ----------------------- | --- |
| SE7-32       |   0.834               | 9.127                  |   0.284   |
 
测试输入为空，测试图片为`datasets/vita_newlog.jpg`，测试音频为`datasets/q1.wav`，保存音频为`test.wav`：
|    测试平台   |first token latency(s)|token per second(tokens/s)| RTF |
| -----------  | --------------------- | ----------------------- | --- |
| SE7-32       |   0.871               | 9.141                  |  0.283    |
 

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. 加载模型和第一次问答可能会比较慢，后续几次就会恢复正常。2