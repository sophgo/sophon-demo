# Seamless <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. 简介](#1-简介)
- [2. 特性](#2-特性)
- [3. 运行环境准备](#3-运行环境准备)
- [4. 模型编译](#4-模型编译)
- [5. 例程测试](#5-例程测试)
- [6. 精度测试](#6-精度测试)
  - [6.1 测试方法](#61-测试方法)
  - [6.2 测试结果](#62-测试结果)
- [7. 性能测试](#7-性能测试)

## 1. 简介
Seamless 是一个开源的深度学习语音识别模型，由 Meta 开发，它能够实现实时、多语言的语音识别、翻译，并支持跨多种环境和设备的灵活部署。本例程对[Seamless官方开源仓库](https://github.com/facebookresearch/seamless_communication)中的SeamlessStreaming（流式识别）和M4t（离线识别）算法进行移植，使之能在SOPHON BM1684X和BM1688上进行推理。同时该例程也参考了[FunASR官方开源仓库](https://github.com/modelscope/FunASR)搭建推理的websocket服务端和客户端。

## 2. 特性
* 支持BM1684X(x86 PCIe, SoC)
* 支持FP16(BM1684X)和FP32(BM1684X)模型编译和推理
* 支持FP16(BM1688)和FP32(BM1688)模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持websocket服务客户端高性能推理

## 3. 运行环境准备
在PCIe和BM1688 SOC上无需修改内存，以下为BM1684X SoC模式相关：
对于1684X系列设备（如SE7/SM7），都可以通过这种方式完成环境准备，使得满足Seamless运行条件。首先，在1684x SoC环境上，参考如下命令修改设备内存。
```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_v2.9.tar.xz
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息
./memory_edit.sh -c -npu 7615 -vpu 512 -vpp 3048 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```
> **注意：**
> 1. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)。

## 4. 模型编译
### 4.1 准备原始模型
该模型目前只支持在BM1684X和BM1688上运行，已提供原始导出的onnx模型，onnx模型导出方法可参考[Seamless(S2T)_Export_ONNX.md](docs/Seamless(S2T)_Export_ONNX.md)

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本
```bash
└── scripts
    └── download_onnx.sh                                                     # 通过该脚本下载SeamlessStreaming(s2t任务)和M4t(s2t任务)的onnx模型
```

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download_onnx.sh
```

下载的模型包括：
```
./models
├── onnx
    ├── m4t_s2t_onnx
    |   ├── m4t_s2t_decoder                                                                          # M4t(s2t任务) Decoder模块，onnx模型目录
    |   ├── m4t_s2t_decoder_frontend                                                                 # M4t(s2t任务) Decoder前端模块，用于BM1684X的onnx模型目录
    |   ├── m4t_s2t_decoder_frontend_bm1688                                                          # M4t(s2t任务) Decoder前端模块，用于BM1688的onnx模型目录
    |   ├── m4t_s2t_encoder                                                                          # M4t(s2t任务) Encoder模块，onnx模型目录
    |   ├── m4t_s2t_encoder_frontend                                                                 # M4t(s2t任务) Encoder前端模块，onnx模型目录
    |   └── m4t_s2t_final_proj                                                                       # M4t(s2t任务) Decoder线性模块，onnx模型目录
    └── streaming_s2t_onnx
        ├── streaming_s2t_decoder_step_bigger_than_1                                                 # SeamlessStreaming(s2t任务) Decoder模块，onnx模型目录
        ├── streaming_s2t_decoder_step_equal_1                                                       # SeamlessStreaming(s2t任务) Decoder模块，onnx模型目录
        ├── streaming_s2t_decoder_frontend                                                           # SeamlessStreaming(s2t任务) Decoder前端模块，用于BM1684X的onnx模型目录
        ├── streaming_s2t_decoder_frontend_bm1688                                                    # SeamlessStreaming(s2t任务) Decoder前端模块，用于BM1688的onnx模型目录
        ├── streaming_s2t_encoder                                                                    # SeamlessStreaming(s2t任务) Encoder模块，onnx模型目录
        ├── streaming_s2t_encoder_frontend                                                           # SeamlessStreaming(s2t任务) Encoder前端模块，onnx模型目录
        └── streaming_s2t_final_proj                                                                 # SeamlessStreaming(s2t任务) Decoder线性模块，onnx模型目录
```

### 4.2 编译环境搭建
导出的模型需要编译成BModel才能在SOPHON TPU上运行，需要使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，需要参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)中1、2、3(3)步骤，执行如下命令下载mlir包，并在docker容器中启动。
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/tpu-mlir_v1.10.beta.0-3-gdf397407d-20240805.tar.gz
tar -zxvf tpu-mlir_v1.10.beta.0-3-gdf397407d-20240805.tar.gz
source tpu-mlir_v1.10.beta.0-3-gdf397407d-20240805/envsetup.sh
```

### 4.3 编译命令执行
安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophon.cn/site/index/material/all/all.html)相应版本的SDK中获取)。

- 生成SeamlessStreaming BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译BModel的脚本，请注意修改`gen_streaming_s2t_bmodel.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，如：

```bash
cd ./scripts
# 编译BM1684X上的模型
./gen_streaming_s2t_bmodel.sh bm1684x
# 编译BM1688上的模型
./gen_streaming_s2t_bmodel.sh bm1688
```

​执行上述命令会在`models/BM1684X`或`models/BM1688`文件夹下生成`seamless_streaming_encoder_fp16_s2t.bmodel `等文件，即转换好的BModel。

- 生成M4t BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译BModel的脚本，请注意修改`gen_m4t_s2t_bmodel.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，如：

```bash
cd ./scripts
# 编译BM1684X上的模型
./gen_m4t_s2t_bmodel.sh bm1684x
# 编译BM1688上的模型
./gen_m4t_s2t_bmodel.sh bm1688
```

​执行上述命令会在`models/BM1684X`或`models/BM1688`文件夹下生成`m4t_decoder_beam_size_fp16_s2t.bmodel`等文件，即转换好的BModel。

## 5. 例程测试

- [Python例程](./python/README.md)
- [websocket高性能例程](./python/websocket_demo/README.md)

> **注意**：
> 1. websocket高性能例程在提高性能的同时会损失一定的流式推理精度，非流式精度不变，可根据需要进一步修改非流式的beam size为1来提高性能并损失可忽略不计的精度。

## 6. 精度测试
### 6.1 测试方法
首先，参考[Python例程](python/README.md#32-使用方式)进行中文语音到中文文字的转换，生成预测结果至result路径，注意修改数据集为../datasets/aishell_S0764和相关参数。

然后使用`tools`目录下的`eval_aishell.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出语音识别的评价指标，命令如下：
```bash
# 请根据实际情况修改程序路径和txt文件路径
python3 tools/eval_aishell.py --char=1 --v=1 datasets/ground_truth.txt python/results  > online_wer
cat online_wer | grep "Overall"
```
> **注意**：
> 1. 若遇到报错`OSError: libopencc.so.1: cannot open shared object file: No such file or directory`，需要安装依赖库`opencc`，执行`sudo apt-get install opencc`
> 2. 与开源项目输出的差异来自该例程将pt转onnx和使用更大padding mask，pt转为onnx带来的影响大于padding mask，但对精度指标都几乎没有影响。

### 6.2 测试结果
在aishell数据集上，ASR任务精度测试结果如下：
|   测试平台    |             测试程序                 |              测试模型                                  | WER    |
| ------------ | ------------------------------------ | ----------------------------------------------------- | ------ |
|   SE7-32     | pipeline_seamless_streaming_s2t.py   |     SeamlessStreaming(s2t任务)模型                     | 18.73% |
|   SE7-32     | pipeline_m4t_s2t.py                  |     M4t(s2t任务)模型                                   | 2.10%  |
|   SE9-16     | pipeline_seamless_streaming_s2t.py   |     SeamlessStreaming(s2t任务)模型                     | 18.95% |
|   SE9-16     | pipeline_seamless_streaming_s2t.py   |     SeamlessStreaming(s2t任务)2core模型                | 18.80% |
|   SE9-16     | pipeline_m4t_s2t.py                  |     M4t(s2t任务)模型                                   | 2.32%  |
|   SE9-16     | pipeline_m4t_s2t.py                  |     M4t(s2t任务)2core模型                              | 2.10%  |

> **测试说明**：
> 1. 在使用的模型相同的情况下，wer在不同的测试平台上是相同的。
> 2. 由于SDK版本之间的差异，实测的wer与本表有1%以内的差值是正常的。
> 3. 运行时均采用默认参数，其中SeamlessStreaming(s2t任务)模型提高一次性处理的序列长度和`--consecutive_segments_num`能大幅提高精度到5%以内；M4t(s2t任务)模型提高`--beam_size`到5（最大值）也能提高精度。

## 7. 性能测试
|    测试平台   |              测试程序                 |           测试模型                  |  load time(ms) |  Inference time(ms)   |  encode time(ms) |  decode time(ms)   |
| -----------  | ------------------------------------- | -----------------------------------| ---------------- | ------------------- | ---------------- | ------------------ | 
|   SE7-32     | pipeline_seamless_streaming_s2t.py    |   SeamlessStreaming(s2t任务)模型    | 26.15            |  387.62             |  28.36           |  340.70            |           
|   SE7-32     | pipeline_m4t_s2t.py                   |   M4t(s2t任务)模型                  | 17.30            |  358.25             |  21.78           |  309.99            |
|   SE9-16     | pipeline_seamless_streaming_s2t.py    |   SeamlessStreaming(s2t任务)模型    | 27.28            |  851.11             |  109.34          |  720.05            |
|   SE9-16     | pipeline_seamless_streaming_s2t.py    | SeamlessStreaming(s2t任务)2core模型 | 27.64            |  947.19             |  143.10          |  825.02            |               
|   SE9-16     | pipeline_m4t_s2t.py                   |   M4t(s2t任务)模型                  | 13.98            |  956.21             |  82.21           |  853.54            |      
|   SE9-16     | pipeline_m4t_s2t.py                   |   M4t(s2t任务)2core模型             | 8.18             |  785.10             |  52.65           |  720.89            |

> **测试说明**：
> 1. 该性能使用datasets/test/demo.wav音频进行测试，执行ASR+翻译为中文任务，计算后得出平均每秒音频所需推理时间。
> 2. seamless模型的预处理主要包括加载语音，特征提取等，推理后的结果可直接转换为自然语言，时间可忽略不计，因此后处理部分时间统计到推理部分。
> 3. 性能测试结果具有一定的波动性，实测结果与该表结果有误差属正常现象，建议多次测试取平均值。
> 4. BM1684X SoC的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz。
> 5. 运行时均采用默认参数。
> 6. `encode time`和`decode time`未包含全流程中涉及的cpu操作，例如beam search，而`Inference time`包含。