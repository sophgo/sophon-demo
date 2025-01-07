# ChatTTS

## 目录
- [ChatTTS](#chattts)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [4. 准备模型](#4-准备模型)
    - [4.1 使用提供的模型](#41-使用提供的模型)
    - [4.2 自行编译模型](#42-自行编译模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 程序性能测试](#6-程序性能测试)

## 1. 简介
ChatTTS 是一款专门为对话场景（例如 LLM 助手）设计的文本转语音模型。本例程参考了[ChatTTS-ONNX](https://github.com/ZillaRU/ChatTTS-ONNX)，对[ChatTTS官方仓库](https://github.com/2noise/ChatTTS)中的算法进行移植，使之能在SOPHON BM1684X/BM1688上进行推理测试。


## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)、BM1688(SoC)
* 支持BF16、INT8、INT4模型编译和推理
* 支持基于SAIL推理的Python例程

## 4. 准备模型

### 4.1 使用提供的模型

​本例程在`scripts`目录下提供了下载脚本`download.sh`

**注意：**在运行前，应该保证存储空间大于3GB。

```bash
chmod -R +x scripts/
./scripts/download.sh
```

执行下载脚本后，当前目录下的文件如下：

```bash
├── docs
│   └── ChatTTS_Export_Guide.md    #onnx导出和bmodel编译指南
├── models
|   ├── asset                      #一些不需要编译成bmodel的权重文件
|   ├── chattts-llama_int4_1dev_1024_bm1684x.bmodel # gpt bmodel，int4精度，使用1个device，seq_len=1024，运行在bm1684x。
|   ├── chattts-llama_int4_1dev_1024_bm1688.bmodel  # gpt bmodel，int4精度，使用1个device，seq_len=1024，运行在bm1688。
|   ├── decoder_1-768-1024_bm1684x.bmodel           # decoder bmodel，bf16精度，输入大小为[1,768,1024]，运行在bm1684x
|   ├── decoder_1-768-1024_bm1688.bmodel            # decoder bmodel，bf16精度，输入大小为[1,768,1024]，运行在bm1688
|   ├── vocos_1-100-2048_bm1684x.bmodel             # vocos bmodel，bf16精度，输入大小为[1,100,2048]，运行在bm1684x
|   └── vocos_1-100-2048_bm1688.bmodel              # vocos bmodel，bf16精度，输入大小为[1,100,2048]，运行在bm1684x
├── python/
|   ├── ChatTTS                     #封装好的ChatTTS模块，推理部分用sail实现。
|   ├── README.md                   #运行指南
|   └── test.py                     #模块调用示例
├── README.md                       #ChatTTS例程指南
├── scripts                         
│   ├── download.sh                 #下载脚本
|   ├── gen_decoder_bmodel.sh       #decoder bmodel编译脚本
|   ├── gen_gpt_bmodel.sh           #gpt bmodel编译脚本
|   └── gen_vocos_bmodel.sh         #vocos bmodel编译脚本
└── tools
    ├── config.py                   #导出onnx需要的配置
    ├── dvae.py                     #dvae模型结构文件
    ├── exporter.py                 #导出脚本
    ├── gpt.py                      #gpt模型结构文件
    └── modeling_llama.py           #gpt里llama的模型结构文件。
```


### 4.2 自行编译模型

此部分请参考[ChatTTS模型导出与编译](./docs/ChatTTS_Export_Guide.md)

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

|    测试平台   |     测试程序       |           测试模型                     |   RTF  | tpu利用率(100%) | cpu利用率(800%) | 
| -----------  | ----------------  | ---------------------------            | ------ | --------       | --------- |
|     SE9-16   |  test.py         | gpt(int4) + decoder(bf16) + vocos(bf16) |   2.5 | 15%~30%        |  100%~150% |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE9-16的SDK版本是V1.7；
