# Audio_assistant 例程

## 目录

- [Audio_assistant 例程](#Audio_assistant-例程)
  - [目录](#目录)
  - [简介](#简介)
  - [特性](#特性)
  - [1. 工程目录](#1-工程目录)
  - [2. 准备模型与数据](#2-准备模型与数据)
  - [3. 例程](#3-例程)


## 简介
Audio_assistant 例程是一个基于Whisper、MiniCPM、Llama3、VITS模型的语音助手系统，支持输入为音频，输出为音频，其中包括对输入音频内容回答的相关信息，暂只支持中文。可以实现流畅的人机交互，能够应用到智能机器人、智能家具、自动驾驶等多样化的应用场景。

## 特性
* 支持BM1688(SoC)
* 支持麦克风、文件输入，喇叭、文件输出

## 1. 工程目录

```bash
Audio_assistant
├── BM1688
│   ├── minicpm    # MiniCPM LLM模型文件夹	
│   ├── vits       # VITS语音生成模型文件夹	
│   └── whisper    # Whisper语音识别模型文件夹
├── datasets       # 包含了音频测试文件
├── python
│   ├── Llama3     # Llama3源码和依赖库文件夹
│   │   ├── python_demo   # Llama3源码文件夹
│   │   └── support       # Llama3依赖库文件夹
│   ├── libfirmware_core.so # Bmodel运行时库
│   ├── MiniCPM    # MiniCPM源码和依赖库文件夹
│   │   ├── demo          # MiniCPM源码文件夹
│   │   └── support       # MiniCPM依赖库文件夹
│   ├── whisper-TPU_py    # Whisper源码文件夹
│   ├── whisper_minicpm_llama3_vits.py     # 全流程串通源代码
│   └── whisperWrapper.py                  # Whisper接口源代码
└── scripts
    └── download.sh       # 模型、数据下载脚本
```

## 2. 准备模型与数据
​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，
```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

## 3. 例程
- [Python例程](./python/README.md)
