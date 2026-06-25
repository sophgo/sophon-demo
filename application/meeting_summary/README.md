# meeting_summary

## 目录
- [meeting_summary](#meeting_summary)
  - [简介](#简介)
  - [特性](#特性)
  - [1. 工程目录](#1-工程目录)
  - [2. 准备模型与数据](#2-准备模型与数据)
  - [3. 例程](#3-例程)

## 简介

meeting_summary 例程是一个基于BM1684X的会议文本总结应用，使用Qwen1.5-7B模型。

## 特性

* 支持BM1684X(PCIE、SOC)

## 1. 工程目录

```bash
meeting_summary
├── models
│   ├── qwen1.5-7b_int4_6k_1dev.bmodel
├── python
│   ├── utils                         # 工具库
│   ├── main.py                       # 主函数
│   └── prompt.txt                    # prompt模板
│   └── meeting.txt                   # 会议文本示例
│   └── requirements.txt              # python依赖
└── scripts
    ├── download_model.sh       # 模型下载脚本
    ├── download_tokenizer.sh   # tokenizer下载脚本
```

## 2. 准备模型与数据

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/

# 下载tokenizer
./scripts/download_tokenizer.sh 

# 下载模型文件
./scripts/download_model.sh 
```


## 3. 例程

- [Python例程](./python/README.md)


