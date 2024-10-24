# LLM_api_server

## 目录
- [LLM_api_server](#LLM_api_server)
  - [简介](#简介)
  - [特性](#特性)
  - [1. 工程目录](#1-工程目录)
  - [2. 准备模型与数据](#2-准备模型与数据)
  - [3. 例程](#3-例程)
  - [4. 性能测试](#4-性能测试)

## 简介

LLM_api_server 例程是一个基于BM1684X构建的一个类Openai_api的LLM服务，目前支持ChatGLM3、Qwen、Qwen1.5、Qwen2。

## 特性

* 支持BM1684X(PCIE、SOC)
* 支持openai库进行调用
* 支持web接口调用

## 1. 工程目录

```bash
LLM_api_server
├── models
│   ├── BM1684X
│   │   ├── chatglm3-6b_int4.bmodel                # BM1684X chatglm3-6b模型
│   │   └── qwen2-7b_int4_seq512_1dev.bmodel       # BM1684X qwen2-7b模型	
│   └── BM1688
│       ├── chatglm3-6b_int4_2core.bmodel                           # BM1688  chatglm3-6b 双核模型
│       ├── qwen1.5-1.8b_int4_seq512_bm1688_1dev.bmodel             # BM1688  qwen1.5-1.8b 单核模型	
│       └── qwen1.5-1.8b_int4_seq512_bm1688_1dev_2core.bmodel       # BM1688  qwen1.5-1.8b 双核模型	
├── python
│   ├── utils                         # 工具库
│   ├── api_server.py                 # 服务启动程序
│   ├── config.yaml                   # 服务配置文件
│   ├── request.py                    # 请求示例程序
│   └── requirements.txt              # python依赖
└── scripts
    ├── download_model.sh       # 模型下载脚本
    └── download_tokenizer.sh   # tokenizer下载脚本
```

## 2. 准备模型与数据

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/

# 下载tokenizer
./scripts/download_tokenizer.sh 

# 下载BM1684X的模型文件
./scripts/download_model.sh 
# 下载BM1688的模型文件
./scripts/download_model.sh bm1688
```


## 3. 例程

- [Python例程](./python/README.md)

## 4. 性能测试

模型性能可参考sophon-demo/sample对应的模型仓库。

