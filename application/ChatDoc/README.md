# ChatDoc

## 目录
- [ChatDoc](#ChatDoc)
  - [简介](#简介)
  - [特性](#特性)
  - [1. 工程目录](#1-工程目录)
  - [2. 准备模型与数据](#2-准备模型与数据)
  - [3. 例程](#3-例程)

## 简介

ChatDoc例程是一个基于BM1684X构建的用自然语言与文档进行交互的服务，可快速提取文档内容并用于问答，此项目基于[LangChain](https://github.com/langchain-ai/langchain)。本项目需要和demo/application下的[LLM_api_server](../LLM_api_server/README.md)服务配合使用，先启动LLM_api_server服务，再启动本项目；加入[bce-reranker-base_v1](https://huggingface.co/maidalun1020/bce-reranker-base_v1)优化文本对话能力，总体流程如下图所示：![Flow](<./pics/embedding.png>)

## 特性

* 支持BM1684X(PCIE、SOC)和BM1688(SOC)
* 支持多种文档格式(PDF, DOCX, TXT)
* 提供用户界面
* 支持bce-reranker

## 1. 工程目录

```shell
├── models
│   ├── BM1684X                                       # BM1684X专用模型
│   │   ├── bce_embedding                             # BM1684X上运行的bce_embedding
│   │   │   ├── bce-embedding-base_v1.bmodel
│   │   │   └── token_config
│   │   │       ├── special_tokens_map.json
│   │   │       ├── tokenizer_config.json
│   │   │       └── tokenizer.json
│   │   ├── bce_reranker                              # BM1684X上运行的bce_reranker
│   │   │   ├── bce-reranker-base_v1.bmodel
│   │   │   └── token_config
│   │   │       ├── special_tokens_map.json
│   │   │       ├── tokenizer_config.json
│   │   │       └── tokenizer.json
│   │   └── qwen                                      # BM1684X上运行的qwen1.5-7b, int4量化, 上下文长度2k, 单芯模型
│   │       ├── qwen1.5-7b_int4_seq2048_1dev.bmodel
│   │       └── token_config
│   │           ├── tokenizer_config.json
│   │           ├── tokenizer.json
│   │           └── vocab.json
│   └── BM1688                                        # BM1688专用模型
│       ├── bce_embedding                             # BM1688上运行的bce_embedding
│       │   ├── bce-embedding-base_v1.bmodel
│       │   └── token_config
│       │       ├── special_tokens_map.json
│       │       ├── tokenizer_config.json
│       │       └── tokenizer.json
│       ├── bce_reranker                              # BM1688上运行的bce_reranker
│       │   ├── bce-reranker-base_v1.bmodel
│       │   └── token_config
│       │       ├── special_tokens_map.json
│       │       ├── tokenizer_config.json
│       │       └── tokenizer.json
│       └── qwen                                      # BM1688上运行的qwen2.5-1.5b, int4量化, 上下文长度2k, 双核模型
│           ├── qwen2.5-1.5b_int4_seq2048_1688_2core.bmodel
│           └── token_config
│               ├── tokenizer_config.json
│               ├── tokenizer.json
│               └── vocab.json
├── nltk_data
├── pics                            # 文档用图
│   ├── embedding.png
│   ├── img1.png
│   └── img2.png
├── python                          # python例程
│   ├── chat                        # 聊天机器人
│   │   ├── chatbot.py
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── config.ini                  # 本项目的配置方法
│   ├── config.yaml                 # LLM_server_api服务的配置，LLM模型部分
│   ├── data                        # 存储文档和保存知识库
│   │   ├── db_tpu
│   │   └── uploaded
│   ├── doc_processor               # 文档处理模块
│   ├── embedding                   # embedding推理
│   │   ├── embedding.py
│   │   ├── __init__.py
│   │   ├── npuengine.py
│   │   └── sentence_model.py
│   ├── README.md                   # python例程的README
│   ├── requirements.txt            # 需要安装的第三方库
│   ├── reranker                    # reranker模块
│   │   ├── __init__.py
│   │   └── reranker_tpu.py
│   └── web_demo_st.py              # python例程启动文件
├── README.md                       # 项目总文档
└── scripts
    └── download.sh                 # 下载脚本
```

## 2. 准备模型与数据

```shell
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/

# 下载BM1684X的模型文件和nltk
./scripts/download.sh 
# 下载BM1688的模型文件和nltk
./scripts/download.sh bm1688
```

## 3. 例程

- [Python例程](./python/README.md)
