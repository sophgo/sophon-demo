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

* 支持BM1684X(PCIE、SOC)
* 支持多种文档格式(PDF, DOCX, TXT)
* 提供用户界面
* 支持bce-reranker

## 1. 工程目录

```shell
├── models
│   ├── BM1684X                                # BM1684X专用模型
│   │   ├── bce_embedding                      # BM1684X上运行的bce_embedding
│   │   │   ├── bce-embedding-base_v1.bmodel
│   │   │   └── token_config
│   │   │       ├── special_tokens_map.json
│   │   │       ├── tokenizer_config.json
│   │   │       └── tokenizer.json
│   │   ├── bce_reranker                      # BM1684X上运行的bce_reranker
│   │   │   ├── bce-reranker-base_v1.bmodel
│   │   │   └── token_config
│   │   │       ├── special_tokens_map.json
│   │   │       ├── tokenizer_config.json
│   │   │       └── tokenizer.json
│   │   └── qwen1.5-7b_int4_seq2048_1dev.bmodel # BM1684X上运行的qwen1.5-7b, int4量化, 上下文长度2k, 单芯模型
│   └── qwen                                  # qwen系列模型的提词器
│       └── token_config
│           ├── tokenizer_config.json
│           ├── tokenizer.json
│           └── vocab.json
├── nltk_data
├── pics                                      # 文档用图
│   ├── embedding.png
│   ├── img1.png
│   └── img2.png
├── python
│   ├── chat                                  # 聊天机器人
│   │   ├── chatbot.py
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── config.ini                            # 本项目的配置方法
│   ├── config.yaml                           # LLM_server_api服务的配置，LLM模型部分
│   ├── data                                  # 存储文档和保存知识库
│   │   ├── db_tpu
│   │   └── uploaded
│   ├── doc_processor                         # 文档处理模块
│   │   ├── document_loaders
│   │   │   ├── FilteredCSVloader.py
│   │   │   ├── __init__.py
│   │   │   ├── mydocloader.py
│   │   │   ├── myimgloader.py
│   │   │   ├── mypdfloader.py
│   │   │   ├── mypptloader.py
│   │   │   └── ocr.py
│   │   ├── __init__.py
│   │   ├── knowledge_file.py
│   │   └── text_splitter
│   │       ├── ali_text_splitter.py
│   │       ├── chinese_recursive_text_splitter.py
│   │       ├── chinese_text_splitter.py
│   │       ├── __init__.py
│   │       └── zh_title_enhance.py
│   ├── embedding                            # embedding推理
│   │   ├── embedding.py
│   │   ├── __init__.py
│   │   ├── npuengine.py
│   │   └── sentence_model.py
│   ├── knowledge_base
│   ├── README.md                             # python例程的README
│   ├── requirements.txt
│   ├── reranker
│   │   ├── __init__.py
│   │   └── reranker_tpu.py
│   └── web_demo_st.py
├── README.md                                   # 项目总文档
└── scripts                                     # 下载脚本
    └── download.sh
```

## 2. 准备模型与数据

```shell
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/

# 下载模型文件和nltk
./scripts/download.sh 
```

## 3. 例程

- [Python例程](./python/README.md)
