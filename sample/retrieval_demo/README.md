[简体中文](./README.md)

# Retrieval demo

## 目录

- [Retrieval demo](#Retrieval demo)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 环境准备](#2-环境准备)
  - [3. 例程测试](#3-例程测试)
    - [3.1 例程运行方法](#31-例程运行方法)
    - [3.2 测试结果说明](#32-测试结果说明)

## 1. 简介

本例程基于SDK中的text-embeddings-server服务，展示了用text-embeddings-server里的检索/重排模型把文本转化为稠密向量，再用稠密向量进行信息检索的基本过程。用户可参考该例程来设计自己的具体任务，如语料库分类等。

## 2.环境准备

```shell
# 安装第三方库
pip install -r requirements.txt
```
## 3.例程测试
### 3.1 例程运行方法

请参考SDK中text-embeddings-server项目下的`README_SOPH.md`文件启动retrieval模型的服务，如果要使用reranker，也需要另起一个reranker模型的服务。
执行`python3 main.py --retrieval_model bge-large-zh-v1.5 --retrieval_server_port <port> --query "谁可能是一个神经网络领域的专家？"`可在终端进行测试，`main.py`支持如下命令行参数：
```shell
--retrieval_model 检索模型的名字，用来给query添加instruction
--retrieval_server_ip 检索模型服务的IP，默认为localhost
--retrieval_server_port 检索模型服务的端口，必需
--reranker_server_ip 重排模型服务的IP，默认为localhost
--reranker_server_port 重排模型服务的端口，不提供则不会跑重排
--query 用来对语料库进行查询的提问
```

### 3.2 测试结果说明

`main.py`程序运行结束，会在终端对语料库中的内容按与query的相关度降序打印，如果加载了重排模型，会对嵌入模型给出的前几个句子进行重排序，如：
与"谁可能是一个神经网络领域的专家？" 按余弦相似度降序排序为：
李飞飞是斯坦福大学的教授，通过 ImageNet 项目彻底革新了计算机视觉领域。
杰弗里·辛顿作为人工智能领域的奠基人物，因其在深度学习方面的卓越贡献获得了图灵奖。
山姆·奥特曼担任 OpenAI 首席执行官，在 GPT 系列模型的开发中取得了惊人成就，并致力于打造安全且有益的人工智能
徐艺真是一位备受赞誉的网络短剧演员，参演了郭敬明执导的网络短剧《AI》。
吴恩达通过 Coursera 和斯坦福大学的公开课程，将人工智能知识传播到全世界。

经reranker对候选语料库进行重排后，按相关度降序排序为：
吴恩达通过 Coursera 和斯坦福大学的公开课程，将人工智能知识传播到全世界。
李飞飞是斯坦福大学的教授，通过 ImageNet 项目彻底革新了计算机视觉领域。
杰弗里·辛顿作为人工智能领域的奠基人物，因其在深度学习方面的卓越贡献获得了图灵奖。
山姆·奥特曼担任 OpenAI 首席执行官，在 GPT 系列模型的开发中取得了惊人成就，并致力于打造安全且有益的人工智能
徐艺真是一位备受赞誉的网络短剧演员，参演了郭敬明执导的网络短剧《AI》。