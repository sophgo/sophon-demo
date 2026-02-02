# RAG Demo

简易的RAG Demo。

## 特性

* 使用`API`调用大模型，支持本地部署的`TGI`等开源框架或调用第三方提供的接口
* 支持多用户访问
* 支持`windows`和`ubuntu`操作系统

## 使用方法

```shell
# 注意，默认的配置文件使用api调用llm，使用本地权重调用embedding model。使用前可以根据实际状况修改
pip install -r requirements.txt
streamlit run ./st_demo.py
```

## 参考配置

配置文件：[config.json](app/config/config.json)

```json
{
    "llm_config": {
        "model": "spn3/Qwen2.5-72B-Instruct",
        "api_url": "https://www.sophnet.com/api/open-apis",
        "api_key": ""
    },
    "doc_config": {
        "embedding_model": "bge",
        "embedding_model_path": "./bge-small-zh",
        "split_method": "FixedLength",
        "chunk_length": 512,
        "overlap": 50,
        "database_method": "Faiss",
        "dimension": 512,
        "topk": 1
    }
}
```

## 可选参数列表

| 参数名称 | 可选项 | 含义 |
| ------ | ----- | ----- |
| llm_config.model | - | 语言模型名称 |
| llm_config.api_url | - | Api url。用于初始化`client` |
| llm_config.api_key | - | Api key。用于初始化`client`。会优先从环境变量中读取`RAG_GENERATOR_API_KEY`，若读不到则读取配置文件 |
| doc_config.embedding_model | "bge" | `embedding`模型名称 |
| doc_config.embedding_model_path | - | `embedding`模型路径。如果模型路径存在则从本地初始化，否则根据模型名称从云端仓库拉取。如果设置了`doc_config.api_url`，优先使用API |
| doc_config.api_url | - | Api url。用于初始化`client` |
| doc_config.api_key | - | Api key。用于初始化`client` |
| doc_config.split_method | "fixedlength" | 文档切分逻辑，"fixedlength"指按照固定长度切分 |
| doc_config.chunk_length | - | 当`split_method`为`fixedlength`时，切分出每个文本块的长度 |
| doc_config.overlap | - | 当`split_method`为`fixedlength`时，切分文本块时，相邻块之间重叠的长度 |
| doc_config.database_method | "faiss" | `embedding`向量搜索的方式 |
| doc_config.dimension | - | `embedding`向量长度 |
| doc_config.topk | - | 搜索相关文本块时结果的最大数量 |

## 说明

> `embedding`功能依赖开源库[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)，请按照该仓库中的文档获取`embedding`模型。

> 对于本地部署`VLLM`、`TGI`等`LLM`框架，本项目通过API调用推理接口的情况：`backend_type`设置为`api`，`api_url`设置为`http://localhost:port/v1`，`api_key`置空即可。

> 本项目依赖`streamlit`搭建前端页面，而`streamlit`对多用户数据实现了隔离，所以理论上可以在云端部署，提供给一定规模的组织使用。

> 对于`windows`操作系统，临时文件保存在`D:/`目录下；对于`ubuntu`操作系统，临时文件保存在`/tmp/`目录下。请确保对于此目录有读写权限。

## 本地部署

在本地 TPU 环境中离线部署 RAG 所需的推理服务，利用 TPU 算力加速 LLM 与 Embedding 推理，无需云端依赖。

- 操作系统：Ubuntu 24.04（x86 架构）
- 硬件：BM1690
- 前置准备：TGI、TEI 镜像、LLM/Embedding 模型权重文件

### 1. 启动 TGI 服务

```sh
# 启动并测试 TGI 服务
export IMAGE=soph_tgi:3.2.0-slim
export MODEL_HOME=/workspace/models
export MODEL_ID=DeepSeek-R1-Distill-Qwen-32B
export NUM_SHARD=2
export CHIP_MAP=0,1
export MAX_BATCH_SIZE=16
export MAX_TOTAL_TOKENS=2048
export MAX_INPUT_TOKENS=1024
export MAX_BATCH_TOTAL_TOKENS=$((MAX_BATCH_SIZE * MAX_TOTAL_TOKENS))
export MAX_BATCH_PREFILL_TOKENS=$((MAX_BATCH_SIZE * MAX_INPUT_TOKENS))
export DEFAULT_GENERATION_LENGTH=$MAX_TOTAL_TOKENS
export KVCACHE_BLOCKS=$((MAX_BATCH_TOTAL_TOKENS / 16))

docker run -d --privileged --name rag-tgi \
  -e TPU_CACHE_MAX_MEMORY=8589934592 \
  -e TPU_CACHE_TAG_REUSE=0 \
  -e TPU_ALLOCATOR_ALIGN_SIZE=4096 \
  -e KVCACHE_BLOCKS=$KVCACHE_BLOCKS \
  -e DEFAULT_GENERATION_LENGTH=$DEFAULT_GENERATION_LENGTH \
  -e CHIP_MAP=$CHIP_MAP \
  --shm-size 1g \
  -p 8080:80 \
  -v /dev/:/dev/ \
  -v /opt/tpuv7:/opt/tpuv7 \
  -v $MODEL_HOME:/data \
  $IMAGE \
  --model-id /data/$MODEL_ID \
  --num-shard $NUM_SHARD \
  --max-batch-size $MAX_BATCH_SIZE \
  --max-batch-total-tokens $MAX_BATCH_TOTAL_TOKENS \
  --max-batch-prefill-tokens $MAX_BATCH_PREFILL_TOKENS \
  --max-total-tokens $MAX_TOTAL_TOKENS \
  --max-input-tokens $MAX_INPUT_TOKENS

docker logs -f rag-tgi  # 等待启动完成

curl localhost:8080/v1/chat/completions -X POST -H 'Content-Type: application/json' -d '{"model": "tgi", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "你好"}], "stream": false, "max_tokens": 20}'
```

### 2. 启动 TEI 服务

```sh
# 启动并测试 TEI 服务
export IMAGE=soph_tei:0.1
export MODEL_HOME=/workspace/models
export MODEL_ID=bge-large-zh-v1.5
export CHIP_MAP=2
export MAX_BATCH_SIZE=256
export MAX_BATCH_TOTAL_TOKENS=131072
docker run -d --privileged --name rag-tei \
  -e CHIP_MAP=$CHIP_MAP \
  -v /dev/:/dev/ \
  -v /opt/tpuv7/:/opt/tpuv7/ \
  -v $MODEL_HOME:/data \
  -p 8081:80 \
  --ipc host \
  $IMAGE \
  text-embeddings-router \
  --model-id /data/$MODEL_ID \
  --max-client-batch-size $MAX_BATCH_SIZE \
  --max-batch-tokens $MAX_BATCH_TOTAL_TOKENS \
  --tokenization-workers 4

docker logs -f rag-tei  # 等待启动完成

curl localhost:8081/embed -X POST -H 'Content-Type: application/json' -d '{"inputs": ["What is Deep Learning?"], "dimensions": 1024}'
```

### 3. 启动应用

配置文件：[config_self_hosted.json](app/config/config_self_hosted.json)

```json
{
    "llm_config": {
        "model": "tgi",
        "api_url": "http://<host_ip>:8080/v1",
        "api_key": ""
    },
    "doc_config": {
        "embedding_model": "bge",
        "api_url": "http://<host_ip>:8081/v1",
        "api_key": "",
        "split_method": "FixedLength",
        "chunk_length": 256,
        "overlap": 50,
        "database_method": "Faiss",
        "dimension": 1024,
        "topk": 5
    }
}
```

```sh
streamlit run ./st_demo.py -- --config ./app/config/config_self_hosted.json
```
