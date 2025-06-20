import argparse
import os
import sys

from loguru import logger
import numpy as np
import requests

# 模型的instruction需要自己查找，如BAAI模型的指示在FlagEmbedding仓库可见，snowflake指示在模型仓库config_sentence_transformers.json文件里
instruction = {
    "bge-large-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
    "bge-m3": "",
    "snowflake-arctic-embed-l-v2.0": "query: ",
}

# 虚构的语料库，可自己添加别的文本块
zh_corpus=[
"迈克尔·杰克逊是一位传奇流行音乐偶像，以其屡次打破纪录的音乐作品和舞蹈创新著称。",
"李飞飞是斯坦福大学的教授，通过 ImageNet 项目彻底革新了计算机视觉领域。",
"布拉德·皮特是一位多才多艺的演员兼制片人，以在《搏击俱乐部》和《好莱坞往事》等影片中的角色闻名。",
"杰弗里·辛顿作为人工智能领域的奠基人物，因其在深度学习方面的卓越贡献获得了图灵奖。",
"埃米纳姆是一位著名说唱歌手，也是史上最畅销的音乐艺术家之一。",
"泰勒·斯威夫特是一位格莱美获奖的创作型歌手，以其叙事性强的音乐风格而广受好评。",
"山姆·奥特曼担任 OpenAI 首席执行官，在 GPT 系列模型的开发中取得了惊人成就，并致力于打造安全且有益的人工智能。",
"徐艺真是一位备受赞誉的网络短剧演员，参演了郭敬明执导的网络短剧《AI》。",
"吴恩达通过 Coursera 和斯坦福大学的公开课程，将人工智能知识传播到全世界。",
"小罗伯特·唐尼是一位标志性演员，最广为人知的角色是在漫威电影宇宙中饰演钢铁侠。",
]

# 构建发送请求的函数，用于发送请求到text-embeddings-server服务，连续三次无响应则返回None
def build_post_sender(ip, port, headers={"Content-Type": "application/json"}):
    def send_request(api, payload):
        attemp = 0
        while attemp < 3:
            try:
                url = f"http://{ip}:{port}/{api}"
                response = requests.post(url, headers=headers, json=payload, timeout=3)
                response.raise_for_status()
                return response.json()
            except requests.Timeout:
                logger.info("request timed out.")
            except requests.RequestException as e:
                logger.info(f"send request failed: {e}")
                return None
            attemp += 1
            time.sleep(1)
        return None
    return send_request

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_model", type=str, help="retrieval model name, used for matching instruction", default="bge-large-zh-v1.5")
    parser.add_argument("--retrieval_server_ip", type=str, help="retrieval server ip, default to localhost", default="localhost")
    parser.add_argument("--retrieval_server_port", type=int, help="retrieval server port", default=None)
    parser.add_argument("--reranker_model", type=str, help="reranker model name, not working for now", default="bge-reranker-large")
    parser.add_argument("--reranker_server_ip", type=str, help="reranker server ip, default to localhost", default="localhost")
    parser.add_argument("--reranker_server_port", type=int, help="reranker server port", default=None)
    parser.add_argument("--query", type=str, help="query text", default="谁可能是一个神经网络领域的专家？")
    args = parser.parse_args()

    # step1: 测试text-embeddings-server服务是否正常
    if args.retrieval_server_port is None:
        logger.error("retrieval server port is not set")
        sys.exit(1)
    else:
        retrieval_post_sender = build_post_sender(args.retrieval_server_ip, args.retrieval_server_port)
        embeddings = retrieval_post_sender("embed", {"inputs": "测试服务是否可用"})
        if embeddings is None:
            logger.error("retrieval server is not available")
            sys.exit(1)
    if args.reranker_server_port is None:
        logger.info("reranker server port is not set, will not use reranker")
    else:
        reranker_post_sender = build_post_sender(args.reranker_server_ip, args.reranker_server_port)
        scores = reranker_post_sender("rerank", {"query": "重排服务是否可用？", "texts": ["重排服务可用"]})
        if scores is None:
            logger.error("reranker server is not available")
            sys.exit(1)

    # step2: 生成corpus的嵌入数据
    corpus_embeddings = []
    for index in range(0, len(zh_corpus)):
        embeddings = retrieval_post_sender("embed", {"inputs": zh_corpus[index], "truncate": True, "truncation_direction": "Right"})
        if embeddings is None:
            logger.error("please check the retrieval server's status")
            sys.exit(1)
        corpus_embeddings.append(np.array(embeddings).astype(np.float32))
    corpus_embeddings = np.array(corpus_embeddings)

    # step3: 生成query的嵌入数据
    query_instruction = instruction[args.retrieval_model] if args.retrieval_model in instruction else ""
    query_embeddings = retrieval_post_sender("embed", {"inputs": query_instruction + args.query, "truncate": True, "truncation_direction": "Right"})
    if query_embeddings is None:
        logger.error("please check the retrieval server's status")
        sys.exit(1)
    query_embedding = np.array(query_embeddings).astype(np.float32)

    # step4: 计算相似度，可以替换别的相似度计算方法
    def cos_sim(vec1, vec2):
        vec1_item = vec1.flatten().astype(np.float32)
        vec2_item = vec2.flatten().astype(np.float32)
        dot_product = np.dot(vec1_item, vec2_item)
        norm_vec1 = np.sqrt(np.sum(vec1_item ** 2))
        norm_vec2 = np.sqrt(np.sum(vec2_item ** 2))
        cos_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cos_similarity

    scores = []
    for idx in range(corpus_embeddings.shape[0]):
        cos_similarity = cos_sim(query_embedding, corpus_embeddings[idx])
        scores.append(cos_similarity)

    # step5: 按相似度进行排序
    sorted_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    logger.info(f"语料库与提问：\"{args.query}\" 按余弦相似度降序排序为：")
    for idx in sorted_indices:
        logger.info(f"score: {scores[idx]}, {zh_corpus[idx]}")

    # step6: 使用reranker模型进行重排序
    if args.reranker_server_port is None:
        sys.exit(0)
    candidate_corpus = [zh_corpus[idx] for idx in sorted_indices[:6]]

    reranker_scores = []
    for idx in range(len(candidate_corpus)):
        scores = reranker_post_sender("rerank", {"query": args.query, "texts": [candidate_corpus[idx]]})
        if scores is None:
            logger.error("please check the reranker server's status")
            sys.exit(1)
        reranker_scores.append(scores[0]['score'])
    reranker_sorted_indices = sorted(range(len(reranker_scores)), key=lambda x: reranker_scores[x], reverse=True)
    logger.info(f"retrieval给出的候选语料库与提问：\"{args.query}\" 按reranker给出的相关度降序排序为：")
    for idx in reranker_sorted_indices:
        logger.info(f"score: {reranker_scores[idx]}, {candidate_corpus[idx]}")
