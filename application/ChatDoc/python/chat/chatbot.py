# coding=utf-8
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import shutil
import time
import numpy as np
from datetime import datetime
import faiss
import logging
import pickle
from typing import List
from glob import glob
from tqdm import tqdm

from embedding import Word2VecEmbedding
from reranker import RerankerTPU
import doc_processor
from doc_processor.knowledge_file import KnowledgeFile

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class DocChatbot:
    _instance = None

    def __init__(self) -> None:
        self.llm = os.getenv("LLM_MODEL")
        self.use_reranker = True
        dev_id = 0
        if os.getenv("DEVICE_ID"):
            dev_id = int(os.getenv("DEVICE_ID"))
        else:
            logging.warning("DEVICE_ID is empty in env var, use default {}".format(dev_id))

        self.vector_db = None
        self.string_db = None
        self.files = None

        self.db_base_path = "data/db_tpu"
        # embeddings_size hard code here, can read from model output size
        self.embeddings_size = 1024
        self.embeddings = Word2VecEmbedding()
        # self.reranker = LangchainReranker(top_n=3, device='cpu', max_length=1024, model_name_or_path="../bge-reranker-large")
        self.reranker = RerankerTPU()
        logging.info("chatbot init success!")

    def docs2embedding(self, docs):
        emb = []
        for i in tqdm(range(len(docs) // 4)):
            emb += self.embeddings.embed_documents(docs[i * 4: i * 4 + 4])
        if len(docs) % 4 != 0:
            residue = docs[-(len(docs) % 4):] + [" " for _ in range(4 - len(docs) % 4)]
            emb += self.embeddings.embed_documents(residue)[:len(docs) % 4]

        return emb

    def query_from_doc(self, query_string, k=1):
        query_vec = self.embeddings.embed_query(query_string)
        _, i = self.vector_db.search(x=np.array([query_vec]), k=k)
        return [self.string_db[ind] for ind in i[0]]

    # split documents, generate embeddings and ingest to vector db
    def init_vector_db_from_documents(self, file_list: List[str]):
        docs = []
        for file in file_list:
            kb_file = KnowledgeFile(filename=file)
            doc = kb_file.docs2texts()
            docs.extend(doc)

        # 文件解析失败
        if len(docs) == 0:
            return False

        emb_num = 0
        start_time = time.time()
        if self.vector_db is None:
            self.files = ", ".join([item.split("/")[-1] for item in file_list])
            emb = self.docs2embedding([x.page_content for x in docs])
            emb = np.array(emb).astype(np.float32)
            if not emb.flags['C_CONTIGUOUS']:
                emb = np.ascontiguousarray(emb)
            emb_num = len(emb)
            self.embeddings_size = emb.shape[1]
            self.vector_db = faiss.IndexFlatL2(self.embeddings_size)
            self.vector_db.add(emb)
            self.string_db = docs
        else:
            self.files = self.files + ", " + ", ".join([item.split("/")[-1] for item in file_list])
            emb = self.docs2embedding([x.page_content for x in docs])
            emb_num = len(emb)
            self.vector_db.add(np.array(emb))
            self.string_db += docs

        logging.info("Total embedding docs time {}, embedding vector size {}, embedding vector num {}".format(time.time()- start_time, self.embeddings_size, emb_num))
        return True

    def load_vector_db_from_local(self, index_name: str):
        with open(f"{self.db_base_path}/{index_name}/db.string", "rb") as file:
            byte_stream = file.read()
        self.string_db = pickle.loads(byte_stream)
        self.vector_db = faiss.read_index(f"{self.db_base_path}/{index_name}/db.index")
        self.files = open(f"{self.db_base_path}/{index_name}/name.txt", 'r', encoding='utf-8').read()

    def save_vector_db_to_local(self):
        now = datetime.now()
        folder_name = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
        os.mkdir(f"{self.db_base_path}/{folder_name}")
        faiss.write_index(self.vector_db, f"{self.db_base_path}/{folder_name}/db.index")
        byte_stream = pickle.dumps(self.string_db)
        with open(f"{self.db_base_path}/{folder_name}/db.string", "wb") as file:
            file.write(byte_stream)
        with open(f"{self.db_base_path}/{folder_name}/name.txt", "w", encoding="utf-8") as file:
            file.write(self.files)

    def del_vector_db(self, file_name):
        shutil.rmtree(f"{self.db_base_path}/" + file_name)
        self.vector_db = None

    def get_vector_db(self):
        file_list = glob(f"{self.db_base_path}/*")
        return [x.split("/")[-1] for x in file_list]

    def time2file_name(self, path):
        return open(f"{self.db_base_path}/{path}/name.txt", 'r', encoding='utf-8').read()

    def load_first_vector_db(self):
        file_list = glob(f"{self.db_base_path}/*")
        index_name = file_list[0].split("/")[-1]
        self.load_vector_db_from_local(index_name)

    def rename(self, file_list, new_name):
        with open(f"{self.db_base_path}/{file_list}/name.txt", "w", encoding="utf-8") as file:
            file.write(new_name)

    def stream_predict(self, query, history):
        history.append((query, ''))
        res = ''
        response = "根据文件内容,这个合同的甲方(购买方)是内蒙古北方航空科技有限公司。"
        for i in response:
            res += i
            time.sleep(0.01)
            history[-1] = (query, res)
            yield res, history

    def filter_space(self, string):
        result = ""
        count = 0
        for char in string:
            if char == " " or char == '\t':
                count += 1
                if count < 4:
                    result += char
            else:
                result += char
                count = 0
        return result

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DocChatbot()
        return cls._instance

