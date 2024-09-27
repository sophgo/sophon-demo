import os
import sys

from transformers import AutoTokenizer
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union, Sequence
from langchain_core.documents import Document
import sophon.sail as sail
import configparser
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class RerankerTPU:
    def __init__(self,):
        config = configparser.ConfigParser()
        config.read('config.ini')
        reranker_model = os.getenv("RERANKER_MODEL")
        if reranker_model not in ["reranker_model", "bce_reranker"]:
            logging.warning("Reranker model only support reranker_model and bce_reranker, please checkout the env var. Use default reranker_model.")
            reranker_model = "reranker_model"
        bmodel_path = config.get(reranker_model, 'bmodel_path')
        token_path = config.get(reranker_model, 'token_path')
        dev_id = 0
        if os.getenv("DEVICE_ID"):
            dev_id = int(os.getenv("DEVICE_ID"))
        else:
            logging.warning("DEVICE_ID is empty in env var, use default {}".format(dev_id))

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = token_path, revision = None, local_files_only = False, trust_remote_code = False)
        self.net = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success, dev_id {}".format(bmodel_path, dev_id))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)
        self.top_n = 3

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
    ) -> Sequence[Document]:
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        sentence_pairs = [[query, _doc] for _doc in _docs]
        results = self.predict(sentences=sentence_pairs,
                                      convert_to_tensor=True
                                      )
        top_k = self.top_n if self.top_n < len(results) else len(results)

        values, indices = results.topk(top_k)
        final_results = []
        for value, index in zip(values, indices):
            doc = doc_list[index]
            doc.metadata["relevance_score"] = value
            final_results.append(doc)
        return final_results
    
    def predict(
        self,
        sentences: List[List[str]],
        activation_fct=None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> Union[List[float], np.ndarray, torch.Tensor]:
        
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True
            
        texts = [[] for _ in range(len(sentences[0]))]
        for example in sentences:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())
        

        if activation_fct is None:
            activation_fct = torch.nn.modules.activation.Sigmoid()
        tokenized = self.tokenizer(
            *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=1024
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        pred_scores = []
        input_ids, attention_mask = input_ids.numpy(), attention_mask.numpy()
        if input_ids.shape[1] > 512:
            input_ids = input_ids[:, :512]
            attention_mask = attention_mask[:, :512]
        elif input_ids.shape[1] < 512:
            input_ids = np.pad(input_ids,
                        ((0, 0), (0, 512 - input_ids.shape[1])),
                        mode='constant', constant_values=1)
            attention_mask = np.pad(attention_mask,
                        ((0, 0), (0, 512 - attention_mask.shape[1])),
                        mode='constant', constant_values=0)
        input_data = {self.input_names[0]: input_ids,
                      self.input_names[1]: attention_mask}
        outputs = self.net.process(self.graph_name, input_data)
        model_predictions = torch.from_numpy(outputs[self.output_names[0]])
        logits = activation_fct(model_predictions)

        pred_scores.extend(logits)

        pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores