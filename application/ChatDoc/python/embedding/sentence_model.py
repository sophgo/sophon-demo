# -*- coding: utf-8 -*-
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
from enum import Enum
from typing import List, Union, Optional
import numpy as np
import torch
from tqdm.autonotebook import trange
from transformers import AutoTokenizer
import configparser
import logging
import os

from .npuengine import EngineOV
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class EncoderType(Enum):
    FIRST_LAST_AVG = 0
    LAST_AVG = 1
    CLS = 2
    POOLER = 3
    MEAN = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError()


class SentenceModel:
    def __init__(
            self,
            model_name_or_path: str = "shibing624/text2vec-base-chinese",
            encoder_type: Union[str, EncoderType] = "MEAN",
            max_seq_length: int = 256,
            device: Optional[str] = None,
    ):
        """
        Initializes the base sentence model.

        :param model_name_or_path: The name of the model to load from the huggingface models library.
        :param encoder_type: The type of encoder to use, See the EncoderType enum for options:
            FIRST_LAST_AVG, LAST_AVG, CLS, POOLER(cls + dense), MEAN(mean of last_hidden_state)
        :param max_seq_length: The maximum sequence length.
        :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if GPU.

        bert model: https://huggingface.co/transformers/model_doc/bert.html?highlight=bert#transformers.BertModel.forward
        BERT return: <last_hidden_state>, <pooler_output> [hidden_states, attentions]
        Note that: in doc, it says <last_hidden_state> is better semantic summery than <pooler_output>.
        thus, we use <last_hidden_state>.
        """
        config = configparser.ConfigParser()
        config.read('config.ini')
        embedding_model = os.getenv("EMBEDDING_MODEL")
        if embedding_model == "bert_model":
            self.model_name_or_path = "shibing624/text2vec-base-chinese"
        elif embedding_model == "bce_embedding":
            self.model_name_or_path = "maidalun1020/bce-embedding-base_v1"
        else:
            logging.warning("Embedding model only support bert_model and bce_embedding, please checkout the env var. Use default bert_model.")
            embedding_model = "bert_model"
        bmodel_path = config.get(embedding_model, 'bmodel_path')
        token_path = config.get(embedding_model, 'token_path')
        dev_id = 0
        if os.getenv("DEVICE_ID"):
            dev_id = int(os.getenv("DEVICE_ID"))
        else:
            logging.warning("DEVICE_ID is empty in env var, use default {}".format(dev_id))
        encoder_type = EncoderType.from_string(encoder_type) if isinstance(encoder_type, str) else encoder_type
        if encoder_type not in list(EncoderType):
            raise ValueError(f"encoder_type must be in {list(EncoderType)}")
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)

        self.net = EngineOV(model_path=bmodel_path, device_id=dev_id)
        self.net.padding_to = 512


    def __str__(self):
        return f"<SentenceModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, " \
               f"max_seq_length: {self.max_seq_length}, emb_dim: {self.get_sentence_embedding_dimension()}>"

    def get_sentence_embedding_dimension(self):
        """
        Get the dimension of the sentence embeddings.

        Returns
        -------
        int or None
            The dimension of the sentence embeddings, or None if it cannot be determined.
        """
        # Use getattr to safely access the out_features attribute of the pooler's dense layer
        return getattr(self.net.pooler.dense, "out_features", None)

    def get_sentence_embeddings_tpu(self, input_ids, attention_mask, token_type_ids=None):
        """
        Returns the model output by encoder_type as embeddings.

        Utility function for self.net() method.
        """
        input_ids, attention_mask = input_ids.numpy(), attention_mask.numpy()
        if input_ids.shape[1] > self.net.padding_to:
            input_ids = input_ids[:, :self.net.padding_to]
            attention_mask = attention_mask[:, :self.net.padding_to]
        elif input_ids.shape[1] < self.net.padding_to:
            input_ids = np.pad(input_ids,
                            ((0, 0), (0, self.net.padding_to - input_ids.shape[1])),
                            mode='constant', constant_values=0)
            attention_mask = np.pad(attention_mask,
                                ((0, 0), (0, self.net.padding_to - attention_mask.shape[1])),
                                mode='constant', constant_values=0)
        kwargs = {
            "input_ids": input_ids.astype(np.float32),
            "attention_mask": attention_mask.astype(np.float32)
        }
        if token_type_ids is not None:
            token_type_ids = token_type_ids.numpy()
            if token_type_ids.shape[1] > self.net.padding_to:
                token_type_ids = token_type_ids[:, :self.net.padding_to]
            elif token_type_ids.shape[1] < self.net.padding_to:
                token_type_ids = np.pad(token_type_ids,
                                    ((0, 0), (0, self.net.padding_to - token_type_ids.shape[1])),
                                    mode='constant', constant_values=0)
            kwargs["token_type_ids"] = token_type_ids.astype(np.float32)
        model_output = self.net(**kwargs)

        if self.encoder_type == EncoderType.MEAN:
            """
            Mean Pooling - Take attention mask into account for correct averaging
            """
            token_embeddings = torch.from_numpy(model_output)  # Contains all token embeddings
            input_mask_expanded = torch.from_numpy(attention_mask).unsqueeze(-1).expand(token_embeddings.size()).float()
            final_encoding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)
            return final_encoding  # [batch, hid_size]
        else:
            raise NotImplementedError

    def encode_tpu(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = False,
            max_seq_length: int = None,
    ):
        """
        Returns the embeddings for a batch of sentences.

        :param sentences: str/list, Input sentences
        :param batch_size: int, Batch size
        :param show_progress_bar: bool, Whether to show a progress bar for the sentences
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which device to use for the computation
        :param normalize_embeddings: If true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :param max_seq_length: Override value for max_seq_length
        """
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        if convert_to_tensor:
            convert_to_numpy = False
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            # Compute sentences embeddings
            with torch.no_grad():
                features = self.tokenizer(
                    sentences_batch, max_length=max_seq_length,
                    padding=True, truncation=True, return_tensors='pt'
                )
                embeddings = self.get_sentence_embeddings_tpu(**features)
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                if convert_to_numpy:
                    embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings

