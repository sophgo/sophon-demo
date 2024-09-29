# -*- coding: utf-8 -*-
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
from typing import List
from .sentence_model import SentenceModel


class Word2VecEmbedding():
    model = SentenceModel()

    def embed_query(self, text: str) -> List[float]:
        embeddings_tpu = self.model.encode_tpu([text, "", "", ""])
        return embeddings_tpu.tolist()[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings_tpu = self.model.encode_tpu(texts)
        return embeddings_tpu.tolist()

