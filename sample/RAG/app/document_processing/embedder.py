import os
from typing import List

import numpy as np
import openai

from app.engine.config import DocConfig
from app.utils.logger import get_logger

from FlagEmbedding import FlagAutoModel

logger = get_logger(__name__)


class Embedder:
    @classmethod
    def create(cls, config: DocConfig):
        if config.api_url:
            return EmbedderApi(config)
        else:
            return EmbedderLocal(config)

    def embed(self, text: List[str]) -> np.ndarray:
        raise NotImplementedError()


class EmbedderApi(Embedder):
    def __init__(self, config: DocConfig):
        self.model_name = config.embedding_model
        self.dimension = config.dimension
        api_key = "api_key" if config.api_key == "" else config.api_key
        print(f'{config.api_url=}')
        self.client = openai.OpenAI(api_key=api_key, base_url=config.api_url)
        logger.info(f"EmbedderAPI initialized, api_url: {config.api_url}")

    def embed(self, text: List[str]) -> np.ndarray:
        """calculate embedding of input text

        Args:
            text (List[str]): list of input text

        Returns:
            List: list of embeddings
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
            dimensions=self.dimension,
        )
        embeddings = [d.embedding for d in response.data]
        return np.array(embeddings)


class EmbedderLocal(Embedder):
    def __init__(self, config: DocConfig):
        self.model_name = config.embedding_model
        self.model_path = config.embedding_model_path
        if os.path.exists(self.model_path):
            logger.info(f"Offline loading, model path is {self.model_path}")
            self.model = FlagAutoModel.from_finetuned(
                self.model_path,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )
        else:
            logger.info(f"Online loading, model name is {self.model_name}")
            self.model = FlagAutoModel.from_finetuned(
                self.model_name,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )

    def embed(self, text: List[str]) -> np.ndarray:
        """calculate embedding of input text

        Args:
            text (List[str]): list of input text

        Returns:
            List: list of embeddings
        """
        embeddings = self.model.encode(text)
        return embeddings
