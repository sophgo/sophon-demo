import os
from typing import List

from app.engine.config import DocConfig
from app.utils.logger import get_logger

from FlagEmbedding import FlagAutoModel

logger = get_logger(__name__)


class Embedder:
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

    def embed(self, text: List[str]) -> List[List[float]]:
        """calculate embedding of input text

        Args:
            text (List[str]): list of input text

        Returns:
            List: list of embeddings
        """
        embeddings = self.model.encode(text)
        return embeddings
