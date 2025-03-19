import importlib
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import numpy as np
from app.engine.config import DocConfig
from app.utils.logger import get_logger

DATABASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DATABASE_DIR)

DATABASE_CONFIG_MODULENAME_CLASSNAME_MAP = {
    "faiss": ("database_faiss", "DatabaseFaiss")
}

logger = get_logger(__name__)


class Database(ABC):

    def __init__(self, config: DocConfig):
        self.dimension = config.dimension
        self.topk = config.topk
        self.index_map = {}

    @abstractmethod
    def add_vector(self, filename: str, vectors: np.array):
        """add vector to faiss database

        Args:
            filename (str): filename
            vectors (np.array): all embeddings, shape like [num of vectors, dimension]

        Raises:
            ValueError: dimension mismatch
        """
        raise NotImplementedError("add_vector must be implemented in subclasses.")

    @abstractmethod
    def search(self, query: np.array, k: int = 5) -> List[Tuple[str, int, float]]:
        """search for top k vectors with max similarity

        Args:
            query_vector: query vector

        Returns:
            List[Tuple[str, int, float]]: a list of (filename, chunk id, similarity)
        """
        raise NotImplementedError("search must be implemented in subclasses.")

    @classmethod
    def from_config(cls, config: DocConfig):
        """create subclass instance from config

        Args:
            config (DocConfig): config

        Raises:
            ValueError: if invalid subclass name

        Returns:
            _type_: subclass instance
        """
        database_method = config.database_method.lower()
        if database_method not in DATABASE_CONFIG_MODULENAME_CLASSNAME_MAP:
            raise ValueError(f"Invalid generator backend type: {database_method}")
        module_name, class_name = DATABASE_CONFIG_MODULENAME_CLASSNAME_MAP[
            database_method
        ]
        module = importlib.import_module(module_name)
        derived_class = getattr(module, class_name, None)
        if not derived_class or not issubclass(derived_class, cls):
            raise ValueError(
                f"Class {class_name} not found or is not subclass of {cls}"
            )
        return derived_class(config)

    def remove_vectors(self, filename: str):
        """remove vectors by filename

        Args:
            filename (str): filename
        """
        self.index_map.pop(filename)

    def update_topk(self, topk: int):
        """update topk param while retrieval

        Args:
            topk (int): _description_
        """
        self.topk = topk
