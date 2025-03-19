from typing import List, Tuple

import faiss
import numpy as np
from app.document_processing.database.database import Database
from app.engine.config import DocConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseFaiss(Database):
    def __init__(self, config: DocConfig):
        super().__init__(config)

    def add_vector(self, filename: str, vectors: np.array):
        if self.dimension != vectors.shape[1]:
            logger.error(
                f"Dimension mismatch! Database dimension is {self.dimension} and embeddings dimension is {len[vectors[0]]}"
            )
            raise ValueError()
        self.index_map[filename] = faiss.IndexFlatL2(self.dimension)
        self.index_map[filename].add(vectors)

    def search(self, query_vector: np.array) -> List[Tuple[str, int, float]]:
        if not self.index_map:
            logger.warning("No vectors in the database to search.")
            return {}

        global_results = []

        for filename, index in self.index_map.items():
            num = index.ntotal if index.ntotal < self.topk else self.topk
            distances, indices = index.search(query_vector, num)
            for i, idx in enumerate(indices[0]):
                global_results.append((filename, idx, distances[0][i]))

            global_results.sort(key=lambda x: x[2])
        return global_results[: min(self.topk, len(global_results))]
