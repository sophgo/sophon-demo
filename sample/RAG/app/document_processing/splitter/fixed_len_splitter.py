from typing import List

from app.document_processing.splitter.doc_splitter import DocSplitterBase
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FixedLengthSplitter(DocSplitterBase):
    def __init__(self, config):
        super().__init__(config)
        self.chunk_length = config.chunk_length
        self.overlap = config.overlap

    def split_text(self, text: str) -> List[str]:
        """split text into chunks with fixed length

        Args:
            text (str): input text

        Returns:
            List[str]: list of chunks after splitting
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_length
            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks
