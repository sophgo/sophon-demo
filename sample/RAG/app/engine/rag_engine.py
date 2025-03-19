import os
from typing import Dict, List, Union

from app.document_processing.doc_processor import (DocProcessor,
                                                   check_if_support_docx)
from app.engine.config import RAGConfig
from app.llm.generator.generator import Generator
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGEngine:
    def __init__(self, config: RAGConfig):
        self.generator = Generator.from_config(config.llm_config)
        self.doc_processor = DocProcessor(config.doc_config)
        logger.info(f"RAGEngine is initialized with config {config}")

    def _add_single_file(self, file_path: str) -> bool:
        """add a document to rag system

        Args:
            file_path (str): file path

        Returns:
            whether is successfully added
        """
        real_path = os.path.abspath(file_path)
        logger.info(f"Add document: {real_path}")
        try:
            self.doc_processor.process_document(real_path)
            logger.info(f"Document added successfully: {real_path}")
            return True
        except:
            logger.error(f"Failed to add document: {real_path}")
            return False

    def add_doc(self, file_path: Union[str, List[str]]) -> bool:
        """add a document or a list of documents to rag system

        Args:
            file_path (Union[str, List[str]]): file path(s)

        Returns:
            whether is successfully added
        """
        if isinstance(file_path, str):
            return self._add_single_file(file_path)
        elif isinstance(file_path, List[str]):
            for file in file_path:
                ret = self._add_single_file(file)
                if not ret:
                    return False
            return True

    def remove_doc(self, file_path: str) -> bool:
        """remove a document from rag system

        Args:
            file_path (str): file path

        Returns:
            bool: whether is successfully removed
        """
        real_path = os.path.abspath(file_path)
        logger.info(f"Remove document: {real_path}")
        try:
            self.doc_processor.remove_document(real_path)
            logger.info(f"Document removed successfully: {real_path}")
            return True
        except:
            logger.error(f"Failed to remove document: {real_path}")
            return False

    def query(self, question: str) -> Dict:
        """generate answer to the question from user

        Args:
            question (str): user question
        """
        try:
            results = self.doc_processor.search_ralated_chunk(question)
            context = [tup[1] for tup in results]
            context = "\n".join(context)
            # 这个接口是不是做成generate(context, question) ?
            # 还有对话历史
            answer = self.generator.generate(
                context
                + "以上是检索到的参考文本，请根据你的知识和检索结果回答以下问题\n"
                + question
            )
            return {"answer": answer, "reference": results}
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {"answer": None, "reference": None}

    def check_query_stream_support(self) -> bool:
        """check if llm backend supports generate stream

        Returns:
            bool: support or not
        """
        return self.generator.check_query_stream_support()

    def query_stream(self, question: str):
        """generate answer to question from user, in stream mode

        Args:
            question (str): user question
        """
        try:
            results = self.doc_processor.search_ralated_chunk(question)
            context = [tup[1] for tup in results]
            context = "\n".join(context)
            stream = self.generator.generate_stream(
                context
                + "以上是检索到的参考文本，请根据你的知识和检索结果回答以下问题\n"
                + question
            )
            for chunk in stream:
                if len(chunk.choices) == 0:
                    continue
                partial_answer = chunk.choices[0].delta.content
                if partial_answer is not None:
                    yield {"answer": partial_answer, "reference": results}
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            yield {"answer": None, "reference": None}

    def get_status(self) -> str:
        """get status of service

        Returns:
            bool: True if alive, else False
        """

        return {"status": "alive"}

    def get_doc_list(self) -> List[str]:
        """get all documents stored in database

        Returns:
            List[str]: list of document name
        """
        return self.doc_processor.get_doc_list()

    def update_topk(self, topk: int):
        """update topk param while retrieval

        Args:
            topk (int): _description_
        """
        self.doc_processor.update_topk(topk)

    def check_if_support_docx(self) -> bool:
        return check_if_support_docx()
