import importlib
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Type

from app.engine.config import DocConfig
from app.utils.logger import get_logger

DOCSPLITTER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DOCSPLITTER_DIR)

DOCSPLITTER_CONFIG_MODULENAME_CLASSNAME_MAP = {
    "fixedlength": ("fixed_len_splitter", "FixedLengthSplitter")
}

logger = get_logger(__name__)


class DocSplitterBase(ABC):

    _subclasses: Dict[str, Type["DocSplitterBase"]] = {}

    def __init__(self, config: DocConfig):
        self.method = config.split_method
        logger.info(f"Document split method: {self.method}")

    @classmethod
    def register_subclass(cls, class_name, subclass: Type["DocSplitterBase"]):
        """register a subclass"""
        cls._subclasses[class_name] = subclass

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError("split_text must be implemented in subclasses.")

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
        splitter_type = config.split_method.lower()
        if splitter_type not in DOCSPLITTER_CONFIG_MODULENAME_CLASSNAME_MAP:
            raise ValueError(f"Invalid generator backend type: {splitter_type}")
        module_name, class_name = DOCSPLITTER_CONFIG_MODULENAME_CLASSNAME_MAP[
            splitter_type
        ]
        module = importlib.import_module(module_name)
        derived_class = getattr(module, class_name, None)
        if not derived_class or not issubclass(derived_class, cls):
            raise ValueError(
                f"Class {class_name} not found or is not subclass of {cls}"
            )
        return derived_class(config)
