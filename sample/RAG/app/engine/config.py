import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GeneratorConfig:
    model: str
    backend_type: Optional[str] = "api"
    path: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class DocConfig:
    embedding_model: str
    embedding_model_path: str
    split_method: str
    chunk_length: int
    overlap: int
    database_method: str
    dimension: int
    topk: int


@dataclass
class RAGConfig:
    llm_config: GeneratorConfig
    doc_config: DocConfig

    @classmethod
    def from_json(cls, config_path: str) -> "RAGConfig":
        """
        load RAGConfig from json file
        :param config_path: json file path
        :return: RAGConfig object
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = json.load(f)

        return cls(
            llm_config=GeneratorConfig(**config_data["llm_config"]),
            doc_config=DocConfig(**config_data["doc_config"]),
        )
