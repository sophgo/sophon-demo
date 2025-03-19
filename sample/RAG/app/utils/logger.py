import json
import logging
import logging.config
from pathlib import Path


def setup_logging(config_path: str = "logging_config.json"):
    """
    init logger config
    :param config_path: config path
    """
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        raise FileNotFoundError(f"Logging config file not found: {config_path}")


def get_logger(name: str) -> logging.Logger:
    """
    get logger by name
    :param name: logger name
    :return: Logger object
    """
    return logging.getLogger(name)
