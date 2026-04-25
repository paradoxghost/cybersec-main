"""Logging configuration helper."""
import logging
import logging.config
from pathlib import Path

import yaml


def configure_logging(config_path: str | Path = "configs/logging.yaml") -> None:
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(level=logging.INFO)
