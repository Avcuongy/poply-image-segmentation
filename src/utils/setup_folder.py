from __future__ import annotations

from pathlib import Path
import logging
import sys
from utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_FILE = [
    "config.log",
]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"

logger = get_logger(__name__, "config")


def _ensure_logs_folder() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    for log_file in LOGS_FILE:
        log_path = LOGS_DIR / log_file
        if not log_path.exists():
            log_path.touch()


def _ensure_data_folder() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def _ensure_checkpoints_folder() -> None:
    CHECKPOINTS_DIR.mkdir(exist_ok=True)


def _ensure_models_folder() -> None:
    MODELS_DIR.mkdir(exist_ok=True)


def setup_folder() -> None:
    logger.info("[Config] Setup project folders")
    _ensure_logs_folder()
    _ensure_data_folder()
    _ensure_checkpoints_folder()
    _ensure_models_folder()
    logger.info("[Config] Setup project folders is complete")


if __name__ == "__main__":
    setup_folder()
