from __future__ import annotations

from pathlib import Path
import logging
import sys
from utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_FILE = [
    "config.log",
]
DATA_SUBFOLDERS = [
    # completed layer
    "completed/db",
    "completed/ohlcs",
]

logger = get_logger(__name__, "config")


def _ensure_data_folders() -> None:
    DATA_ROOT.mkdir(exist_ok=True)
    for relative in DATA_SUBFOLDERS:
        folder = DATA_ROOT / relative
        folder.mkdir(parents=True, exist_ok=True)


def _ensure_logs_folder() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    for log_file in LOGS_FILE:
        log_path = LOGS_DIR / log_file
        if not log_path.exists():
            log_path.touch()


def setup_folder() -> None:
    logger.info("[Config] Setup data folders at %s", DATA_ROOT)
    _ensure_data_folders()
    logger.info("[Config] Setup logs folder at %s", LOGS_DIR)
    _ensure_logs_folder()


if __name__ == "__main__":
    setup_folder()
