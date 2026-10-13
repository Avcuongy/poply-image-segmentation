import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.logger import get_logger
from utils.setup_folder import setup_folder


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Set up project folders and log files."
    )
    parser.parse_args(argv)
    logger = get_logger("config", "config")
    try:
        setup_folder()
    except Exception:
        logger.exception("Project folder setup failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
