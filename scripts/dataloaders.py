import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.download import DEFAULT_DATASET, DEFAULT_OUTPUT, download_dataset
from utils.logger import get_logger


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download and validate polyp segmentation data."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if local data is complete.",
    )
    args = parser.parse_args(argv)
    logger = get_logger("dataloaders", "data")
    try:
        download_dataset(args.dataset, args.output_dir, args.force, logger)
    except Exception:
        logger.exception("Dataset preparation failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
