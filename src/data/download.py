import csv
from pathlib import Path

DEFAULT_DATASET = "avcuongy/bkai-igh-neopolyp-small-sample"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "kagglehub"


def validate_dataset(root):
    """Check every image/mask referenced by the three split CSVs."""
    root = Path(root)
    data_dir = root / "data" if (root / "data" / "sample").is_dir() else root
    counts = {}
    for split in ("train", "valid", "test"):
        csv_path = data_dir / "sample" / f"{split}.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"image_path", "mask_path"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{csv_path}: missing image_path/mask_path columns")
            count = 0
            for row in reader:
                for column in ("image_path", "mask_path"):
                    value = row.get(column)
                    if not value or not (data_dir / value).is_file():
                        raise FileNotFoundError(
                            f"{csv_path}: missing {column}={value!r}"
                        )
                count += 1
        if count == 0:
            raise ValueError(f"Empty dataset split: {csv_path}")
        counts[split] = count
    return data_dir, counts


def download_dataset(
    handle=DEFAULT_DATASET, output_dir=DEFAULT_OUTPUT, force=False, logger=None
):
    from utils.logger import get_logger

    logger = logger or get_logger(__name__, "data")
    output_dir = Path(output_dir).expanduser().resolve()

    if handle != DEFAULT_DATASET and output_dir == DEFAULT_OUTPUT.resolve():
        raise ValueError("Use --output-dir for a different dataset handle.")
    if not force and output_dir.exists():
        try:
            data_dir, counts = validate_dataset(output_dir)
        except (OSError, ValueError) as exc:
            logger.warning("Local dataset incomplete: %s; downloading again", exc)
            force = True
        else:
            logger.info("Using existing data: %s | splits=%s", data_dir, counts)
            return data_dir
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements.txt (kagglehub is required for downloads)."
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s into %s", handle, output_dir)
    resolved = Path(
        kagglehub.dataset_download(
            handle,
            output_dir=str(output_dir),
            force_download=force,
        )
    ).resolve()
    if not resolved.is_relative_to(output_dir):
        raise RuntimeError(
            f"Download returned a path outside the requested directory: {resolved}"
        )
    data_dir, counts = validate_dataset(resolved)
    logger.info("Download validated: %s | splits=%s", data_dir, counts)
    return data_dir
