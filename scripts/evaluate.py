import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from data.download import DEFAULT_OUTPUT
from data.loaders import build_dataloaders
from evaluation.checkpoints import MODEL_TYPES, build_and_load_model
from evaluation.metrics import evaluate_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model-type", required=True, choices=MODEL_TYPES)
    parser.add_argument("--branch", choices=("student", "teacher"), default="teacher")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "evaluation",
    )
    args = parser.parse_args(argv)
    model, config, _ = build_and_load_model(
        args.checkpoint,
        args.model_type,
        {
            "image_height": args.image_size,
            "image_width": args.image_size,
            "mean_teacher_state": args.branch,
        },
    )
    loaders = build_dataloaders(
        args.data_root,
        image_size=(config["image_height"], config["image_width"]),
        batch_size=args.batch_size,
        normalize_input=config.get("normalize_input", True),
        input_mean=config.get("input_mean", (0.485, 0.456, 0.406)),
        input_std=config.get("input_std", (0.229, 0.224, 0.225)),
    )
    names = (
        ["background", "non-neoplastic", "neoplastic"]
        if config["num_classes"] == 3
        else None
    )
    table, matrix = evaluate_model(
        model,
        args.checkpoint.stem,
        loaders["test"],
        args.device,
        config["num_classes"],
        names,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "metrics.csv", index=False)
    np.savetxt(
        args.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d"
    )
    print(table.to_string(index=False))
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
