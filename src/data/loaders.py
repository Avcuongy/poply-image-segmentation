import pandas as pd
import torch
from torch.utils.data import DataLoader

from .download import DEFAULT_OUTPUT, download_dataset, validate_dataset
from .datasets import (
    PolypSegmentationDataset,
    MeanTeacherLabeledDataset,
    MeanTeacherUnlabeledDataset,
)
from . import transforms as T


def build_dataloaders(
    data_root=DEFAULT_OUTPUT,
    *,
    method="supervised",
    subset="half",
    image_size=256,
    batch_size=8,
    num_workers=0,
    seed=42,
    download=True,
    normalize_input=True,
    input_mean=(0.485, 0.456, 0.406),
    input_std=(0.229, 0.224, 0.225),
):
    if method not in ("supervised", "mean_teacher", "mean_teacher_infonce"):
        raise ValueError("Unknown training method")
    if subset not in ("half", "quarter"):
        raise ValueError("subset must be half or quarter")
    root = (
        download_dataset(output_dir=data_root)
        if download
        else validate_dataset(data_root)[0]
    )
    height, width = (
        (image_size, image_size) if isinstance(image_size, int) else image_size
    )
    frames = {
        name: pd.read_csv(root / "sample" / f"{name}.csv")
        for name in ("train", "valid", "test")
    }
    datasets = {}
    for split in ("valid", "test"):
        datasets[split] = PolypSegmentationDataset(
            frames[split],
            root,
            root,
            transform=T.build_eval_transform(
                height, width, seed, normalize_input, input_mean, input_std
            ),
            image_col="image_path",
            mask_col="mask_path",
        )
    if method == "supervised":
        datasets["train"] = PolypSegmentationDataset(
            frames["train"],
            root,
            root,
            transform=T.build_supervised_transform(height, width, seed),
            image_col="image_path",
            mask_col="mask_path",
        )
    else:
        flag = "labeled_1_2" if subset == "half" else "labeled_1_4"
        train = frames["train"]
        if flag not in train or not train[flag].isin([0, 1]).all():
            raise ValueError(f"Training CSV requires binary column {flag}")
        datasets["labeled"] = MeanTeacherLabeledDataset(
            train[train[flag] == 1],
            root,
            root,
            T.build_labeled_shared_geometry(height, width, seed),
            T.build_labeled_weak_transform(seed),
            T.build_labeled_strong_transform(seed),
            image_col="image_path",
            mask_col="mask_path",
        )
        datasets["unlabeled"] = MeanTeacherUnlabeledDataset(
            train[train[flag] == 0],
            root,
            T.build_unlabeled_shared_geometry(height, width, seed),
            T.build_unlabeled_weak_transform(seed),
            T.build_unlabeled_strong_transform(seed),
            image_col="image_path",
        )
    loaders = {}
    for name, dataset in datasets.items():
        if len(dataset) == 0:
            raise ValueError(f"Empty dataset: {name}")
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name not in ("valid", "test"),
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed),
        )
    return loaders
