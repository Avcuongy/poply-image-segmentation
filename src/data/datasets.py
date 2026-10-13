from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def decode_ground_truth(mask_rgb):
    """
    To RGB mask -> class-index mask 0/1/2.

    Class mapping:
        0 = background      -> black
        1 = non-neoplastic  -> green
        2 = neoplastic      -> red

    Using nearest-color matching to handle or correct any color variations in the mask image.
    This function will convert an RGB mask into a 2D array where each pixel's value corresponds to its class ID (0, 1, or 2).
    """
    mask = np.asarray(mask_rgb)

    if mask.ndim == 2:
        unique_values = np.unique(mask)

        if np.isin(unique_values, [0, 1, 2]).all():
            return mask.astype(np.uint8)

        raise ValueError(
            "Grayscale mask must contain only class IDs 0, 1, 2. "
            f"Found: {unique_values[:20]}"
        )

    if mask.ndim != 3 or mask.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB mask, got shape={mask.shape}")

    colors = np.array(
        [
            [0, 0, 0],  # background
            [0, 255, 0],  # non-neoplastic
            [255, 0, 0],  # neoplastic
        ],
        dtype=np.float32,
    )

    pixels = mask.astype(np.float32)

    # Shape:
    # pixels -> [H, W, 1, 3]
    # colors -> [1, 1, 3, 3]
    # result -> [H, W, 3]
    diff = pixels[:, :, None, :] - colors[None, None, :, :]

    distances = np.sum(
        diff**2,
        axis=-1,
    )

    class_mask = np.argmin(
        distances,
        axis=-1,
    )

    return class_mask.astype(np.uint8)


def _pick_column(
    df,
    explicit_column=None,
    candidates=None,
    required=True,
):
    if explicit_column is not None:
        if explicit_column not in df.columns:
            raise KeyError(
                f"Column '{explicit_column}' not found. "
                f"Available: {df.columns.tolist()}"
            )

        return explicit_column

    candidates = candidates or []

    normalized = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    if len(df.columns) == 1:
        return df.columns[0]

    if required:
        raise KeyError(
            "Cannot infer CSV column automatically. "
            f"Available columns: {df.columns.tolist()}"
        )

    return None


class PolypSegmentationDataset(Dataset):
    IMAGE_COLUMN_CANDIDATES = [
        "image",
        "image_name",
        "image_path",
        "filename",
        "file_name",
        "file",
        "id",
    ]

    MASK_COLUMN_CANDIDATES = [
        "mask",
        "mask_name",
        "mask_path",
        "label",
        "label_path",
        "gt",
        "gt_path",
    ]

    def __init__(
        self,
        dataframe,
        image_dir,
        mask_dir=None,
        transform=None,
        image_col=None,
        mask_col=None,
        has_mask=True,
    ):
        self.df = dataframe.reset_index(drop=True).copy()

        self.image_dir = Path(image_dir)

        self.mask_dir = Path(mask_dir) if mask_dir is not None else None

        self.transform = transform
        self.has_mask = has_mask

        self.image_col = _pick_column(
            self.df,
            explicit_column=image_col,
            candidates=self.IMAGE_COLUMN_CANDIDATES,
            required=True,
        )

        if self.has_mask:
            if self.mask_dir is None:
                raise ValueError("mask_dir is required when has_mask=True.")

            self.mask_col = _pick_column(
                self.df,
                explicit_column=mask_col,
                candidates=self.MASK_COLUMN_CANDIDATES,
                required=True,
            )

            if self.mask_col == self.image_col:
                raise ValueError(
                    "Mask column resolved to the same column as the image column. "
                    "Provide an explicit mask_col or fix the CSV schema."
                )

        else:
            self.mask_col = None

    @staticmethod
    def _resolve_path(root_dir, value):
        candidate = Path(str(value))

        if candidate.is_absolute():
            return candidate

        return Path(root_dir) / str(value)

    def __len__(self):
        return len(self.df)

    def get_image_path(self, index):
        value = self.df.iloc[index][self.image_col]

        return self._resolve_path(
            self.image_dir,
            value,
        )

    def get_mask_path(self, index):
        if not self.has_mask:
            return None

        row = self.df.iloc[index]
        mask_value = row[self.mask_col]

        return self._resolve_path(
            self.mask_dir,
            mask_value,
        )

    def __getitem__(self, index):
        image_path = self.get_image_path(index)

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        if not self.has_mask:
            if self.transform is not None:
                transformed = self.transform(image=image)

                image = transformed["image"]

            return {
                "image": image,
                "image_path": str(image_path),
                "image_name": image_path.name,
            }

        mask_path = self.get_mask_path(index)

        mask_raw = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED,
        )

        if mask_raw is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        if mask_raw.ndim == 3:
            mask_raw = cv2.cvtColor(
                mask_raw,
                cv2.COLOR_BGR2RGB,
            )

        # Nearest-color RGB -> class ID.
        mask = decode_ground_truth(mask_raw)

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask.long(),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_name": image_path.name,
        }


class MeanTeacherUnlabeledDataset(Dataset):
    """Return aligned weak and strong views of the same unlabeled image."""

    IMAGE_COLUMN_CANDIDATES = PolypSegmentationDataset.IMAGE_COLUMN_CANDIDATES

    def __init__(
        self,
        dataframe,
        image_dir,
        shared_geometry,
        weak_transform,
        strong_transform,
        image_col=None,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.shared_geometry = shared_geometry
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.image_col = _pick_column(
            self.df,
            explicit_column=image_col,
            candidates=self.IMAGE_COLUMN_CANDIDATES,
            required=True,
        )

    def __len__(self):
        return len(self.df)

    def get_image_path(self, index):
        value = self.df.iloc[index][self.image_col]
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else self.image_dir / str(value)

    def __getitem__(self, index):
        image_path = self.get_image_path(index)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Geometry is sampled exactly once. Only photometric appearance differs.
        aligned_image = self.shared_geometry(image=image)["image"]
        weak_image = self.weak_transform(image=aligned_image)["image"]
        strong_image = self.strong_transform(image=aligned_image)["image"]

        return {
            "weak_image": weak_image,
            "strong_image": strong_image,
            "image_path": str(image_path),
            "image_name": image_path.name,
        }


class MeanTeacherLabeledDataset(Dataset):
    """Return aligned weak (Teacher) and strong (Student) views of the same
    labeled image, plus the mask aligned to the shared geometry.

    This mirrors MeanTeacherUnlabeledDataset so labeled examples also
    receive a consistency signal, matching the original Mean Teacher design
    where the consistency cost is evaluated on all input data, not only the
    unlabeled subset.
    """

    IMAGE_COLUMN_CANDIDATES = PolypSegmentationDataset.IMAGE_COLUMN_CANDIDATES
    MASK_COLUMN_CANDIDATES = PolypSegmentationDataset.MASK_COLUMN_CANDIDATES

    def __init__(
        self,
        dataframe,
        image_dir,
        mask_dir,
        shared_geometry,
        weak_transform,
        strong_transform,
        image_col=None,
        mask_col=None,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.shared_geometry = shared_geometry
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.image_col = _pick_column(
            self.df,
            explicit_column=image_col,
            candidates=self.IMAGE_COLUMN_CANDIDATES,
            required=True,
        )
        self.mask_col = _pick_column(
            self.df,
            explicit_column=mask_col,
            candidates=self.MASK_COLUMN_CANDIDATES,
            required=True,
        )

        if self.mask_col == self.image_col:
            raise ValueError(
                "Mask column resolved to the same column as the image column. "
                "Provide an explicit mask_col or fix the CSV schema."
            )

    def __len__(self):
        return len(self.df)

    def get_image_path(self, index):
        value = self.df.iloc[index][self.image_col]
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else self.image_dir / str(value)

    def get_mask_path(self, index):
        row = self.df.iloc[index]
        mask_value = row[self.mask_col]
        candidate = Path(str(mask_value))
        return candidate if candidate.is_absolute() else self.mask_dir / str(mask_value)

    def __getitem__(self, index):
        image_path = self.get_image_path(index)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_path = self.get_mask_path(index)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        if mask_raw.ndim == 3:
            mask_raw = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2RGB)
        mask = decode_ground_truth(mask_raw)

        # Geometry sampled once so mask, weak view and strong view stay aligned.
        aligned = self.shared_geometry(image=image, mask=mask)
        aligned_image, aligned_mask = aligned["image"], aligned["mask"]

        weak_image = self.weak_transform(image=aligned_image)["image"]
        strong_image = self.strong_transform(image=aligned_image)["image"]

        if not torch.is_tensor(aligned_mask):
            aligned_mask = torch.from_numpy(aligned_mask)

        return {
            "weak_image": weak_image,
            "strong_image": strong_image,
            "mask": aligned_mask.long(),
            "image_index": int(index),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_name": image_path.name,
        }


def validate_labeled_dataset(dataset, dataset_name="dataset", max_samples=8):
    """Fail early if image/mask columns or representative mask paths are wrong."""
    if getattr(dataset, "mask_col", None) is None:
        raise ValueError(f"[{dataset_name}] mask_col was not resolved.")

    if dataset.image_col == dataset.mask_col:
        raise ValueError(
            f"[{dataset_name}] image_col and mask_col are both '{dataset.image_col}'."
        )

    if len(dataset) == 0:
        raise ValueError(f"[{dataset_name}] dataset is empty.")

    sample_count = min(int(max_samples), len(dataset))
    indices = np.unique(np.linspace(0, len(dataset) - 1, num=sample_count, dtype=int))
    seen_classes = set()

    for index in indices:
        image_path = Path(dataset.get_image_path(int(index)))
        mask_path = Path(dataset.get_mask_path(int(index)))

        if not image_path.exists():
            raise FileNotFoundError(f"[{dataset_name}] Missing image: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"[{dataset_name}] Missing mask: {mask_path}")

        if image_path.resolve() == mask_path.resolve():
            raise ValueError(
                f"[{dataset_name}] Image and mask resolve to the same file: {image_path}"
            )

        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            raise FileNotFoundError(f"[{dataset_name}] Cannot read mask: {mask_path}")
        if mask_raw.ndim == 3:
            mask_raw = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2RGB)

        decoded = decode_ground_truth(mask_raw)
        unique_classes = set(np.unique(decoded).tolist())
        if not unique_classes.issubset({0, 1, 2}):
            raise ValueError(
                f"[{dataset_name}] Invalid decoded class IDs {sorted(unique_classes)} "
                f"in mask {mask_path}"
            )
        seen_classes.update(unique_classes)

    print(
        f"[Mask check] {dataset_name}: image_col='{dataset.image_col}', "
        f"mask_col='{dataset.mask_col}', checked={len(indices)}, "
        f"classes_seen={sorted(seen_classes)}"
    )
