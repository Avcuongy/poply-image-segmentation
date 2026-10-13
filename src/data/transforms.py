import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42


def build_labeled_shared_geometry(height=256, width=256, seed=SEED):
    """Geometric-only view applied jointly to image and mask."""
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                scale=(0.85, 1.15),
                rotate=(-45, 45),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=120,
                        sigma=120 * 0.05,
                        p=0.5,
                    ),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(
                        distort_limit=(-1.0, 1.0),
                        p=0.5,
                    ),
                ],
                p=0.4,
            ),
        ],
        seed=int(seed),
        strict=True,
    )


def build_labeled_weak_transform(seed=SEED):
    """Teacher view of labeled images: normalize only, no photometric noise."""
    return A.Compose(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        seed=int(seed),
        strict=True,
    )


def build_labeled_strong_transform(seed=SEED):
    """Student view: photometric-only perturbations after shared geometry."""
    return A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.10, 0.10),
                        contrast_limit=(-0.10, 0.10),
                        p=0.5,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=(-10, 10),
                        sat_shift_limit=(-15, 15),
                        val_shift_limit=(-10, 10),
                        p=0.5,
                    ),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                ],
                p=0.2,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        seed=int(seed),
        strict=True,
    )


def build_unlabeled_shared_geometry(height=224, width=224, seed=SEED):
    """One geometric view shared by Teacher and Student to preserve alignment."""
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                scale=(0.90, 1.10),
                rotate=(-20, 20),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
        ],
        seed=int(seed),
        strict=True,
    )


def build_unlabeled_weak_transform(seed=SEED):
    return A.Compose(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        seed=int(seed),
        strict=True,
    )


def build_unlabeled_strong_transform(seed=SEED):
    """Photometric-only perturbations keep Teacher/Student pixels aligned."""
    return A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.20, 0.20),
                        contrast_limit=(-0.20, 0.20),
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=(-15, 15),
                        sat_shift_limit=(-25, 25),
                        val_shift_limit=(-15, 15),
                        p=1.0,
                    ),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.3,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        seed=int(seed),
        strict=True,
    )


def build_eval_transform(
    height=224,
    width=224,
    seed=SEED,
    normalize=True,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(
                mean=mean if normalize else (0, 0, 0),
                std=std if normalize else (1, 1, 1),
            ),
            ToTensorV2(),
        ],
        seed=int(seed),
        strict=True,
    )


def build_supervised_transform(height=224, width=224, seed=SEED):
    return A.Compose(
        [
            build_labeled_shared_geometry(height, width, seed),
            build_labeled_strong_transform(seed),
        ],
        seed=int(seed),
    )
