import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch
from torch.nn import functional as F


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def hd95_binary(pred_mask, true_mask, spacing=(1.0, 1.0)):
    pred, true = np.asarray(pred_mask, dtype=bool), np.asarray(true_mask, dtype=bool)
    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError("HD95 requires matching 2D masks")
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("spacing must contain two finite positive values")
    if not pred.any() and not true.any():
        return float("nan")
    if not pred.any() or not true.any():
        return float(np.linalg.norm(np.asarray(true.shape) * spacing))
    pred_surface = pred ^ binary_erosion(pred)
    true_surface = true ^ binary_erosion(true)
    to_true = distance_transform_edt(~true_surface, sampling=spacing)[pred_surface]
    to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[true_surface]
    return float(max(np.percentile(to_true, 95), np.percentile(to_pred, 95)))


def confusion_matrix(predictions, targets, num_classes=3, ignore_index=-100):
    pred = torch.as_tensor(predictions).detach().cpu()
    true = torch.as_tensor(targets).detach().cpu()
    if pred.shape != true.shape:
        raise ValueError("Prediction and target shapes must match")
    valid = true != ignore_index
    pred, true = pred[valid], true[valid]
    for values in (pred, true):
        if values.is_floating_point() and not torch.equal(
            values, values.long().to(values.dtype)
        ):
            raise ValueError("Masks must contain integer class IDs")
        if torch.any((values < 0) | (values >= num_classes)):
            raise ValueError("Class IDs outside [0, num_classes)")
    encoded = num_classes * true.long() + pred.long()
    return (
        torch.bincount(encoded, minlength=num_classes**2)
        .reshape(num_classes, num_classes)
        .numpy()
    )


def metrics_from_confusion(
    confusion, class_names=None, model_label="model", hd95_values=None
):
    matrix = np.asarray(confusion)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or (matrix < 0).any():
        raise ValueError("Expected a square, nonnegative confusion matrix")
    names = (
        class_names if class_names is not None else [str(i) for i in range(len(matrix))]
    )
    total = matrix.sum()
    rows = []
    for c in range(len(matrix)):
        tp = matrix[c, c]
        fp, fn = matrix[:, c].sum() - tp, matrix[c].sum() - tp
        tn = total - tp - fp - fn
        hd = np.asarray(hd95_values[c] if hd95_values is not None else [], dtype=float)
        hd = hd[~np.isnan(hd)]
        rows.append(
            dict(
                model=model_label,
                class_id=c,
                class_name=names[c],
                dice=safe_divide(2 * tp, 2 * tp + fp + fn),
                iou=safe_divide(tp, tp + fp + fn),
                precision=safe_divide(tp, tp + fp),
                recall=safe_divide(tp, tp + fn),
                specificity=safe_divide(tn, tn + fp),
                accuracy=safe_divide(tp + tn, total),
                pixel_accuracy=safe_divide(np.trace(matrix), total),
                hd95=float(hd.mean()) if hd.size else float("nan"),
                gt_pixels=int(matrix[c].sum()),
                pred_pixels=int(matrix[:, c].sum()),
            )
        )
    frame = pd.DataFrame(rows)
    metric_columns = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "pixel_accuracy",
        "hd95",
    ]
    for key, label, part in (
        ("macro_fg", "macro_foreground_only", frame.iloc[1:]),
        ("macro_all", "macro_all_classes", frame),
    ):
        rows.append(
            dict(
                model=model_label,
                class_id=key,
                class_name=label,
                **part[metric_columns].mean(skipna=True).to_dict(),
                gt_pixels=int(part.gt_pixels.sum()),
                pred_pixels=int(part.pred_pixels.sum()),
            )
        )
    return pd.DataFrame(rows)


def extract_logits(output, model_label="model"):
    if isinstance(output, dict):
        output = next(
            (output[k] for k in ("logits", "out", "prediction", "pred") if k in output),
            None,
        )
    elif isinstance(output, (tuple, list)):
        output = next(
            (item for item in output if torch.is_tensor(item) and item.ndim == 4), None
        )
    elif hasattr(output, "logits"):
        output = output.logits
    if not torch.is_tensor(output) or output.ndim != 4:
        raise ValueError(f"{model_label}: expected logits [B, C, H, W]")
    if not torch.isfinite(output).all():
        raise FloatingPointError(f"{model_label}: non-finite logits")
    return output


@torch.inference_mode()
def predict_batch(model, images, model_label="model", num_classes=3, output_size=None):
    logits = extract_logits(model(images), model_label)
    if logits.shape[1] != num_classes:
        raise ValueError("Output channel count differs from num_classes")
    if output_size is not None and logits.shape[-2:] != tuple(output_size):
        logits = F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )
    return logits.argmax(1)


@torch.inference_mode()
def evaluate_model(
    model,
    model_label,
    loader,
    device,
    num_classes=3,
    class_names=None,
    ignore_index=-100,
    compute_hd95=True,
):
    """Return (per-class/macro DataFrame, confusion matrix); restore training mode.

    Ignored pixels contribute to neither confusion counts nor HD95 masks.
    HD95 averages images; other ratios aggregate pixels over the entire loader.
    The model remains on the requested device.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    hd = [[] for _ in range(num_classes)]
    modes = {module: module.training for module in model.modules()}
    model.to(device).eval()
    batches = 0
    try:
        for batch in loader:
            images, targets = (
                (batch["image"], batch["mask"])
                if isinstance(batch, dict)
                else batch[:2]
            )
            targets = torch.as_tensor(targets).cpu()
            predictions = predict_batch(
                model, images.to(device), model_label, num_classes, targets.shape[-2:]
            ).cpu()
            matrix += confusion_matrix(predictions, targets, num_classes, ignore_index)
            if compute_hd95:
                for pred, true in zip(predictions.numpy(), targets.numpy()):
                    valid = true != ignore_index
                    for c in range(num_classes):
                        hd[c].append(
                            hd95_binary((pred == c) & valid, (true == c) & valid)
                        )
            batches += 1
    finally:
        for module, training in modes.items():
            module.training = training
    if batches == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return metrics_from_confusion(matrix, class_names, model_label, hd), matrix
