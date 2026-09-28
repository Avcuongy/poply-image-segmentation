"""Load baseline and Mean Teacher segmentation checkpoints for evaluation."""

from pathlib import Path

import torch
from torch import nn


MODEL_TYPES = ("unet", "swin", "mean_teacher_unet", "mean_teacher_swin")
SWIN_DEFAULTS = {
    "img_size": 256, "patch_size": 4, "in_channels": 3,
    "embed_dim": 96, "depths": (2, 2, 6, 2),
    "num_heads": (3, 6, 12, 24), "window_size": 8,
    "mlp_ratio": 4.0, "qkv_bias": True, "drop_rate": 0.0,
    "attn_drop_rate": 0.0, "drop_path_rate": 0.2,
    "ape": False, "patch_norm": True, "use_checkpoint": False,
    "pretrained_window_sizes": (0, 0, 0, 0),
}


def checkpoint_entries(groups):
    """Accept {type: path} or {type: {run_name: path}}; skip empty entries."""
    for model_type, runs in groups.items():
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model type {model_type!r}; use {MODEL_TYPES}.")
        if not runs:
            continue
        if isinstance(runs, (str, Path)):
            yield model_type, model_type, Path(runs)
        elif isinstance(runs, dict):
            for name, path in runs.items():
                if path:
                    yield f"{model_type}/{name}", model_type, Path(path)
        else:
            raise TypeError(f"{model_type}: expected a path or a dictionary of paths.")


def load_checkpoint_file(checkpoint_path, map_location="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary or raw state_dict.")
    return checkpoint


def is_raw_state_dict(value):
    return (isinstance(value, dict) and bool(value)
            and all(torch.is_tensor(item) for item in value.values()))


def clean_state_dict(state):
    for _ in range(8):
        for prefix in ("module.", "model.", "net.", "network.", "_orig_mod."):
            if state and all(key.startswith(prefix) for key in state):
                state = {key[len(prefix):]: value for key, value in state.items()}
                break
        else:
            break
    return state


def select_model_weights(checkpoint, mean_teacher=False):
    """Prefer best Teacher, never silently fall back to Student."""
    for key in ("best_teacher_state_dict", "teacher_state_dict"):
        if is_raw_state_dict(checkpoint.get(key)):
            return clean_state_dict(checkpoint[key]), key, True

    containers = [("raw_state_dict", checkpoint)]
    for key in ("model_state_dict", "state_dict", "model", "net", "network", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, nn.Module):
            value = value.state_dict()
        containers.append((key, value))
        if isinstance(value, dict) and not is_raw_state_dict(value):
            for nested_key in ("state_dict", "model_state_dict", "weights"):
                containers.append((f"{key}.{nested_key}", value.get(nested_key)))

    has_mt_metadata = (
        "mean_teacher" in str(checkpoint.get("training_method", ""))
        or any(key in checkpoint for key in (
            "student_state_dict", "teacher_state_dict", "best_teacher_state_dict"
        ))
    )
    for key, state in containers:
        if not is_raw_state_dict(state):
            continue
        state = clean_state_dict(state)
        teacher_state = {
            name[len("teacher."):]: tensor
            for name, tensor in state.items() if name.startswith("teacher.")
        }
        if teacher_state:
            return clean_state_dict(teacher_state), f"{key}:teacher", True
        if has_mt_metadata or any(name.startswith("student.") for name in state):
            continue
        # A separately exported Teacher is indistinguishable from a baseline
        # state_dict; the explicit dictionary group supplies its provenance.
        return state, key, bool(mean_teacher)
    raise KeyError("No usable weights found. Mean Teacher requires Teacher weights.")


def detect_architecture(state):
    if "outc.conv.weight" in state and "encoder.conv1.weight" in state:
        return "unet"
    if "segmentation_head.weight" in state and "encoder.patch_embed.proj.weight" in state:
        if not any(key.endswith(".attn.logit_scale") for key in state):
            raise ValueError("Expected Swin V2 weights; Swin V1 is not supported.")
        return "swin"
    raise ValueError("Weights do not match the project's UNet or SwinUNet architecture.")


def resolve_model_config(checkpoint, state, architecture, eval_config):
    """Checkpoint metadata controls architecture and input preprocessing only."""
    saved = checkpoint.get("config", {})
    if not isinstance(saved, dict):
        saved = {}
    config = dict(eval_config)
    for key in ("image_height", "image_width", "num_classes", "normalize_input",
                "input_mean", "input_std"):
        if key in saved:
            config[key] = saved[key]

    head = "outc.conv.weight" if architecture == "unet" else "segmentation_head.weight"
    num_classes = state[head].shape[0]
    if "num_classes" in saved and int(saved["num_classes"]) != num_classes:
        raise ValueError("Checkpoint num_classes does not match its output weights.")
    config["num_classes"] = num_classes
    if architecture == "unet":
        inferred_backbone = (
            "resnet50" if "encoder.layer1.0.conv3.weight" in state else "resnet34"
        )
        config["backbone"] = saved.get("backbone", inferred_backbone)
        config["bilinear"] = saved.get("bilinear", "up4.up.1.weight" in state)
    else:
        for key, default in SWIN_DEFAULTS.items():
            config[key] = saved.get(key, default)
        config["img_size"] = int(saved.get("img_size", saved.get(
            "image_height", config.get("image_height", 256)
        )))
        config["image_height"] = int(saved.get("image_height", config["img_size"]))
        config["image_width"] = int(saved.get("image_width", config["img_size"]))
        if config["image_height"] != config["img_size"] or config["image_width"] != config["img_size"]:
            raise ValueError("SwinUNet requires square inputs matching img_size.")
        config["backbone"] = saved.get("backbone", "swinv2_tiny_patch4_window8_256")
    return config


def build_and_load_model(checkpoint_path, model_type, eval_config):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    checkpoint = load_checkpoint_file(checkpoint_path)
    state, weight_key, is_mt = select_model_weights(
        checkpoint, mean_teacher=model_type.startswith("mean_teacher_")
    )
    architecture = detect_architecture(state)
    expected = model_type.removeprefix("mean_teacher_")
    if architecture != expected:
        raise ValueError(
            f"{checkpoint_path}: detected {architecture}, but placed in {model_type}."
        )
    config = resolve_model_config(checkpoint, state, architecture, eval_config)
    if architecture == "unet":
        from model.unet import UNet

        model = UNet(
            backbone=config["backbone"], n_classes=config["num_classes"],
            bilinear=config["bilinear"], pretrained=False,
        )
    else:
        from model.swin_unet import SwinUNet

        kwargs = {key: config[key] for key in SWIN_DEFAULTS}
        kwargs["use_checkpoint"] = False
        model = SwinUNet(
            **kwargs, num_classes=config["num_classes"],
            pretrained_path=None, pretrained_decoder=False,
        )
    # Partial loading can silently produce invalid evaluation results.
    model.load_state_dict(state, strict=True)
    model.eval()
    detected_type = f"mean_teacher_{architecture}" if is_mt else architecture
    info = {
        "model_type": detected_type, "weights": weight_key,
        "best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_path": str(checkpoint_path),
    }
    print(f"[{detected_type}] {checkpoint_path} | weights={weight_key} | strict load OK")
    return model, config, info
