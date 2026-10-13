import numpy as np
import torch
from torch.nn import functional as F

DEFAULT_CONTRASTIVE_CONFIG = {
    "objective": "contrastiveseg_paper_memory",
    "seed": 1042,
    # Model
    "feature_key": "decoder",  # U-Net adaptation; full encoder/decoder gradient path.
    "feature_stride": 4,  # Pool final decoder before projection to bound memory.
    "embedding_dim": 256,
    "ignore_index": -100,
    "anchors_per_class": 50,
    "pixel_samples_per_image": 10,
    "pixel_queue_images": 10,  # Queue capacity T = 10 * N per class.
    "positive_samples": 1024,
    "negative_samples": 2048,
    "semi_hard_fraction": 0.1,
    # Hyperparameters
    "temperature": 1.0,  # Tune
    "max_weight": 1.5,  # Tune
    # Warmup
    "warmup_epochs": 8,
    "anchor_chunk_size": 16,
}


CONTRASTIVE_METRIC_NAMES = (
    "contrastive_loss",
    "weighted_contrastive_loss",
    "contrastive_weight",
    "contrastive_to_supervised_ratio",
    "contrastive_active_fraction",
    "contrastive_groups",
    "contrastive_pixels",
    "contrastive_anchors",
    "hard_pixels",
    "easy_pixels",
    "pixels_background",
    "pixels_non_neoplastic",
    "pixels_neoplastic",
    "memory_pixels",
    "memory_regions",
    "positive_samples",
    "negative_samples",
)


def validate_contrastive_config(cfg):
    if cfg["objective"] != "contrastiveseg_paper_memory":
        raise ValueError(
            "Use a fresh run or a paper-memory checkpoint, not no-memory weights."
        )
    for key in (
        "embedding_dim",
        "feature_stride",
        "anchors_per_class",
        "pixel_samples_per_image",
        "pixel_queue_images",
        "positive_samples",
        "negative_samples",
        "anchor_chunk_size",
    ):
        if int(cfg[key]) != cfg[key] or int(cfg[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not np.isfinite(cfg["temperature"]) or cfg["temperature"] <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(cfg["max_weight"]) or cfg["max_weight"] < 0:
        raise ValueError("max_weight must be finite and nonnegative")
    if not 0 < cfg["semi_hard_fraction"] <= 1:
        raise ValueError("semi_hard_fraction must be in (0, 1]")
    if int(cfg["warmup_epochs"]) != cfg["warmup_epochs"] or cfg["warmup_epochs"] < 0:
        raise ValueError("warmup_epochs must be a nonnegative integer")
    if cfg["feature_key"] != "decoder":
        raise ValueError("This U-Net adaptation projects final decoder features")


def contrastive_weight_for_epoch(epoch, cfg):
    return 0.0 if int(epoch) <= int(cfg["warmup_epochs"]) else float(cfg["max_weight"])


def _resize_targets_for_feature_map(labels, size):
    return (
        F.interpolate(labels.unsqueeze(1).float(), size=size, mode="nearest")
        .squeeze(1)
        .long()
    )


def _resize_predictions_for_feature_map(logits, size):
    return F.interpolate(
        logits.detach().float(), size=size, mode="bilinear", align_corners=False
    ).argmax(dim=1)


def _contrastive_generator(device, seed, step, stream=0):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 1000003 * int(step) + 10007 * int(stream))
    return generator


def _random_take(indices, count, generator):
    count = min(int(count), indices.numel())
    if count == indices.numel():
        return indices
    return indices[
        torch.randperm(indices.numel(), device=indices.device, generator=generator)[
            :count
        ]
    ]


class PixelRegionMemory:
    """Paper Sec. 3.2: C x (10N + N) x D detached Student embeddings.

    Plain training state (not an nn.Module): excluded from model/EMA weights.
    Pixel queues are FIFO per class; region slots are indexed by class AND
    stable labeled-dataset image index. Uninitialized slots never enter loss.
    Only labeled training data may call update; evaluation never writes here.
    """

    def __init__(self, num_classes, image_keys, dim, cfg, device):
        self.image_keys = tuple(image_keys)
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError(
                "Memory requires nonempty, unique labeled-image identities"
            )
        self.num_classes, self.dim = int(num_classes), int(dim)
        self.num_images = len(self.image_keys)
        self.capacity = int(cfg["pixel_queue_images"]) * self.num_images
        self.pixel_samples = int(cfg["pixel_samples_per_image"])
        self.ignore_index = int(cfg["ignore_index"])
        self.seed = int(cfg["seed"])
        self.pixel_bank = torch.zeros(
            self.num_classes, self.capacity, self.dim, device=device
        )
        self.region_bank = torch.zeros(
            self.num_classes, self.num_images, self.dim, device=device
        )
        self.pixel_ptr = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        self.pixel_count = torch.zeros_like(self.pixel_ptr)
        self.region_valid = torch.zeros(
            self.num_classes, self.num_images, dtype=torch.bool, device=device
        )

    def references(self):
        features, labels = [], []
        for cls in range(self.num_classes):
            pixels = self.pixel_bank[cls, : int(self.pixel_count[cls])]
            regions = self.region_bank[cls, self.region_valid[cls]]
            keys = torch.cat((pixels, regions), dim=0)
            features.append(keys)
            labels.append(
                torch.full((len(keys),), cls, dtype=torch.long, device=keys.device)
            )
        return torch.cat(features).detach(), torch.cat(labels)

    def to(self, device):
        """Move training memory without changing its FP32 storage precision."""
        for name in (
            "pixel_bank",
            "region_bank",
            "pixel_ptr",
            "pixel_count",
            "region_valid",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @torch.no_grad()
    def update(self, embeddings, labels, image_indices, step):
        if embeddings.shape[0] != len(image_indices) or embeddings.shape[1] != self.dim:
            raise ValueError("Memory input shape mismatch")
        ids = torch.as_tensor(image_indices).detach().cpu().tolist()
        if len(set(ids)) != len(ids) or any(i < 0 or i >= self.num_images for i in ids):
            raise ValueError("Invalid or duplicate labeled image indices")
        small_labels = _resize_targets_for_feature_map(labels, embeddings.shape[-2:])
        if torch.any(
            (small_labels != self.ignore_index)
            & ((small_labels < 0) | (small_labels >= self.num_classes))
        ):
            raise ValueError("Invalid semantic label for contrastive memory")
        vectors = F.normalize(embeddings.detach().float(), dim=1).permute(0, 2, 3, 1)
        if not torch.isfinite(vectors).all():
            raise FloatingPointError("Non-finite embeddings cannot enter memory")
        generator = _contrastive_generator(embeddings.device, self.seed, step, stream=2)
        for batch_index, image_index in enumerate(ids):
            for cls in range(self.num_classes):
                class_vectors = vectors[batch_index][small_labels[batch_index] == cls]
                if len(class_vectors) == 0:
                    # A crop may omit a class: preserve the last observed region.
                    continue
                region = class_vectors.mean(dim=0)
                if region.norm() > 1e-12:
                    self.region_bank[cls, image_index] = F.normalize(region, dim=0)
                    self.region_valid[cls, image_index] = True
                indices = torch.arange(len(class_vectors), device=embeddings.device)
                selected = _random_take(
                    indices, min(self.pixel_samples, self.capacity), generator
                )
                keys = class_vectors[selected]
                pointer = int(self.pixel_ptr[cls])
                slots = (
                    torch.arange(len(keys), device=embeddings.device) + pointer
                ) % self.capacity
                self.pixel_bank[cls, slots] = keys
                self.pixel_ptr[cls] = (pointer + len(keys)) % self.capacity
                self.pixel_count[cls] = min(
                    self.capacity, int(self.pixel_count[cls]) + len(keys)
                )

    def state_dict(self):
        return {
            "version": 1,
            "image_keys": self.image_keys,
            "pixel_samples": self.pixel_samples,
            "seed": self.seed,
            "ignore_index": self.ignore_index,
            **{
                name: getattr(self, name).detach().cpu().clone()
                for name in (
                    "pixel_bank",
                    "region_bank",
                    "pixel_ptr",
                    "pixel_count",
                    "region_valid",
                )
            },
        }

    def load_state_dict(self, state):
        if (
            state.get("version") != 1
            or tuple(state.get("image_keys", ())) != self.image_keys
            or state.get("pixel_samples") != self.pixel_samples
            or state.get("seed") != self.seed
            or state.get("ignore_index") != self.ignore_index
        ):
            raise ValueError(
                "Memory checkpoint does not match the labeled split/order or configuration"
            )
        names = (
            "pixel_bank",
            "region_bank",
            "pixel_ptr",
            "pixel_count",
            "region_valid",
        )
        for name in names:
            target, saved = getattr(self, name), state[name]
            if target.shape != saved.shape or target.dtype != saved.dtype:
                raise ValueError(f"Memory checkpoint shape/dtype mismatch: {name}")
            if not torch.isfinite(saved).all():
                raise ValueError(f"Non-finite memory checkpoint: {name}")
        if (
            torch.any(state["pixel_count"] < 0)
            or torch.any(state["pixel_count"] > self.capacity)
            or torch.any(state["pixel_ptr"] < 0)
            or torch.any(state["pixel_ptr"] >= self.capacity)
        ):
            raise ValueError("Invalid memory queue counters")
        for name in names:
            getattr(self, name).copy_(state[name])


def sample_contrastiveseg_pixels(embeddings, ground_truth, student_logits, cfg, step):
    """Paper: up to 50 anchors per CLASS across the batch, half hard + half random.

    Random anchors come from the remaining class pixels (including hard ones).
    Never duplicate an anchor; scarce classes do not reduce other class budgets.
    Single-pixel classes are valid because positives can come from memory.
    """
    size = embeddings.shape[-2:]
    labels = _resize_targets_for_feature_map(ground_truth, size).reshape(-1)
    predictions = _resize_predictions_for_feature_map(student_logits, size).reshape(-1)
    features = embeddings.permute(0, 2, 3, 1).reshape(-1, embeddings.shape[1])
    generator = _contrastive_generator(embeddings.device, cfg["seed"], step)
    selections = []
    num_classes = student_logits.shape[1]
    for cls in range(num_classes):
        candidates = torch.nonzero(labels == cls, as_tuple=False).flatten()
        quota = min(int(cfg["anchors_per_class"]), len(candidates))
        if quota == 0:
            continue
        hard = candidates[predictions[candidates] != cls]
        chosen_hard = _random_take(hard, quota // 2, generator)
        remaining = candidates[~torch.isin(candidates, chosen_hard)]
        chosen_random = _random_take(remaining, quota - len(chosen_hard), generator)
        selections.append(torch.cat((chosen_hard, chosen_random)))
    if not selections:
        return None
    indices = torch.cat(selections)
    anchor_labels = labels[indices]
    hard_count = int((predictions[indices] != anchor_labels).sum())
    image_ids = indices // (size[0] * size[1])
    return {
        "features": features[indices],
        "labels": anchor_labels,
        "indices": indices,
        "pixels": len(indices),
        "groups": len(torch.unique(image_ids * num_classes + anchor_labels)),
        "hard_pixels": hard_count,
        "easy_pixels": len(indices) - hard_count,
        "per_class_counts": torch.bincount(anchor_labels, minlength=num_classes),
    }


@torch.no_grad()
def semi_hard_indices(similarities, count, fraction, largest, generator):
    """Random subset of top 10% nearest negatives / farthest positives.

    Cap K at the available candidate count for a small/partially filled bank;
    never pad with zeros or repeat keys to manufacture the paper's K.
    """
    pool_size = min(
        similarities.shape[1], max(1, int(np.ceil(fraction * similarities.shape[1])))
    )
    pool = similarities.topk(pool_size, dim=1, largest=largest).indices
    count = min(int(count), pool_size)
    order = torch.rand(pool.shape, device=pool.device, generator=generator).argsort(
        dim=1
    )[:, :count]
    return pool.gather(1, order)


def contrastiveseg_memory_loss(sampled, memory, cfg, step):
    """Eq. (3), averaged over valid anchors, without repo-only tau/base_tau scaling.

    Each positive is contrasted with the SUM of selected negatives. Other
    positives are not added to its denominator. Pixel and region keys share
    the same candidate pool, labels and semi-hard mining rule.
    """
    anchors = F.normalize(sampled["features"].float(), dim=1)
    keys, key_labels = memory.references()
    generator = _contrastive_generator(anchors.device, cfg["seed"], step, stream=1)
    losses, positive_total, negative_total = [], 0, 0
    for cls in torch.unique(sampled["labels"]).tolist():
        positives = keys[key_labels == cls]
        negatives = keys[key_labels != cls]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        queries = anchors[sampled["labels"] == cls]
        for query in queries.split(int(cfg["anchor_chunk_size"])):
            positive_indices = semi_hard_indices(
                query.detach() @ positives.T,
                cfg["positive_samples"],
                cfg["semi_hard_fraction"],
                False,
                generator,
            )
            negative_indices = semi_hard_indices(
                query.detach() @ negatives.T,
                cfg["negative_samples"],
                cfg["semi_hard_fraction"],
                True,
                generator,
            )
            pos = (
                torch.einsum("ad,akd->ak", query, positives[positive_indices])
                / cfg["temperature"]
            )
            neg = (
                torch.einsum("ad,akd->ak", query, negatives[negative_indices])
                / cfg["temperature"]
            )
            log_neg_sum = torch.logsumexp(neg, dim=1, keepdim=True)
            losses.append((torch.logaddexp(pos, log_neg_sum) - pos).mean(dim=1))
            positive_total += pos.numel()
            negative_total += neg.numel()
    if not losses:
        return anchors.sum() * 0.0, {
            "contrastive_anchors": 0.0,
            "positive_samples": 0.0,
            "negative_samples": 0.0,
        }
    per_anchor = torch.cat(losses)
    return per_anchor.mean(), {
        "contrastive_anchors": float(len(per_anchor)),
        "positive_samples": positive_total / len(per_anchor),
        "negative_samples": negative_total / len(per_anchor),
    }


def compute_student_contrastive(
    student,
    student_feature_map,
    labeled_logits,
    labeled_masks,
    image_indices,
    epoch,
    step,
):
    cfg = student.contrastive_config
    memory = student.contrastive_memory
    weight = contrastive_weight_for_epoch(epoch, cfg)
    metrics = {name: 0.0 for name in CONTRASTIVE_METRIC_NAMES}
    metrics["contrastive_weight"] = weight
    zero = student_feature_map.sum() * 0.0
    if float(cfg["max_weight"]) == 0:
        return zero, metrics

    with torch.set_grad_enabled(torch.is_grad_enabled() and weight > 0):
        embeddings = student.project_pixels(student_feature_map)
    loss = zero
    if weight > 0:
        sampled = sample_contrastiveseg_pixels(
            embeddings, labeled_masks, labeled_logits, cfg, step
        )
        if sampled is not None:
            loss, mining_metrics = contrastiveseg_memory_loss(
                sampled, memory, cfg, step
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite ContrastiveSeg loss")
            metrics.update(mining_metrics)
            metrics.update(
                contrastive_loss=float(loss.detach()),
                weighted_contrastive_loss=weight * float(loss.detach()),
                contrastive_active_fraction=float(
                    mining_metrics["contrastive_anchors"] > 0
                ),
                contrastive_groups=float(sampled["groups"]),
                contrastive_pixels=float(sampled["pixels"]),
                hard_pixels=float(sampled["hard_pixels"]),
                easy_pixels=float(sampled["easy_pixels"]),
            )
            for cls, name in enumerate(
                ("pixels_background", "pixels_non_neoplastic", "pixels_neoplastic")
            ):
                counts = sampled["per_class_counts"]
                metrics[name] = float(counts[cls]) if cls < len(counts) else 0.0
    # Read BEFORE write: current anchors cannot contrast against their own newly
    # inserted keys. The loss owns a detached snapshot, so this update is safe.
    memory.update(embeddings, labeled_masks, image_indices, step)
    metrics["memory_pixels"] = float(memory.pixel_count.sum())
    metrics["memory_regions"] = float(memory.region_valid.sum())
    return weight * loss, metrics
