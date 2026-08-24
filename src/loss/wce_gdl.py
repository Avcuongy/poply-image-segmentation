import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss (GDL): Automatically balances class weights based on the inverse square of the true area."""

    def __init__(self, num_classes=3, smooth=1e-6, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probabilities = torch.softmax(logits, dim=1)

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            safe_targets = targets

        targets_one_hot = F.one_hot(safe_targets.long(), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1).float()
        probabilities = probabilities * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        dims = (0, 2, 3)
        targets_sum = torch.sum(targets_one_hot, dim=dims)

        present = targets_sum > 0
        safe_targets_sum = torch.where(
            present,
            targets_sum,
            torch.ones_like(targets_sum),
        )
        weights = present.to(probabilities.dtype) / (
            safe_targets_sum + self.smooth
        ).pow(2)
        weights = weights / weights.sum().clamp_min(self.smooth)

        intersection = torch.sum(probabilities * targets_one_hot, dim=dims)
        pred_sum = torch.sum(probabilities, dim=dims)

        weighted_intersection = torch.sum(weights * intersection)
        weighted_denominator = torch.sum(weights * (pred_sum + targets_sum))

        gdl = (2.0 * weighted_intersection + self.smooth) / (
            weighted_denominator + self.smooth
        )
        return 1.0 - gdl


class WeightedCE_GDL_Loss(nn.Module):
    def __init__(
        self,
        class_weights=None,
        num_classes=3,
        ce_loss_weight=1.0,
        gdl_loss_weight=1.0,
        smooth=1e-6,
        ignore_index=-100,
    ):
        super().__init__()
        self.ce_loss_weight = float(ce_loss_weight)
        self.gdl_loss_weight = float(gdl_loss_weight)

        ce_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )

        self.register_buffer("ce_weights", ce_weights, persistent=True)

        self.cross_entropy = nn.CrossEntropyLoss(
            weight=self.ce_weights, ignore_index=ignore_index
        )
        self.gdl = GeneralizedDiceLoss(
            num_classes=num_classes, smooth=smooth, ignore_index=ignore_index
        )

    def get_extra_state(self):
        return {
            "ce_loss_weight": self.ce_loss_weight,
            "gdl_loss_weight": self.gdl_loss_weight,
            "num_classes": self.gdl.num_classes,
            "smooth": self.gdl.smooth,
            "ignore_index": self.gdl.ignore_index,
        }

    def set_extra_state(self, state):
        if not state:
            return

        self.ce_loss_weight = float(state["ce_loss_weight"])
        self.gdl_loss_weight = float(state["gdl_loss_weight"])
        self.gdl.num_classes = int(state["num_classes"])
        self.gdl.smooth = float(state["smooth"])
        self.gdl.ignore_index = state["ignore_index"]
        self.cross_entropy.ignore_index = state["ignore_index"]

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy(logits, targets)
        gdl_loss = self.gdl(logits, targets)
        return self.ce_loss_weight * ce_loss + self.gdl_loss_weight * gdl_loss
