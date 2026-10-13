import torch
import torch.nn as nn
import torch.nn.functional as F


from .gdl import GeneralizedDiceLoss


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
        valid = targets != self.cross_entropy.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        # A zero-weight-only batch has no CE contribution, rather than NaN.
        weights = self.cross_entropy.weight
        denominator = (valid.sum() if weights is None else weights[targets[valid]].sum())
        ce_loss = F.cross_entropy(
            logits.float(), targets, weight=weights,
            ignore_index=self.cross_entropy.ignore_index, reduction="sum",
        ) / denominator.clamp_min(1e-12)
        gdl_loss = self.gdl(logits, targets)
        return self.ce_loss_weight * ce_loss + self.gdl_loss_weight * gdl_loss
