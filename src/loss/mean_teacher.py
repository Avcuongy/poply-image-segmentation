"""Mean Teacher softmax-MSE, normalized over B, C, H and W.

Reference: CuriousAI/mean-teacher, pytorch/mean_teacher/losses.py.
Matches both existing segmentation training notebooks.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SoftmaxMSEConsistencyLoss(nn.Module):
    def forward(self, student_logits, teacher_logits):
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "Student and Teacher logits must have identical shapes, "
                f"got {student_logits.shape} and {teacher_logits.shape}."
            )
        return F.mse_loss(
            torch.softmax(student_logits, dim=1),
            torch.softmax(teacher_logits.detach(), dim=1),
            reduction="mean",
        )
