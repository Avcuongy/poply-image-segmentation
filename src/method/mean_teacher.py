import copy
import math

import torch
from torch import nn


def create_teacher(student):
    memo = {}
    for name in ("projection_head", "contrastive_memory"):
        value = getattr(student, name, None)
        if value is not None:
            memo[id(value)] = None
    teacher = copy.deepcopy(student, memo)
    for name in ("projection_head", "contrastive_memory"):
        if hasattr(teacher, name):
            delattr(teacher, name)
    teacher.requires_grad_(False)
    teacher.train()
    return teacher


@torch.no_grad()
def update_ema_teacher(student, teacher, ema_decay, global_step):
    """Call after optimizer.step(), with the incremented global step."""
    if not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be in [0, 1).")
    if global_step < 0:
        raise ValueError("global_step must be nonnegative.")
    effective_decay = min(float(ema_decay), 1.0 - 1.0 / (float(global_step) + 1.0))
    student_parameters = dict(student.named_parameters())
    teacher_parameters = dict(teacher.named_parameters())
    segmentation_names = {
        name for name in student_parameters if not name.startswith("projection_head.")
    }
    if segmentation_names != teacher_parameters.keys():
        raise RuntimeError("Student and Teacher parameter structures do not match.")
    for name, teacher_parameter in teacher_parameters.items():
        teacher_parameter.mul_(effective_decay).add_(
            student_parameters[name], alpha=1.0 - effective_decay
        )


def sigmoid_rampup(current, rampup_length):
    rampup_length = float(rampup_length)
    if rampup_length < 0:
        raise ValueError("rampup_length must be nonnegative.")
    if rampup_length == 0:
        return 1.0
    current = min(max(float(current), 0.0), rampup_length)
    return math.exp(-5.0 * (1.0 - current / rampup_length) ** 2)


def get_consistency_weight(epoch, mt_config):
    return float(mt_config["max_consistency_weight"]) * sigmoid_rampup(
        max(epoch - 1, 0), int(mt_config["consistency_rampup_epochs"])
    )


class MeanTeacher(nn.Module):
    def __init__(self, student, ema_decay=0.999):
        super().__init__()
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1).")
        self.student = student
        self.teacher = create_teacher(student)
        self.ema_decay = float(ema_decay)

    def forward(self, strong_images, weak_images):
        student_logits = self.student(strong_images)
        with torch.no_grad():
            teacher_logits = self.teacher(weak_images)
        return student_logits, teacher_logits

    def update_teacher(self, global_step):
        update_ema_teacher(self.student, self.teacher, self.ema_decay, global_step)
