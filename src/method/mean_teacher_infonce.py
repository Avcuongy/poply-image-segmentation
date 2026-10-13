from loss.contrastive import PixelRegionMemory, validate_contrastive_config
from .mean_teacher import MeanTeacher


class MeanTeacherInfoNCE(MeanTeacher):
    def __init__(self, student, image_keys, ema_decay=0.999):
        if not hasattr(student, "project_pixels"):
            raise TypeError("Use ContrastiveUNet or ContrastiveSwinUNet as the student")
        cfg = student.contrastive_config
        validate_contrastive_config(cfg)
        super().__init__(student, ema_decay)
        num_classes = getattr(
            student, "n_classes", getattr(student, "num_classes", None)
        )
        student.contrastive_memory = PixelRegionMemory(
            num_classes,
            image_keys,
            cfg["embedding_dim"],
            cfg,
            next(student.parameters()).device,
        )

    def get_extra_state(self):
        return self.student.contrastive_memory.state_dict()

    def set_extra_state(self, state):
        self.student.contrastive_memory.load_state_dict(state)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.student.contrastive_memory.to(next(self.student.parameters()).device)
        return result
