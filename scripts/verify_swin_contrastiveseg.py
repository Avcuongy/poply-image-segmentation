"""Check Swin notebook integration and shared ContrastiveSeg behavior on CPU.

Run from the repository: .venv/Scripts/python.exe scripts/verify_swin_contrastiveseg.py
No downloads or full training datasets are needed.
"""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/swin_unet/swin_mean_teacher_infonce.ipynb"
NB = json.loads(PATH.read_text(encoding="utf-8"))
UNET = json.loads((ROOT / "notebooks/unet/unet_mean_teacher_infonce.ipynb").read_text(encoding="utf-8"))
NS = dict(globals(), SEED=42,
          CLASS_NAMES={0: "background", 1: "non-neoplastic", 2: "neoplastic"},
          CLASS_COLORS={0: (0, 0, 0), 1: (0, 255, 0), 2: (255, 0, 0)})
for index in (5, 7, 17, 18, 21, 23, 25, 27, 29, 30, 33, 35):
    exec(compile("".join(NB["cells"][index]["source"]), f"swin-cell-{index}", "exec"), NS)

# Reuse the numerical, sampling, queue and resume checks against the Swin
# notebook's actual definitions, rather than maintaining two copies of tests.
spec = importlib.util.spec_from_file_location("unet_contrastive_checks", ROOT / "tests/verify_contrastiveseg_notebook.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.NS, shared.NB = NS, NB


def make_student(checkpointing=True):
    cfg = copy.deepcopy(NS["DEFAULT_CONTRASTIVE_CONFIG"])
    cfg.update(embedding_dim=8, anchors_per_class=6, warmup_epochs=0)
    return NS["ContrastiveSwinUNet"](
        cfg, img_size=32, embed_dim=12, depths=(1, 1, 1, 1),
        num_heads=(3, 3, 6, 12), window_size=2,
        drop_path_rate=0, use_checkpoint=checkpointing, pretrained_path=None,
    )


def attach_memory(student):
    cfg = student.contrastive_config
    student.contrastive_memory = NS["PixelRegionMemory"](
        3, ["image-a", "image-b"], cfg["embedding_dim"], cfg, "cpu")


def data():
    x = torch.randn(2, 3, 32, 32)
    y = torch.zeros(2, 32, 32, dtype=torch.long)
    y[:, :, 12:24] = 1
    y[:, :, 24:] = 2
    return x, y


class SwinContrastTests(shared.PaperContrastTests):
    # Override the architecture-specific U-Net check; all shared checks still run.
    def test_actual_unet_gradient_teacher_and_inference(self):
        student = make_student()
        teacher = NS["create_teacher"](student)
        attach_memory(student)
        x, y = data()
        counts = []
        hook = student.encoder.register_forward_hook(lambda *args: counts.append(1))
        logits, features = student(x, return_features=True)
        self.assertEqual(tuple(features["decoder"].shape), (2, 12, 32, 32))
        self.assertEqual(tuple(features["decoder_tokens"].shape), (2, 12, 8, 8))
        projected = student.project_pixels(features["decoder"])
        self.assertEqual(tuple(projected.shape), (2, 8, 8, 8))
        torch.testing.assert_close(projected.norm(dim=1), torch.ones(2, 8, 8))
        student.contrastive_memory.update(projected, y, [0, 1], 0)
        loss, metrics = NS["compute_student_contrastive"](
            student, features["decoder"], logits, y, [0, 1], 1, 1)
        self.assertEqual(metrics["contrastive_active_fraction"], 1)
        self.assertEqual(len(counts), 1)
        hook.remove()
        loss.backward()
        for name, parameter in (
            ("patch embedding", student.encoder.patch_embed.proj.weight),
            ("deep encoder", student.encoder.norm.weight),
            ("decoder", student.layers_up[0].expand.weight),
            ("final expansion", student.final_up.expand.weight),
            ("projection", student.projection_head.proj[-1].weight),
        ):
            with self.subTest(layer=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.norm().item(), 0)
        NS["update_ema_teacher"](student, teacher, .999, 1)
        self.assertIsInstance(teacher, NS["SwinUNet"])
        self.assertFalse(isinstance(teacher, NS["ContrastiveSwinUNet"]))
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        self.assertFalse(any("projection_head" in key for key in teacher.state_dict()))
        self.assertFalse(any("memory" in key for key in student.state_dict()))
        student.eval()
        with torch.no_grad():
            self.assertEqual(tuple(student(x).shape), (2, 3, 32, 32))

    def test_projection_stays_float32_inside_autocast(self):
        student = make_student(False)
        features = torch.randn(2, 12, 32, 32, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = student.project_pixels(features)
        self.assertEqual(output.dtype, torch.float32)
        output.square().mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_shared_core_matches_unet(self):
        def definitions(book):
            tree = ast.parse("".join(book["cells"][25]["source"]))
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        a, b = definitions(NB), definitions(UNET)
        for name in a:
            if name != "validate_contrastive_config":
                with self.subTest(name=name):
                    self.assertEqual(a[name], b[name])
        with self.assertRaises(ValueError):
            NS["validate_contrastive_config"](dict(NS["DEFAULT_CONTRASTIVE_CONFIG"], feature_stride=0))

    def test_labeled_dataset_has_stable_memory_index(self):
        tree = ast.parse("".join(NB["cells"][14]["source"]))
        labeled = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MeanTeacherLabeledDataset")
        getitem = next(n for n in labeled.body if isinstance(n, ast.FunctionDef) and n.name == "__getitem__")
        returns = [n for n in ast.walk(getitem) if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)]
        fields = dict(zip([k.value for k in returns[-1].value.keys], returns[-1].value.values))
        self.assertEqual(ast.unparse(fields["image_index"]), "int(index)")

    def test_real_swin_training_and_resume(self):
        student = make_student()
        teacher = NS["create_teacher"](student)
        attach_memory(student)
        x, y = data()
        labeled = [{"weak_image": x, "strong_image": x, "mask": y,
                    "image_index": torch.tensor([0, 1])}] * 2
        unlabeled = [{"weak_image": x, "strong_image": x}]
        val = [{"image": x, "mask": y}]
        mt = dict(NS["DEFAULT_MEAN_TEACHER_CONFIG"], contrastive=student.contrastive_config)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            cfg = dict(NS["DEFAULT_CONFIG"], device="cpu", use_amp=False, epochs=1,
                       model_dir=directory, checkpoint_dir=directory, save_every=1,
                       image_height=32, image_width=32, embed_dim=12, depths=[1]*4,
                       num_heads=[3,3,6,12], window_size=2, drop_path_rate=0)
            args = dict(student=student, teacher=teacher, labeled_loader=labeled, unlabeled_loader=unlabeled,
                        val_loader=val, supervised_criterion=nn.CrossEntropyLoss(),
                        consistency_criterion=NS["SoftmaxMSEConsistencyLoss"](), optimizer=optimizer,
                        scheduler=scheduler, config=cfg, mean_teacher_config=mt)
            _, _, history = NS["train_mean_teacher"](**args)
            self.assertGreater(history["contrastive_anchors"][0], 0)
            self.assertTrue(all(np.isfinite(v).all() for v in history.values()))
            checkpoint = next(Path(directory).glob("*epoch_001*.pth"))
            saved = torch.load(checkpoint, weights_only=False)
            self.assertIn("grad_scaler_state_dict", saved)
            self.assertIn("contrastive_memory_state_dict", saved)
            self.assertEqual(saved["selection_model"], "teacher")
            cfg["epochs"] = 2
            _, _, history = NS["train_mean_teacher"](**args, checkpoint_path=checkpoint)
            self.assertEqual(len(history["total_loss"]), 2)
            self.assertFalse(student.training)
            self.assertFalse(teacher.training)
            best = torch.load(next(Path(directory).glob("best_*.pth")), weights_only=False)
            base = NS["SwinUNet"](**student._teacher_kwargs)
            base.load_state_dict(best["teacher_state_dict"], strict=True)
            self.assertNotIn("contrastive_memory_state_dict", best)
            # A completed resume must restore the validation-selected epoch.
            completed = next(Path(directory).glob("*epoch_002*.pth"))
            _, _, restored = NS["train_mean_teacher"](**args, checkpoint_path=completed)
            self.assertEqual(len(restored["total_loss"]), 2)
            for name, value in student.state_dict().items():
                torch.testing.assert_close(value, best["student_state_dict"][name])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
