import copy
import torch
from torch import nn
from torch.nn import functional as F
from .unet import UNet
from .swin_unet import SwinUNet
from loss.contrastive import validate_contrastive_config


class PixelProjectionHead(nn.Module):
    """ContrastiveSeg-style dense pixel projection head.

    Reference implementation:
        tfzhou/ContrastiveSeg/lib/models/modules/projection.py

    The official head is:
        1x1 Conv -> BN/ReLU -> 1x1 Conv -> L2 normalization.

    BatchNorm2d is used here instead of SyncBatchNorm because this notebook is
    designed for ordinary single-device / Kaggle training.
    """

    def __init__(self, dim_in, proj_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, proj_dim, kernel_size=1),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class ContrastiveUNet(UNet):
    """U-Net Student with a Student-only dense pixel projection head.

    Segmentation inference is unchanged: ``forward(x)`` still returns only
    segmentation logits. The contrastive head is called explicitly from the
    training loop on a cached Student feature map, so the encoder is NOT run a
    second time.
    """

    def __init__(self, contrastive_config, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_config = copy.deepcopy(contrastive_config)
        validate_contrastive_config(self.contrastive_config)
        if self.contrastive_config["feature_key"] != "decoder":
            raise ValueError(
                "The paper-memory U-Net adaptation uses final decoder features"
            )
        dim_in = self.outc.conv.in_channels
        proj_dim = int(self.contrastive_config["embedding_dim"])

        devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.contrastive_config["seed"]))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.contrastive_config["seed"]))
            self.projection_head = PixelProjectionHead(
                dim_in=dim_in,
                proj_dim=proj_dim,
            )

    def project_pixels(self, feature_map):
        """Project a dense Student feature map to normalized pixel embeddings."""
        stride = int(self.contrastive_config["feature_stride"])
        if stride < 1 or min(feature_map.shape[-2:]) < stride:
            raise ValueError("Invalid contrastive feature stride")
        pooled = F.avg_pool2d(feature_map, kernel_size=stride, stride=stride)
        return self.projection_head(pooled)


class ContrastiveSwinUNet(SwinUNet):
    """Swin Student with a training-only projection of final decoder features."""

    def __init__(self, contrastive_config, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_config = copy.deepcopy(contrastive_config)
        validate_contrastive_config(self.contrastive_config)
        self._teacher_kwargs = copy.deepcopy(kwargs)
        self._teacher_kwargs["pretrained_path"] = None
        self._teacher_kwargs["pretrained_decoder"] = False
        devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.contrastive_config["seed"]))
            self.projection_head = PixelProjectionHead(
                dim_in=self.segmentation_head.in_channels,
                proj_dim=int(self.contrastive_config["embedding_dim"]),
            )

    def project_pixels(self, feature_map):
        stride = int(self.contrastive_config["feature_stride"])
        if min(feature_map.shape[-2:]) < stride:
            raise ValueError("Contrastive feature stride exceeds feature resolution")
        with torch.autocast(device_type=feature_map.device.type, enabled=False):
            pooled = F.avg_pool2d(
                feature_map.float(), kernel_size=stride, stride=stride
            )
            return self.projection_head(pooled)
