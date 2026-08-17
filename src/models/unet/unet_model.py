import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNetEncoder
from .unet_parts import Up, OutConv


class UNet(nn.Module):
    """
    U-Net with configurable encoder backbone.

    Example:
        UNet(
            backbone="resnet34",
            n_classes=3,
            pretrained=True
        )
    """

    def __init__(
        self,
        backbone="resnet34",
        n_classes=3,
        pretrained=True,
        bilinear=False,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.bilinear = bilinear

        # Encoder
        self.encoder = ResNetEncoder(
            backbone=backbone,
            pretrained=pretrained,
        )

        c0, c1, c2, c3, c4 = self.encoder.out_channels

        # Decoder
        self.up4 = Up(
            decoder_channels=c4,
            skip_channels=c3,
            out_channels=256,
            bilinear=bilinear,
        )

        self.up3 = Up(
            decoder_channels=256,
            skip_channels=c2,
            out_channels=128,
            bilinear=bilinear,
        )

        self.up2 = Up(
            decoder_channels=128,
            skip_channels=c1,
            out_channels=64,
            bilinear=bilinear,
        )

        self.up1 = Up(
            decoder_channels=64,
            skip_channels=c0,
            out_channels=64,
            bilinear=bilinear,
        )

        # ResNet conv1 gives H/2,
        # so one more upsampling is needed to recover H x W.
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.outc = OutConv(
            32,
            n_classes,
        )

    def forward(self, x, return_features=False):
        input_size = x.shape[-2:]

        # Encoder
        features = self.encoder(x)

        x0, x1, x2, x3, x4 = features

        # Decoder
        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)

        decoder_feature = self.final_up(x)

        logits = self.outc(decoder_feature)

        # Safety for odd image sizes
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        if return_features:
            return logits, {
                "encoder_0": x0,
                "encoder_1": x1,
                "encoder_2": x2,
                "encoder_3": x3,
                "bottleneck": x4,
                "decoder": decoder_feature,
            }

        return logits
