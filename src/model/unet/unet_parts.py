import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm -> ReLU) * 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """
    Upsample decoder feature, concat encoder skip, rồi DoubleConv.
    """

    def __init__(
        self,
        decoder_channels,
        skip_channels,
        out_channels,
        bilinear=False,
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.Conv2d(
                    decoder_channels,
                    out_channels,
                    kernel_size=1,
                ),
            )
            up_channels = out_channels

        else:
            self.up = nn.ConvTranspose2d(
                decoder_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            )
            up_channels = out_channels

        self.conv = DoubleConv(
            up_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            n_classes,
            kernel_size=1,
        )

    def forward(self, x):
        return self.conv(x)
