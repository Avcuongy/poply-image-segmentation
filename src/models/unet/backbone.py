import torch.nn as nn

from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)


class ResNetEncoder(nn.Module):
    """
    ResNet encoder backbone.
    Supports resnet18, resnet34, resnet50, resnet101.
    Returns 5 feature maps: [x0, x1, x2, x3, x4] with channels = out_channels.
    """

    def __init__(self, backbone="resnet34", pretrained=True):
        super().__init__()

        backbone = backbone.lower()

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
            self.out_channels = [64, 64, 128, 256, 512]

        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            model = resnet34(weights=weights)
            self.out_channels = [64, 64, 128, 256, 512]

        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
            self.out_channels = [64, 256, 512, 1024, 2048]

        elif backbone == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            model = resnet101(weights=weights)
            self.out_channels = [64, 256, 512, 1024, 2048]

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported: resnet18, resnet34, resnet50, resnet101"
            )

        # Stem
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        # ResNet stages
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))  # H/2,  C = out_channels[0]
        x = self.maxpool(x0)  # H/4
        x1 = self.layer1(x)  # H/4,  C = out_channels[1]
        x2 = self.layer2(x1)  # H/8,  C = out_channels[2]
        x3 = self.layer3(x2)  # H/16, C = out_channels[3]
        x4 = self.layer4(x3)  # H/32, C = out_channels[4]
        return [x0, x1, x2, x3, x4]

    def freeze(self):
        """Freeze all encoder parameters to avoid training them."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """Unfreeze all encoder parameters for fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True
