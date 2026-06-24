# models/resnet_variants.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights

class ResearchResNet(nn.Module):
    """
    Resolution-calibrated ResNet implementation designed for CIFAR-10 dimensions (32x32).
    Modifies the initial 7x7 stride-2 convolution to prevent spatial downsampling bottlenecks.
    """
    def __init__(self, variant="resnet18", num_classes=10, pretrained=True):
        super(ResearchResNet, self).__init__()
        if variant == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
        elif variant == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = resnet34(weights=weights)
        else:
            raise ValueError("Target variant unsupported.")

        # Adapt initial convolution layer for low-resolution 32x32 inputs
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # Prevent extreme downsampling
        
        # Reconfigure linear output layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def get_final_conv_layer(self):
        return self.backbone.layer4[-1].conv2