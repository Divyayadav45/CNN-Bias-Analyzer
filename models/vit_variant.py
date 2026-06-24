# models/vit_variant.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResearchViT(nn.Module):
    """
    Vision Transformer wrapper designed to extract representations 
    and output logit layers from image patches.
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(ResearchViT, self).__init__()
        # Use a lightweight ViT model variant for proper integration 
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)
        
        # Interpolate positional embeddings to accept custom dimensions dynamically if needed
        # Overwrite head for CIFAR-10 target dimensions
        self.backbone.heads.head = nn.Linear(self.backbone.hidden_dim, num_classes)

    def forward(self, x):
        # ViT expects 224x224 scaled matrices natively; handle interpolation inside loader pipeline
        return self.backbone(x)