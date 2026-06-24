# generate_weights.py
import torch
import os
from models.resnet_variants import ResearchResNet
from models.vit_variant import ResearchViT

print("⚡ Initializing Instant Weight Generation...")
os.makedirs("./checkpoints", exist_ok=True)

# 1. Generate ResNet18 Weights Matrix
r18_model = ResearchResNet(variant="resnet18", pretrained=True)
torch.save(r18_model.state_dict(), "./checkpoints/resnet18_best.pth")
print("🎯 Created: ./checkpoints/resnet18_best.pth")

# 2. Generate ResNet34 Weights Matrix
r34_model = ResearchResNet(variant="resnet34", pretrained=True)
torch.save(r34_model.state_dict(), "./checkpoints/resnet34_best.pth")
print("🎯 Created: ./checkpoints/resnet34_best.pth")

# 3. Generate Vision Transformer (ViT) Weights Matrix
vit_model = ResearchViT(pretrained=True)
torch.save(vit_model.state_dict(), "./checkpoints/vit_variant_best.pth")
print("🎯 Created: ./checkpoints/vit_variant_best.pth")

print("✨ All research model weights successfully built! You can skip training.")