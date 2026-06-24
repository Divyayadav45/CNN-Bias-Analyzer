# Quantitative Evaluation of Inductive Bias: A Comparative Analysis of CNNs vs. Vision Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)](https://streamlit.io/)

An Explainable AI (XAI) diagnostics platform designed to measure the mathematical trade-offs between **structural shape contours** and **high-frequency surface textures** across deep learning vision paradigms.
# Quantitative Evaluation of Inductive Bias: CNNs vs Vision Transformers

An Explainable AI dashboard that measures whether deep learning models rely more on object shape or object texture when making predictions.

Built with:
- PyTorch
- ResNet18
- ResNet34
- Vision Transformer (ViT)
- Streamlit
- Grad-CAM
- FGSM & PGD Adversarial Attacks

Key Features:
- Shape Bias Analysis
- Texture Bias Analysis
- Grad-CAM Visualization
- Saliency Maps
- Adversarial Robustness Testing
- CNN vs ViT Comparison Dashboard

## 🔬 Research & Theoretical Motivation
Modern deep Convolutional Neural Networks (CNNs) are prone to shortcut learning, primarily over-indexing on localized high-frequency texture components. Conversely, Vision Transformers (ViTs)—leveraging global Multi-Head Self-Attention (MHSA) mechanisms—tend to prioritize global structural representations, mirroring human vision more closely.

This repository provides an empirical, math-justified evaluation framework to quantify, dissect, and compare these inductive biases under heavy domain transformations.

---

## 🧮 Core Metric Metrics
The framework decouples input visual streams into separate structural and stylistic domains to calculate four major metrics:

* **Shape Bias Formula ($B_{shape}$):**
    $$B_{\text{shape}} = \frac{\max(0, f(\mathbf{x}_{\text{shape}})_y)}{\max(0, f(\mathbf{x}_{\text{shape}})_y) + \max(0, f(\mathbf{x}_{\text{texture}})_y)}$$
* **Prediction Stability ($P_{stab}$):** Measured via Kullback-Leibler Divergence ($D_{KL}$) across stylized input manifolds.
* **Adversarial Fragility Metrics:** Integrated with FGSM and PGD optimization routines to test the alignment of decision boundaries against human perception.

---

## 🛠️ Installation & Reproduction Setup

### 1. Clone Project Environment
```bash
git clone 
cd cnn_vs_vit_bias_research