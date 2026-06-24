# analytical_math/explainability.py
import torch
import numpy as np
import cv2

class InterpretabilityManager:
    """Manages high-fidelity explainability maps for both spatial and patch-based model architectures."""
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

    def _save_gradient(self, grad):
        self.gradients = grad

    def generate_gradcam(self, input_tensor, target_class, target_layer):
        """Generates localized spatial Grad-CAM activation mapping arrays."""
        # Attach hooks to intercept activation vectors
        def hook_fn(module, input, output):
            self.activations = output
            output.register_hook(self._save_gradient)

        hook = target_layer.register_forward_hook(hook_fn)
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        one_hot = torch.zeros((1, output.size(-1)), dtype=torch.float32, device=input_tensor.device)
        one_hot[0][target_class] = 1.0
        
        output.backward(gradient=one_hot, retain_graph=True)
        hook.remove()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.max(cam, 0)
        cam = cv2.resize(cam, (input_tensor.size(-1), input_tensor.size(-2)))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) != 0 else cam
        return cam

    def generate_saliency(self, input_tensor, target_class):
        """Extracts first-order input gradient space optimizations."""
        input_tensor.requires_grad_()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward()
        
        saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
        saliency = saliency.cpu().numpy()[0]
        # Normalize heatmap boundaries
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency