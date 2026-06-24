# pipelines/adversarial_attacks.py
import torch
import torch.nn as nn

class AdversarialEngine:
    """Generates adversarial modifications to evaluate model structural resilience."""
    @staticmethod
    def fgsm_attack(model, image, epsilon, data_grad):
        """Computes Fast Gradient Sign Method perturbation vectors."""
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        # Maintain pixel boundaries
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    @staticmethod
    def generate_pgd(model, image, target_label, epsilon=8/255, alpha=2/255, steps=10):
        """Generates Projected Gradient Descent iterative mapping attacks."""
        device = image.device
        criterion = nn.CrossEntropyLoss()
        perturbed_image = image.clone().detach().requires_grad_(True).to(device)
        
        for _ in range(steps):
            outputs = model(perturbed_image)
            loss = criterion(outputs, target_label)
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                # Step gradient alignment direction
                adv_step = perturbed_image + alpha * perturbed_image.grad.sign()
                # Enforce epsilon configuration boundaries
                eta = torch.clamp(adv_step - image, min=-epsilon, max=epsilon)
                perturbed_image = torch.clamp(image + eta, min=0, max=1).detach()
            perturbed_image.requires_grad_(True)
            
        return perturbed_image