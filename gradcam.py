import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        """Register forward and backward hooks for gradient computation"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx):
        """Generate Class Activation Map for a specific class"""
        self.model.zero_grad()
        output = self.model(input_tensor)
        target_class = output[:, class_idx]
        target_class.backward()

        # Compute weights and generate CAM
        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()

        # Normalize CAM
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        return cam
    
