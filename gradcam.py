import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    highlights the important regions in the input image that led to the model decision.
    red areas = high importance, blue areas = low importance
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """capture activations and gradients"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, target_class=None):
        """generate class activation map"""
        self.model.eval()
        
        # forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        # global average pooling
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # apply relu
        cam = np.maximum(cam, 0)
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam

    def overlay_heatmap(self, original_image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """overlay heatmap on original image"""
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        return overlay
