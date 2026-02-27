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


def visualize_gradcam_prediction(model, image_path, preprocessor, transform, 
                                 device='cuda', save_path='gradcam_result.png'):
    """visualization pipeline"""
    model.eval()
    
    # load and preprocess
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    preprocessed = preprocessor.preprocess_xray(image_path)
    preprocessed_pil = Image.fromarray(preprocessed)
    
    # transform for model
    image_tensor = transform(preprocessed_pil).unsqueeze(0).to(device)
    
    # get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # generate grad-cam (target conv4)
    gradcam = GradCAM(model, model.conv4[-2])
    cam = gradcam.generate_cam(image_tensor, target_class=predicted_class)
    
    # create overlay
    overlay = gradcam.overlay_heatmap(original, cam, alpha=0.4)
    
    # visualize
    class_names = ['normal', 'pneumonia']
    predicted_label = class_names[predicted_class]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('original x-ray', fontsize=14, fontweight='bold'); axes[0].axis('off')
    
    axes[1].imshow(preprocessed, cmap='gray')
    axes[1].set_title('preprocessed (clahe)', fontsize=14, fontweight='bold'); axes[1].axis('off')
    
    axes[2].imshow(cam, cmap='jet')
    axes[2].set_title('heatmap', fontsize=14, fontweight='bold'); axes[2].axis('off')
    
    axes[3].imshow(overlay)
    title_color = 'red' if predicted_class == 1 else 'green'
    axes[3].set_title(f'result: {predicted_label}\nconfidence: {confidence*100:.1f}%',
                     fontsize=14, fontweight='bold', color=title_color)
    axes[3].axis('off')
    
    plt.suptitle('grad-cam visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"grad-cam visualization saved to: {save_path}")
    
    return {
        'predicted_class': predicted_class,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'cam': cam,
        'overlay': overlay
    }


def compare_multiple_gradcams(model, image_paths, preprocessor, transform, 
                               device='cuda', save_path='gradcam_comparison.png'):
    """compare multiple x-rays side by side"""
    num_images = len(image_paths)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    class_names = ['normal', 'pneumonia']
    gradcam = GradCAM(model, model.conv4[-2])
    
    for idx, img_path in enumerate(image_paths):
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        preprocessed = preprocessor.preprocess_xray(img_path)
        preprocessed_pil = Image.fromarray(preprocessed)
        image_tensor = transform(preprocessed_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, predicted_class].item()
        
        cam = gradcam.generate_cam(image_tensor, target_class=predicted_class)
        overlay = gradcam.overlay_heatmap(original, cam, alpha=0.4)
        
        axes[idx, 0].imshow(original, cmap='gray'); axes[idx, 0].axis('off')
        axes[idx, 1].imshow(cam, cmap='jet'); axes[idx, 1].axis('off')
        axes[idx, 2].imshow(overlay); axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"comparison saved to: {save_path}")
