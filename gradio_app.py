import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os

from pneumonia_cnn import PneumoniaCNN, MedicalDataAugmentation, MedicalImagePreprocessor

class PneumoniaDetector:
    def __init__(self, model_path='pneumonia_checkpoints/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['normal', 'pneumonia']
        self.model = PneumoniaCNN(dropout=0.3)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device).eval()
        self.transform = MedicalDataAugmentation.get_val_transforms()
        self.preprocessor = MedicalImagePreprocessor()
    
    def predict(self, image):
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image_pil = Image.fromarray(image.astype('uint8'))
            else:
                image_pil = image.convert('L')
            
            image_np = np.array(image_pil)
            preprocessed = self.preprocessor.preprocess_xray_array(image_np)
            image_tensor = self.transform(Image.fromarray(preprocessed)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = F.softmax(outputs, dim=1)[0]
            
            return {self.class_names[i]: float(probs[i]) for i in range(2)}
        except Exception as e:
            return {"error": 1.0}

def preprocess_xray_array(self, image_array):
    image = self.normalize_intensity(image_array)
    image = self.enhance_contrast(image)
    image = self.remove_noise(image)
    return image

MedicalImagePreprocessor.preprocess_xray_array = preprocess_xray_array
detector = PneumoniaDetector()

demo = gr.Interface(
    fn=detector.predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="pneumonia detection"
)

if __name__ == "__main__":
    demo.launch()
