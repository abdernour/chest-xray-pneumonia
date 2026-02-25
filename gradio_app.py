import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os

# import existing model components
from pneumonia_cnn import PneumoniaCNN, MedicalDataAugmentation, MedicalImagePreprocessor


class PneumoniaDetector:
    """wrapper class for gradio inference"""
    
    def __init__(self, model_path='pneumonia_checkpoints/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['normal', 'pneumonia']
        
        # load model
        self.model = PneumoniaCNN(dropout=0.3)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"model loaded from {model_path}")
        else:
            print(f"model not found at {model_path} - using untrained model for demo")
        
        self.model.to(self.device)
        self.model.eval()
        
        # preprocessing
        self.transform = MedicalDataAugmentation.get_val_transforms()
        self.preprocessor = MedicalImagePreprocessor()
    
    def predict(self, image):
        """
        predict pneumonia from chest x-ray
        """
        try:
            # handle different input types
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image_pil = Image.fromarray(image.astype('uint8'))
            else:
                image_pil = image.convert('L')
            
            # preprocess
            image_np = np.array(image_pil)
            preprocessed = self.preprocessor.preprocess_xray_array(image_np)
            preprocessed_pil = Image.fromarray(preprocessed)
            
            # transform and predict
            image_tensor = self.transform(preprocessed_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            normal_prob = probabilities[0][0].item()
            pneumonia_prob = probabilities[0][1].item()
            
            # determine result
            predicted_class = 1 if pneumonia_prob > normal_prob else 0
            
            # format diagnosis text
            if predicted_class == 1:
                diagnosis = f"pneumonia detected\n\n"
                diagnosis += f"confidence: {pneumonia_prob*100:.1f}%\n\n"
                diagnosis += f"this x-ray shows signs of pneumonia.\n"
                diagnosis += f"recommendation: consult a qualified medical professional immediately.\n\n"
                diagnosis += f"note: this is an ai screening tool, not a replacement for professional diagnosis."
            else:
                diagnosis = f"normal x-ray\n\n"
                diagnosis += f"confidence: {normal_prob*100:.1f}%\n\n"
                diagnosis += f"no signs of pneumonia detected.\n"
                diagnosis += f"this x-ray appears normal based on ai analysis.\n\n"
                diagnosis += f"note: ai screening should be verified by a healthcare professional."
            
            # confidence scores for the label output
            confidence_dict = {
                "normal": float(normal_prob),
                "pneumonia": float(pneumonia_prob)
            }
            
            return diagnosis, confidence_dict
            
        except Exception as e:
            error_msg = f"error during analysis\n\n{str(e)}\n\nplease ensure the image is a valid chest x-ray."
            return error_msg, {"error": 1.0}


# add helper method to preprocessor for array input
def preprocess_xray_array(self, image_array):
    """preprocess from numpy array"""
    image = self.normalize_intensity(image_array)
    image = self.enhance_contrast(image)
    image = self.remove_noise(image)
    return image

# monkey patch the method
MedicalImagePreprocessor.preprocess_xray_array = preprocess_xray_array


# initialize detector
detector = PneumoniaDetector()


# define gradio interface
def predict_pneumonia(image):
    """gradio callback function"""
    return detector.predict(image)


# examples
examples = [
    ["archive/chest_xray/test/NORMAL/IM-0001-0001.jpeg"],
    ["archive/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"]
]


# custom css for styling
custom_css = """
.gradio-container {
    font-family: 'arial', sans-serif;
}
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
}
"""


# build the interface
with gr.Blocks(css=custom_css, title="pneumonia detection ai") as demo:
    
    gr.Markdown(
        """
        # pneumonia detection ai
        
        ### ai-powered chest x-ray analysis
        
        upload a chest x-ray image to detect potential pneumonia using deep learning.
        
        **model performance:**
        - 88.5% test accuracy
        - 93.6% sensitivity
        - 0.95 roc-auc score
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="upload chest x-ray",
                type="pil",
                sources=["upload", "webcam"],
            )
            
            analyze_btn = gr.Button("analyze x-ray", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### instructions:
                1. upload a chest x-ray image (jpg, png)
                2. click "analyze x-ray"
                3. review the ai diagnosis
                """
            )
        
        with gr.Column():
            diagnosis_output = gr.Markdown(label="diagnosis result")
            confidence_output = gr.Label(label="confidence scores", num_top_classes=2)
    
    gr.Markdown(
        """
        ### about this model
        
        this ai model was trained on 5,200+ chest x-ray images to classify normal vs pneumonia.
        
        **technology stack:**
        - pytorch cnn (4.8m parameters)
        - medical image preprocessing (clahe enhancement)
        - class-weighted training
        
        built with ❤️ using pytorch and gradio
        """
    )
    
    analyze_btn.click(
        fn=predict_pneumonia,
        inputs=input_image,
        outputs=[diagnosis_output, confidence_output]
    )
    
    input_image.upload(
        fn=predict_pneumonia,
        inputs=input_image,
        outputs=[diagnosis_output, confidence_output]
    )


# launch config
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
