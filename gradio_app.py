import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os

from pneumonia_cnn import PneumoniaCNN, MedicalDataAugmentation, MedicalImagePreprocessor
from gradcam import GradCAM


class PneumoniaDetectorWithGradCAM:
    """enhanced detector with grad-cam visualization"""
    
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
            print(f"model not found - using untrained model for demo")
        
        self.model.to(self.device)
        self.model.eval()
        
        # setup grad-cam (target conv4)
        self.gradcam = GradCAM(self.model, self.model.conv4[-2])
        
        # preprocessing
        self.transform = MedicalDataAugmentation.get_val_transforms()
        self.preprocessor = MedicalImagePreprocessor()
    
    def preprocess_xray_array(self, image_array):
        """helper for array preprocessing"""
        image = self.preprocessor.normalize_intensity(image_array)
        image = self.preprocessor.enhance_contrast(image)
        image = self.preprocessor.remove_noise(image)
        return image
    
    def predict_with_gradcam(self, image, show_heatmap=True):
        """predict with grad-cam visualization"""
        try:
            # convert to grayscale array
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    image_gray = image
            else:
                image_gray = np.array(image.convert('L'))
            
            # preprocess
            preprocessed = self.preprocess_xray_array(image_gray)
            preprocessed_pil = Image.fromarray(preprocessed)
            
            # transform for model
            image_tensor = self.transform(preprocessed_pil).unsqueeze(0).to(self.device)
            
            # predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            normal_prob = probabilities[0][0].item()
            pneumonia_prob = probabilities[0][1].item()
            predicted_class = 1 if pneumonia_prob > normal_prob else 0
            
            # generate grad-cam
            if show_heatmap:
                cam = self.gradcam.generate_cam(image_tensor, target_class=predicted_class)
                overlay = self.gradcam.overlay_heatmap(image_gray, cam, alpha=0.4)
            else:
                overlay = image_gray
            
            # format diagnosis
            if predicted_class == 1:
                diagnosis = f"## 🔴 result: pneumonia detected\n\n"
                diagnosis += f"**confidence:** {pneumonia_prob*100:.2f}%\n\n"
                diagnosis += f"--- \n"
                diagnosis += f"**ai analysis:** red areas show where the ai detected signs of pneumonia.\n\n"
                diagnosis += f"**recommendation:** professional medical consultation is required.\n\n"
                diagnosis += f"_heatmap shows the lung regions that influenced the decision._"
            else:
                diagnosis = f"## 🟢 result: normal x-ray\n\n"
                diagnosis += f"**confidence:** {normal_prob*100:.2f}%\n\n"
                diagnosis += f"--- \n"
                diagnosis += f"**ai analysis:** no significant indicators of pneumonia detected.\n\n"
                diagnosis += f"**recommendation:** maintain regular checkups.\n\n"
                diagnosis += f"_the heatmap shows normal attention patterns across the lungs._"
            
            confidence_dict = {
                "normal": float(normal_prob),
                "pneumonia": float(pneumonia_prob)
            }
            
            return diagnosis, confidence_dict, overlay
            
        except Exception as e:
            error_msg = f"## ⚠️ error during analysis\n\n"
            error_msg += f"**details:** {str(e)}\n\n"
            error_msg += f"please ensure the image is a valid chest x-ray."
            return error_msg, {"error": 1.0}, None


# initialize detector
detector = PneumoniaDetectorWithGradCAM()


# interface callback
def analyze_xray(image, show_heatmap):
    """gradio callback"""
    return detector.predict_with_gradcam(image, show_heatmap)


# custom css
custom_css = """
.gradio-container {
    font-family: 'arial', sans-serif;
}
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
}
"""


# build interface
with gr.Blocks(css=custom_css, title="pneumonia detection ai") as demo:
    
    gr.Markdown(
        """
        # pneumonia detection ai
        
        ### chest x-ray analysis with explainable ai (grad-cam)
        
        upload a chest x-ray to detect pneumonia and see **where** the ai is looking.
        
        **model stats:**
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
            
            show_heatmap_checkbox = gr.Checkbox(
                label="show grad-cam heatmap",
                value=True
            )
            
            analyze_btn = gr.Button("analyze x-ray", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### interpretation guide:
                - **red/yellow**: high ai attention (potential infection)
                - **blue/green**: low ai attention (likely healthy)
                
                **important notice:** educational tool only. always consult healthcare professionals.
                """
            )
        
        with gr.Column():
            diagnosis_output = gr.Markdown(label="diagnosis result")
            confidence_output = gr.Label(label="confidence scores", num_top_classes=2)
            heatmap_output = gr.Image(label="grad-cam heatmap", type="numpy")
    
    gr.Markdown(
        """
        ### about this model
        
        this ai model was trained on 5,200+ chest x-ray images to classify normal vs pneumonia. it uses **grad-cam** to provide transparency by highlighting the regions of interest in the image.
        
        **technology stack:**
        - pytorch cnn (4.8m parameters)
        - medical image preprocessing (clahe)
        - explainable ai (grad-cam)
        
        ---
        built using pytorch and gradio
        """
    )
    
    analyze_btn.click(
        fn=analyze_xray,
        inputs=[input_image, show_heatmap_checkbox],
        outputs=[diagnosis_output, confidence_output, heatmap_output]
    )
    
    input_image.upload(
        fn=analyze_xray,
        inputs=[input_image, show_heatmap_checkbox],
        outputs=[diagnosis_output, confidence_output, heatmap_output]
    )


# launch
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
