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
    """Enhanced detector with Grad-CAM visualization"""
    
    def __init__(self, model_path='pneumonia_checkpoints/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Normal', 'Pneumonia']
        
        self.model = PneumoniaCNN(dropout=0.3)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device).eval()
        self.gradcam = GradCAM(self.model, self.model.conv4[-2])
        self.transform = MedicalDataAugmentation.get_val_transforms()
        self.preprocessor = MedicalImagePreprocessor()
    
    def preprocess_xray_array(self, image_array):
        """Helper for array preprocessing"""
        image = self.preprocessor.normalize_intensity(image_array)
        image = self.preprocessor.enhance_contrast(image)
        image = self.preprocessor.remove_noise(image)
        return image
    
    def predict_with_gradcam(self, image, show_heatmap=True):
        """Predict with Grad-CAM visualization"""
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    image_gray = image
            else:
                image_gray = np.array(image.convert('L'))
            
            preprocessed = self.preprocess_xray_array(image_gray)
            preprocessed_pil = Image.fromarray(preprocessed)
            image_tensor = self.transform(preprocessed_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            normal_prob = probabilities[0][0].item()
            pneumonia_prob = probabilities[0][1].item()
            predicted_class = 1 if pneumonia_prob > normal_prob else 0
            
            if show_heatmap:
                cam = self.gradcam.generate_cam(image_tensor, target_class=predicted_class)
                overlay = self.gradcam.overlay_heatmap(image_gray, cam, alpha=0.4)
            else:
                overlay = image_gray
            
            if predicted_class == 1:
                diagnosis = f"## Result: Pneumonia Detected\n\n"
                diagnosis += f"**Confidence:** {pneumonia_prob*100:.2f}%\n\n"
                diagnosis += f"--- \n"
                diagnosis += f"**Analysis:** The heatmap highlights regions with clinical signs of pneumonia.\n\n"
                diagnosis += f"**Recommendation:** Professional medical consultation is required.\n\n"
                diagnosis += f"_Visualization shows lung areas that influenced the model decision._"
            else:
                diagnosis = f"## Result: Normal X-Ray\n\n"
                diagnosis += f"**Confidence:** {normal_prob*100:.2f}%\n\n"
                diagnosis += f"--- \n"
                diagnosis += f"**Analysis:** No significant indicators of pneumonia detected.\n\n"
                diagnosis += f"**Recommendation:** Maintain regular checkups.\n\n"
                diagnosis += f"_The heatmap shows normal attention patterns across the lungs._"
            
            confidence_dict = {"Normal": float(normal_prob), "Pneumonia": float(pneumonia_prob)}
            return diagnosis, confidence_dict, overlay
            
        except Exception as e:
            error_msg = f"## Error during analysis\n\n**Details:** {str(e)}\n\nEnsure the image is a valid chest X-ray."
            return error_msg, {"Error": 1.0}, None

detector = PneumoniaDetectorWithGradCAM()

def analyze_xray(image, show_heatmap):
    return detector.predict_with_gradcam(image, show_heatmap)

# Midnight Minimal with subtle rounding
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=outfit:wght@300;400;600&family=inter:wght@300;400;500&display=swap');

.gradio-container {
    background-color: #0c0c0e !important;
    font-family: 'inter', sans-serif;
}

.main-header {
    text-align: center;
    padding: 2.5rem 0;
    font-family: 'outfit', sans-serif;
    background: linear-gradient(to right, #ffffff, #a1a1aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 600;
    letter-spacing: -0.03em;
    font-size: 2.5rem;
}

.minimal-card {
    background: #141416;
    border: 1px solid #232326;
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 24px;
    transition: all 0.3s ease;
}

.minimal-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.gr-button-primary {
    background: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    color: #000000 !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
}

.gr-button-primary:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

.result-stripe {
    border-top: 2px solid #3b82f6;
}

footer {
    display: none !important;
}

h1, h2, h3, p, label, .gr-form span, .gr-markdown, .gr-label {
    color: #e5e5e7 !important;
}

.gr-image, .gr-file {
    background: #1a1a1c !important;
    border: 1px solid #232326 !important;
    border-radius: 8px !important;
}

.info-text {
    color: #a1a1a6 !important;
    font-size: 0.95rem;
    line-height: 1.6;
}
"""

with gr.Blocks(css=custom_css, title="Pneumonia Detection") as demo:
    
    with gr.Column(elem_classes="minimal-card"):
        gr.Markdown(
            """
            <div class="main-header">Pneumonia Detection System</div>
            
            #### Automated Chest X-Ray Analysis & Visualization
            
            Identify patterns associated with pneumonia and visualize model attention regions.
            
            **System Metrics:**
            - **Accuracy:** 88.5% 
            - **Sensitivity:** 93.6% 
            - **ROC-AUC:** 0.95
            """
        )
    
    with gr.Row():
        with gr.Column(elem_classes="minimal-card"):
            input_image = gr.Image(
                label="Upload Chest X-Ray",
                type="pil",
                sources=["upload", "webcam"],
            )
            
            show_heatmap_checkbox = gr.Checkbox(
                label="Enable Heatmap Visualization",
                value=True
            )
            
            analyze_btn = gr.Button("Analyze X-Ray", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### Visualization Guide
                <div class="info-text">
                - Red Zones: High importance regions (potential indicators)<br>
                - Blue Zones: Low importance regions (likely healthy)
                </div>
                
                **Notice:** This is an educational tool. Always consult a medical professional.
                """
            )
        
        with gr.Column(elem_classes="minimal-card result-stripe"):
            diagnosis_output = gr.Markdown(label="Diagnosis Result")
            confidence_output = gr.Label(label="Confidence Levels", num_top_classes=2)
            heatmap_output = gr.Image(label="Focus Map (Grad-CAM)", type="numpy")
    
    with gr.Column(elem_classes="minimal-card"):
        gr.Markdown(
            """
            ### Technical Overview
            <div class="info-text">
            This module was developed using over 5,000 medical images to classify Normal vs Pneumonia scans. It utilizes Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight the specific lung regions that influenced the internal decision process.
            </div>
            
            **Implementation Details:**
            - **Convolutional Network:** Optimized binary classification architecture
            - **Medical Vision:** CLAHE contrast enhancement & bilateral filtering
            - **Interpretability:** Real-time heatmaps for local evidence detection
            
            ---
            <div style="text-align: center; color: #71717a; font-size: 0.85rem;">
                Built for medical transparency • Powered by PyTorch & Gradio
            </div>
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

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
