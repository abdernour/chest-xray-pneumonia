import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os
from typing import Any, Dict, Tuple, Optional

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

def analyze_xray(image: Any, show_heatmap: bool) -> Tuple[str, Dict[str, float], Optional[np.ndarray]]:
    """
    Run a single chest X-ray through the detector.
    Used as the callable for the Gradio interface.
    """
    return detector.predict_with_gradcam(image, show_heatmap)

def load_sample_xray(show_heatmap: bool) -> Tuple[Optional[Image.Image], str, Dict[str, float], Optional[np.ndarray]]:
    """
    Load a bundled sample chest X-ray and run the full pipeline.
    Expects an image file at assets/sample_xray.jpg.
    """
    sample_path = os.path.join("assets", "sample_xray.jpg")
    
    if not os.path.exists(sample_path):
        diagnosis = (
            "## Sample image not found\n\n"
            "Place a chest X-ray at `assets/sample_xray.jpg` and reload the app."
        )
        # Clear image/heatmap but still return a valid structure
        return None, diagnosis, {"Normal": 0.0, "Pneumonia": 0.0}, None
    
    image = Image.open(sample_path).convert("RGB")
    diagnosis, confidence_dict, overlay = detector.predict_with_gradcam(image, show_heatmap)
    return image, diagnosis, confidence_dict, overlay

# Midnight Minimal with cleaner, faster layout
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Inter:wght@300;400;500&display=swap');

.gradio-container {
    background-color: #050509 !important;
    font-family: 'Inter', sans-serif;
    padding: 2.5rem 0 !important;
}

.gr-blocks {
    max-width: 1120px;
    margin: 0 auto !important;
}

.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(to right, #ffffff, #a1a1aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 600;
    letter-spacing: -0.03em;
    font-size: 2.6rem;
}

.subtitle {
    text-align: center;
    color: #a1a1a6;
    margin-top: 0.4rem;
    margin-bottom: 1.4rem;
    font-size: 0.98rem;
}

.metrics-row {
    display: flex;
    justify-content: center;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}

.metric-pill {
    background: linear-gradient(135deg, #111827, #020617);
    border-radius: 4px;
    padding: 0.55rem 1rem;
    border: 1px solid rgba(148, 163, 184, 0.55);
    display: flex;
    flex-direction: column;
    min-width: 110px;
}

.metric-label {
    color: #9ca3af;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-value {
    color: #e5e7eb;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.minimal-card {
    background: #050816;
    border: 1px solid #232326;
    border-radius: 6px;
    padding: 28px 28px 26px 28px;
    margin-bottom: 20px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.minimal-card:hover {
    border-color: #71717a;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.7);
}

.gr-row {
    gap: 1.75rem !important;
    align-items: start;
}

.gr-button-primary, button.primary {
    background: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    color: #020617 !important;
    font-weight: 600 !important;
    padding: 12px 26px !important;
    transition: box-shadow 0.15s ease, opacity 0.15s ease !important;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.55);
}

.gr-button-primary:hover,
button.primary:hover {
    opacity: 0.96;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.75);
}

.result-stripe {
    border-top: 2px solid #52525b;
    border-radius: 6px;
}

/* Emphasize the diagnosis result as the main answer */
#diagnosis-result {
    border-left: 3px solid #71717a;
    padding-left: 1rem;
}

.result-stripe .gr-label {
    font-size: 0.8rem;
    color: #a1a1a6 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}

.notice-text {
    color: #71717a !important;
    font-size: 0.85rem;
    margin-top: 1rem;
}

footer {
    display: none !important;
}

h1, h2, h3, p, label, .gr-form span, .gr-markdown, .gr-label {
    color: #e5e5e7 !important;
}

.gr-image, .gr-file {
    background: #020617 !important;
    border: 1px solid #232326 !important;
    border-radius: 6px !important;
}

.gr-image img {
    object-fit: contain;
    border-radius: 4px;
}

.info-text {
    color: #9ca3af !important;
    font-size: 0.94rem;
    line-height: 1.7;
}

.gr-checkbox, .gr-checkbox label {
    color: #e5e5e7 !important;
}
"""

with gr.Blocks(css=custom_css, title="Pneumonia Detection") as demo:
    
    with gr.Column(elem_classes="minimal-card"):
        gr.Markdown(
            """
            <div class="main-header">Pneumonia Detection System</div>
            <p class="subtitle">Automated chest X-ray triage with Grad-CAM explainability.</p>

            <div class="metrics-row">
                <div class="metric-pill">
                    <span class="metric-label">Accuracy</span>
                    <span class="metric-value">88.5%</span>
                </div>
                <div class="metric-pill">
                    <span class="metric-label">Sensitivity</span>
                    <span class="metric-value">93.6%</span>
                </div>
                <div class="metric-pill">
                    <span class="metric-label">ROC-AUC</span>
                    <span class="metric-value">0.95</span>
                </div>
            </div>
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
            
            with gr.Row():
                sample_btn = gr.Button("Use Sample X-Ray", variant="secondary")
                analyze_btn = gr.Button("Analyze X-Ray", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### Visualization Guide
                <div class="info-text">
                - Red Zones: High importance regions (potential indicators)<br>
                - Blue Zones: Low importance regions (likely healthy)
                </div>
                
                <span class="notice-text">Notice: This is an educational tool. Always consult a medical professional.</span>
                """
            )
        
        with gr.Column(elem_classes="minimal-card result-stripe"):
            diagnosis_output = gr.Markdown(label="Diagnosis Result", elem_id="diagnosis-result")
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
    
    sample_btn.click(
        fn=load_sample_xray,
        inputs=[show_heatmap_checkbox],
        outputs=[input_image, diagnosis_output, confidence_output, heatmap_output]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
