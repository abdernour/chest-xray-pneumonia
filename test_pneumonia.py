import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
import os
import sys

from pneumonia_cnn import PneumoniaCNN, MedicalDataAugmentation, MedicalImagePreprocessor, ChestXRayDataset


class PneumoniaPredictor:
    """predictor for pneumonia detection"""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.class_names = ['NORMAL', 'PNEUMONIA']

        self.model = PneumoniaCNN(dropout=0.3)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        self.transform = MedicalDataAugmentation.get_val_transforms()
        self.preprocessor = MedicalImagePreprocessor()

        print(f"model loaded from: {model_path}")
        print(f"  device: {device}")

    def predict_xray(self, image_path):
        """predict pneumonia from a single x-ray"""
        xray_preprocessed = self.preprocessor.preprocess_xray(image_path)
        xray_pil = Image.fromarray(xray_preprocessed)
        xray_tensor = self.transform(xray_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(xray_tensor)
            probabilities = F.softmax(outputs, dim=1)

        normal_prob = probabilities[0][0].item()
        pneumonia_prob = probabilities[0][1].item()
        predicted_class = 1 if pneumonia_prob > normal_prob else 0

        return {
            'predicted_class':      predicted_class,
            'predicted_label':      self.class_names[predicted_class],
            'confidence':           max(normal_prob, pneumonia_prob),
            'normal_probability':   normal_prob,
            'pneumonia_probability': pneumonia_prob,
            'preprocessed_image':   xray_preprocessed
        }

    def visualize_prediction(self, image_path, save_path=None):
        """visualize x-ray with prediction result"""
        result = self.predict_xray(image_path)

        original    = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        preprocessed = result['preprocessed_image']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # original
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('original chest x-ray', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # preprocessed
        axes[1].imshow(preprocessed, cmap='gray')
        axes[1].set_title('preprocessed (clahe + denoised)', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # prediction bar chart
        prediction_color = 'red' if result['predicted_class'] == 1 else 'green'
        bar_colors = [
            'green'    if result['predicted_class'] == 0 else 'lightgray',
            'red'      if result['predicted_class'] == 1 else 'lightgray'
        ]
        axes[2].barh(['NORMAL', 'PNEUMONIA'],
                     [result['normal_probability'], result['pneumonia_probability']],
                     color=bar_colors)
        axes[2].set_xlabel('probability', fontsize=12)

        # build title string separately
        title_line1 = f'diagnosis: {result["predicted_label"]}'
        title_line2 = f'confidence: {result["confidence"]*100:.1f}%'
        axes[2].set_title(title_line1 + '\n' + title_line2,
                          fontsize=14, fontweight='bold', color=prediction_color)
        axes[2].set_xlim([0, 1])

        axes[2].text(result['normal_probability'] + 0.02, 0,
                     f'{result["normal_probability"]*100:.1f}%',
                     va='center', fontsize=11, fontweight='bold')
        axes[2].text(result['pneumonia_probability'] + 0.02, 1,
                     f'{result["pneumonia_probability"]*100:.1f}%',
                     va='center', fontsize=11, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"prediction saved to: {save_path}")

        plt.savefig('last_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()

        return result


def test_on_dataset_samples(model_path='pneumonia_checkpoints/best_model.pth',
                            data_dir='archive/chest_xray/test',
                            num_samples=12):
    """test on random samples from the test set"""

    print("="*80)
    print("testing pneumonia detector on random x-rays")
    print("="*80)

    predictor = PneumoniaPredictor(model_path)

    # load test dataset
    test_dataset = ChestXRayDataset(data_dir, transform=None, preprocess=False)

    # pick balanced random samples
    normal_indices    = [i for i, l in enumerate(test_dataset.labels) if l == 0]
    pneumonia_indices = [i for i, l in enumerate(test_dataset.labels) if l == 1]

    np.random.shuffle(normal_indices)
    np.random.shuffle(pneumonia_indices)

    half = num_samples // 2
    selected = normal_indices[:half] + pneumonia_indices[:half]
    np.random.shuffle(selected)

    # make predictions
    correct = 0
    total   = 0

    print(f"\ntesting on {num_samples} random x-rays (balanced)...")
    print("-" * 80)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, test_idx in enumerate(selected):
        img_path        = test_dataset.image_paths[test_idx]
        true_label      = test_dataset.labels[test_idx]
        true_label_name = ['NORMAL', 'PNEUMONIA'][true_label]

        result               = predictor.predict_xray(img_path)
        predicted_label      = result['predicted_class']
        predicted_label_name = result['predicted_label']
        confidence           = result['confidence'] * 100

        total   += 1
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct += 1

        status = "correct" if is_correct else "wrong"
        color  = 'green'    if is_correct else 'red'

        print(f"\nx-ray #{test_idx}:")
        print(f"  true:       {true_label_name}")
        print(f"  predicted:  {predicted_label_name}")
        print(f"  confidence: {confidence:.1f}%")
        print(f"  result:     {status}")

        xray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        axes[idx].imshow(xray, cmap='gray')

        # build title without newline inside f-string
        t1 = f"true: {true_label_name}"
        t2 = f"pred: {predicted_label_name} ({confidence:.1f}%)"
        axes[idx].set_title(t1 + '\n' + t2, fontsize=9, color=color, fontweight='bold')
        axes[idx].axis('off')

    accuracy = (correct / total) * 100

    print("\n" + "="*80)
    print(f"results: {correct}/{total} correct  ({accuracy:.1f}% accuracy)")
    print("="*80)

    plt.suptitle(f'pneumonia detection results  ({correct}/{total} correct)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pneumonia_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nresults saved to: pneumonia_test_results.png")

    return accuracy


def predict_single_xray(image_path, model_path='pneumonia_checkpoints/best_model.pth'):
    """predict on a single x-ray image"""

    print("="*80)
    print("pneumonia detection - single x-ray prediction")
    print("="*80)

    predictor = PneumoniaPredictor(model_path)

    print(f"\nanalyzing: {image_path}")
    result = predictor.visualize_prediction(image_path, save_path='single_prediction.png')

    print("\n" + "="*80)
    print("diagnosis result:")
    print("="*80)
    print(f"  classification:   {result['predicted_label']}")
    print(f"  confidence:       {result['confidence']*100:.1f}%")
    print(f"  normal prob:      {result['normal_probability']*100:.1f}%")
    print(f"  pneumonia prob:   {result['pneumonia_probability']*100:.1f}%")
    print("="*80)

    if result['predicted_class'] == 1:
        print("\npneumonia detected. please consult a medical professional.")
    else:
        print("\nx-ray appears normal.")

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # predict on a specific image
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            predict_single_xray(image_path)
        else:
            print(f"error: image not found: {image_path}")
    else:
        # test on random samples from the test set
        print("testing on random samples from test dataset...")
        test_on_dataset_samples(num_samples=12)
