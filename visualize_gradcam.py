import torch
import sys
import os

from pneumonia_cnn import PneumoniaCNN, MedicalDataAugmentation, MedicalImagePreprocessor
from gradcam import visualize_gradcam_prediction, compare_multiple_gradcams


def generate_single_gradcam(image_path, model_path='pneumonia_checkpoints/best_model.pth'):
    """generate grad-cam for single x-ray"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PneumoniaCNN(dropout=0.3)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    preprocessor = MedicalImagePreprocessor()
    transform = MedicalDataAugmentation.get_val_transforms()
    
    print(f"analyzing: {image_path}")
    result = visualize_gradcam_prediction(
        model=model,
        image_path=image_path,
        preprocessor=preprocessor,
        transform=transform,
        device=device,
        save_path='gradcam_single_result.png'
    )
    
    print(f"prediction: {result['predicted_label']} ({result['confidence']*100:.1f}%)")


def generate_comparison_gradcams(num_samples=6, model_path='pneumonia_checkpoints/best_model.pth'):
    """generate comparison for multiple x-rays"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PneumoniaCNN(dropout=0.3)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    from pneumonia_cnn import ChestXRayDataset
    import numpy as np
    
    test_dataset = ChestXRayDataset('archive/chest_xray/test', transform=None, preprocess=False)
    
    normal_indices = [i for i, l in enumerate(test_dataset.labels) if l == 0]
    pneumonia_indices = [i for i, l in enumerate(test_dataset.labels) if l == 1]
    
    np.random.shuffle(normal_indices)
    np.random.shuffle(pneumonia_indices)
    
    half = num_samples // 2
    selected_indices = normal_indices[:half] + pneumonia_indices[:half]
    np.random.shuffle(selected_indices)
    
    image_paths = [test_dataset.image_paths[i] for i in selected_indices]
    
    preprocessor = MedicalImagePreprocessor()
    transform = MedicalDataAugmentation.get_val_transforms()
    
    compare_multiple_gradcams(
        model=model,
        image_paths=image_paths,
        preprocessor=preprocessor,
        transform=transform,
        device=device,
        save_path='gradcam_comparison.png'
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("running default: comparison of 6 x-rays")
        generate_comparison_gradcams(num_samples=6)
    
    elif sys.argv[1] == 'single':
        if len(sys.argv) < 3:
            print("error: provide image path")
            sys.exit(1)
        generate_single_gradcam(sys.argv[2])
    
    elif sys.argv[1] == 'compare':
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        generate_comparison_gradcams(num_samples=num_samples)
    
    else:
        print("unknown option. use: single or compare")
