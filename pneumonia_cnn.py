import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


class GrayscaleToRGB:
    """convert grayscale pil image to rgb for 3-channel cnn input"""
    def __call__(self, img):
        return img.convert('RGB')


class MedicalImagePreprocessor:
    """
    preprocessing for chest x-ray images
    """

    @staticmethod
    def enhance_contrast(image):
        """apply clahe contrast enhancement"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def normalize_intensity(image):
        """normalize x-ray intensity values"""
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return (image * 255).astype(np.uint8)

    @staticmethod
    def remove_noise(image):
        """apply bilateral denoising"""
        return cv2.bilateralFilter(image, 9, 75, 75)

    @staticmethod
    def preprocess_xray(image_path):
        """full preprocessing pipeline for chest x-rays"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not load image: {image_path}")
        image = MedicalImagePreprocessor.normalize_intensity(image)
        image = MedicalImagePreprocessor.enhance_contrast(image)
        image = MedicalImagePreprocessor.remove_noise(image)
        return image


class ChestXRayDataset(Dataset):
    """
    dataset loader for chest x-ray images
    """

    def __init__(self, root_dir, transform=None, preprocess=True):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess = preprocess
        self.preprocessor = MedicalImagePreprocessor()

        self.image_paths = []
        self.labels = []

        # class 0 = normal, class 1 = pneumonia
        for class_name, label in [('NORMAL', 0), ('PNEUMONIA', 1)]:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

        print(f"loaded {len(self.image_paths)} images from {root_dir}")
        print(f"  normal: {self.labels.count(0)}")
        print(f"  pneumonia: {self.labels.count(1)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        if self.preprocess:
            image = self.preprocessor.preprocess_xray(img_path)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


class MedicalDataAugmentation:
    """
    data augmentation for x-ray images
    """

    @staticmethod
    def get_train_transforms(img_size=224):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            GrayscaleToRGB(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_val_transforms(img_size=224):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            GrayscaleToRGB(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class PneumoniaCNN(nn.Module):
    """
    cnn for pneumonia detection (binary classification)
    """

    def __init__(self, dropout=0.3):
        super(PneumoniaCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_feature_maps(self, x):
        features = {}
        x1 = self.conv1(x);  features['conv1'] = x1
        x2 = self.conv2(x1); features['conv2'] = x2
        x3 = self.conv3(x2); features['conv3'] = x3
        x4 = self.conv4(x3); features['conv4'] = x4
        return features


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_xray_preprocessing(image_path, save_path='preprocessing_comparison.png'):
    """visualize preprocessing steps on a sample x-ray"""
    preprocessor = MedicalImagePreprocessor()

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    normalized = preprocessor.normalize_intensity(original.copy())
    contrast_enhanced = preprocessor.enhance_contrast(original.copy())
    denoised = preprocessor.remove_noise(contrast_enhanced.copy())

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('original x-ray', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(normalized, cmap='gray')
    axes[0, 1].set_title('normalized intensity', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(contrast_enhanced, cmap='gray')
    axes[1, 0].set_title('clahe enhanced', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(denoised, cmap='gray')
    axes[1, 1].set_title('denoised (final)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle('chest x-ray preprocessing pipeline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"preprocessing visualization saved to: {save_path}")

    return fig


if __name__ == "__main__":
    print("pneumonia detection cnn")
    print("=" * 70)
    model = PneumoniaCNN(dropout=0.3)
    print(f"parameters: {count_parameters(model):,}")
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print(f"output shape: {out.shape}")
    print("model ok")
