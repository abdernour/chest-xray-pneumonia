import torch
from torch.utils.data import DataLoader
import sys
import os

from pneumonia_cnn import (PneumoniaCNN, ChestXRayDataset,
                           MedicalDataAugmentation, visualize_xray_preprocessing)
from train_pneumonia import PneumoniaTrainer, evaluate_pneumonia_model


def train_pneumonia_detector(data_dir='archive/chest_xray', num_epochs=20, batch_size=32):
    """complete training pipeline"""

    print("\n" + "="*80)
    print("pneumonia detection - chest x-ray classification")
    print("="*80)

    # check data exists
    train_dir = os.path.join(data_dir, 'train')
    test_dir  = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir):
        print(f"\nerror: could not find training data at: {train_dir}")
        print("expected structure:")
        print("  archive/chest_xray/train/NORMAL/")
        print("  archive/chest_xray/train/PNEUMONIA/")
        print("  archive/chest_xray/test/NORMAL/")
        print("  archive/chest_xray/test/PNEUMONIA/")
        sys.exit(1)

    # ── 1. Load datasets ──────────────────────────────────────────────────────
    print("\n1. Loading datasets...")
    print("-" * 80)

    train_transform = MedicalDataAugmentation.get_train_transforms(img_size=224)
    val_transform   = MedicalDataAugmentation.get_val_transforms(img_size=224)

    train_dataset = ChestXRayDataset(train_dir, transform=train_transform, preprocess=True)
    test_dataset  = ChestXRayDataset(test_dir,  transform=val_transform,   preprocess=True)

    # num_workers=0 on windows to avoid multiprocessing errors
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\ndataset summary:")
    print(f"  training set:   {len(train_dataset)} images")
    print(f"  test set:       {len(test_dataset)} images")

    # ── 2. Preprocessing visualisation ───────────────────────────────────────
    print("\n2. Visualizing preprocessing pipeline...")
    print("-" * 80)
    visualize_xray_preprocessing(train_dataset.image_paths[0], 'xray_preprocessing_steps.png')

    # ── 3. Sample X-ray visualisation ────────────────────────────────────────
    print("\n3. Visualizing sample X-rays...")
    print("-" * 80)
    import matplotlib.pyplot as plt
    import cv2

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    normal_count = pneumonia_count = idx = 0
    for img_path, label in zip(train_dataset.image_paths[:300], train_dataset.labels[:300]):
        if idx >= 16:
            break
        if label == 0 and normal_count < 8:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title('NORMAL', color='green', fontweight='bold')
            axes[idx].axis('off')
            normal_count += 1; idx += 1
        elif label == 1 and pneumonia_count < 8:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title('PNEUMONIA', color='red', fontweight='bold')
            axes[idx].axis('off')
            pneumonia_count += 1; idx += 1

    plt.suptitle('sample chest x-rays from training set', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_xrays.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("sample x-rays saved to: sample_xrays.png")

    # ── 4. Create model ───────────────────────────────────────────────────────
    print("\n4. Creating CNN model...")
    print("-" * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = PneumoniaCNN(dropout=0.3)

    print(f"  architecture:  PneumoniaCNN")
    print(f"  parameters:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"  device:        {device}")
    print(f"  classes:       normal vs pneumonia")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("\n5. Starting training...")
    print("=" * 80)

    trainer = PneumoniaTrainer(model, device=device)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=0.0001,
        save_dir='pneumonia_checkpoints',
        early_stopping_patience=8
    )

    # ── 6. Training curves ────────────────────────────────────────────────────
    print("\n6. Generating training visualizations...")
    trainer.plot_training_history('pneumonia_training_curves.png')

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    print("\n7. Evaluating on test set...")
    print("=" * 80)

    checkpoint = torch.load(
        'pneumonia_checkpoints/best_model.pth',
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    test_acc, cm, roc_auc = evaluate_pneumonia_model(model, test_loader, device)

    # final summary
    print("\n" + "=" * 80)
    print("training complete")
    print("=" * 80)
    print(f"best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"test accuracy:            {test_acc:.2f}%")
    print(f"roc auc score:            {roc_auc:.4f}")
    print(f"\ngenerated files:")
    print(f"  pneumonia_checkpoints/best_model.pth")
    print(f"  pneumonia_training_curves.png")
    print(f"  pneumonia_confusion_matrix.png")
    print(f"  pneumonia_roc_curve.png")
    print(f"  xray_preprocessing_steps.png")
    print(f"  sample_xrays.png")
    print("=" * 80)

    return model, history, test_acc


def quick_demo(num_epochs=10):
    """quick demo with 10 epochs"""
    print("\n" + "="*80)
    print("quick demo mode (10 epochs)")
    print("="*80)
    return train_pneumonia_detector(
        data_dir='archive/chest_xray',
        num_epochs=num_epochs,
        batch_size=32
    )


if __name__ == "__main__":
    print("""
pneumonia detection - chest x-ray classification

options:
  - quick demo (10 epochs): python pneumonia_example.py demo
  - full training (20 epochs): python pneumonia_example.py
    """)

    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        model, history, test_acc = quick_demo(num_epochs=10)
    else:
        print("starting full training (20 epochs)...")
        print("with gpu: ~1-2 min per epoch")
        print("press ctrl+c to cancel\n")
        try:
            model, history, test_acc = train_pneumonia_detector(
                data_dir='archive/chest_xray',
                num_epochs=20,
                batch_size=32
            )
        except KeyboardInterrupt:
            print("\ntraining interrupted by user.")
            sys.exit(0)

    print("\n" + "="*80)
    print("done! run: python test_pneumonia.py")
    print("="*80)
