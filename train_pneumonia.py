import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

from pneumonia_cnn import PneumoniaCNN, ChestXRayDataset, MedicalDataAugmentation


class PneumoniaTrainer:
    """trainer for pneumonia detection"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   [],
            'val_sensitivity': [], 'val_specificity': [],
            'learning_rates': []
        }
        self.best_val_acc = 0.0

    def train_epoch(self, train_loader, criterion, optimizer):
        """train for one epoch"""
        self.model.train()
        running_loss = correct = total = 0

        pbar = tqdm(train_loader, desc='training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc':  f'{100 * correct / total:.2f}%'
            })

        return running_loss / total, 100 * correct / total

    def validate(self, val_loader, criterion):
        """validate with medical metrics"""
        self.model.eval()
        running_loss = 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                probabilities = self.model.predict_proba(inputs)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities[:, 1].cpu().numpy())

        epoch_loss = running_loss / len(all_labels)
        epoch_acc  = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return epoch_loss, epoch_acc, sensitivity, specificity, all_labels, all_preds, all_probs

    def train(self, train_loader, val_loader, num_epochs, learning_rate=0.0001,
              save_dir='pneumonia_checkpoints', early_stopping_patience=8):
        """complete training loop"""

        os.makedirs(save_dir, exist_ok=True)

        # class weights for imbalanced dataset
        class_counts = [0, 0]
        for _, label in train_loader.dataset:
            class_counts[label] += 1

        class_weights = torch.FloatTensor([
            1.0 / class_counts[0] if class_counts[0] > 0 else 1.0,
            1.0 / class_counts[1] if class_counts[1] > 0 else 1.0
        ]).to(self.device)
        class_weights = class_weights / class_weights.sum() * 2

        print(f"\nclass distribution:")
        print(f"  normal:    {class_counts[0]} samples")
        print(f"  pneumonia: {class_counts[1]} samples")
        print(f"  weights:   normal={class_weights[0]:.4f}, pneumonia={class_weights[1]:.4f}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        epochs_without_improvement = 0

        print(f"\n{'='*70}")
        print(f"training configuration:")
        print(f"  device:            {self.device}")
        print(f"  epochs:            {num_epochs}")
        print(f"  learning rate:     {learning_rate}")
        print(f"  batch size:        {train_loader.batch_size}")
        print(f"  training samples:  {len(train_loader.dataset)}")
        print(f"  validation samples:{len(val_loader.dataset)}")
        print(f"{'='*70}\n")

        for epoch in range(num_epochs):
            print(f"\nepoch {epoch + 1}/{num_epochs}")
            print("-" * 70)

            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc, sensitivity, specificity, _, _, _ = self.validate(val_loader, criterion)

            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']

            # save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_sensitivity'].append(sensitivity)
            self.history['val_specificity'].append(specificity)
            self.history['learning_rates'].append(current_lr)

            print(f"\nepoch {epoch + 1} summary:")
            print(f"  train loss: {train_loss:.4f} | train acc: {train_acc:.2f}%")
            print(f"  val loss:   {val_loss:.4f} | val acc:   {val_acc:.2f}%")
            print(f"  sensitivity (pneumonia): {sensitivity:.4f}")
            print(f"  specificity (normal):    {specificity:.4f}")
            print(f"  learning rate:           {current_lr:.6f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'history': self.history
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  saved best model (val acc: {val_acc:.2f}%)")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nearly stopping after {epoch + 1} epochs")
                break

        # save final model + history
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, os.path.join(save_dir, 'final_model.pth'))

        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)

        print(f"\n{'='*70}")
        print(f"training complete")
        print(f"best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*70}\n")

        return self.history

    def plot_training_history(self, save_path='pneumonia_training_curves.png'):
        """plot training curves with medical metrics"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)

        # loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'],   'r-', label='Val',   linewidth=2)
        axes[0, 0].set_title('loss', fontweight='bold'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        # accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'],   'r-', label='Val',   linewidth=2)
        axes[0, 1].set_title('accuracy (%)', fontweight='bold'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        # sensitivity & specificity
        axes[0, 2].plot(epochs, self.history['val_sensitivity'], 'g-',      label='sensitivity', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_specificity'], 'purple',  label='specificity', linewidth=2)
        axes[0, 2].set_title('medical metrics', fontweight='bold'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

        # learning rate
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'orange', linewidth=2)
        axes[1, 0].set_title('learning rate', fontweight='bold'); axes[1, 0].set_yscale('log'); axes[1, 0].grid(True, alpha=0.3)

        # overfitting gap
        gap = np.array(self.history['train_acc']) - np.array(self.history['val_acc'])
        axes[1, 1].plot(epochs, gap, 'brown', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_title('train-val gap', fontweight='bold'); axes[1, 1].grid(True, alpha=0.3)

        # balance metric
        f1_scores = [2*(s*sp)/(s+sp) if (s+sp) > 0 else 0
                     for s, sp in zip(self.history['val_sensitivity'], self.history['val_specificity'])]
        axes[1, 2].plot(epochs, f1_scores, 'teal', linewidth=2)
        axes[1, 2].set_title('sensitivity-specificity balance', fontweight='bold'); axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"training curves saved to: {save_path}")

        return fig


def evaluate_pneumonia_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """evaluation with medical metrics"""

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='evaluating'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = model.predict_proba(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())

    class_names = ['NORMAL', 'PNEUMONIA']

    print("\n" + "="*70)
    print("classification report:")
    print("="*70)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('pneumonia detection - confusion matrix', fontsize=16, fontweight='bold')
    plt.ylabel('true label'); plt.xlabel('predicted label')

    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm[i].sum() * 100
            plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                     ha='center', va='center', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('pneumonia_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("confusion matrix saved to: pneumonia_confusion_matrix.png")

    # roc curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('false positive rate'); plt.ylabel('true positive rate')
    plt.title('roc curve - pneumonia detection', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pneumonia_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("roc curve saved to: pneumonia_roc_curve.png")

    # medical metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"\nmedical performance metrics:")
    print(f"  sensitivity (pneumonia recall): {sensitivity:.4f}")
    print(f"  specificity (normal recall):    {specificity:.4f}")
    print(f"  positive predictive value:      {ppv:.4f}")
    print(f"  negative predictive value:      {npv:.4f}")
    print(f"  roc auc score:                  {roc_auc:.4f}")

    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\noverall test accuracy: {accuracy:.2f}%")

    return accuracy, cm, roc_auc
