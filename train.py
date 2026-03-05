import sys
from unittest.mock import MagicMock

# Robust mock for _lzma to satisfy the lzma standard library on macOS environments missing the C extension
mock_lzma = MagicMock()
# Constants expected by lzma.py
for name in ['FORMAT_XZ', 'FORMAT_ALONE', 'FORMAT_RAW', 'FORMAT_AUTO', 
            'CHECK_NONE', 'CHECK_CRC32', 'CHECK_CRC64', 'CHECK_SHA256',
            'FILTER_LZMA1', 'FILTER_LZMA2', 'FILTER_DELTA', 'FILTER_X86', 'FILTER_IA64', 
            'FILTER_ARM', 'FILTER_ARMTHUMB', 'FILTER_SPARC', 'FILTER_POWERPC',
            'MF_BT2', 'MF_BT3', 'MF_BT4', 'MF_HC3', 'MF_HC4', 'MODE_READ', 'MODE_WRITE']:
    setattr(mock_lzma, name, 1)
sys.modules['_lzma'] = mock_lzma

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from model import EmotionCNN
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def train():
    """Train the CNN using the local image dataset in the 'archive' folder."""
    base_dir = 'archive'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found.")
        print("Please ensure your 'archive' folder contains 'train' and 'test' subdirectories.")
        return

    # Hyperparameters
    batch_size = 64
    lr = 0.001
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    early_stopping_patience = 15

    # Transforms (Normalization + Augmentation)
    # The models expects 48x48 grayscale
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False)
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset using ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Classes found: {train_dataset.classes}")
    
    # Calculate class weights for imbalance
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"Class weights: {dict(zip(train_dataset.classes, weights))}")
    
    # Initialize model
    model = EmotionCNN(num_classes=len(train_dataset.classes)).to(device)
    
    # RESUME SUPPORT: Load existing weights if they exist
    model_path = 'emotion_model.pth'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} to resume training...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Initial validation to set the baseline best accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        best_acc = 100 * correct / total
        print(f"Starting with baseline accuracy: {best_acc:.2f}%")
    else:
        print("No existing model found. Starting training from scratch.")
        best_acc = 0.0

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # OneCycleLR instead of generic plateau for better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                             steps_per_epoch=len(train_loader), 
                                             epochs=epochs)

    print(f"Starting training on {device} using local image files...")
    
    no_improve_epochs = 0
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            scheduler.step()
            
            if i % 100 == 99:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Metrics calculation
        acc = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Validation Accuracy: {acc:.2f}%")
        print(f"Validation Macro-F1: {macro_f1:.4f}")
        print(f"Validation Loss: {val_loss/len(test_loader):.4f}")
        
        # Save best model based on macro-F1 (better for imbalance)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_acc = acc
            torch.save(model.state_dict(), 'emotion_model.pth')
            print(">>> Best Macro-F1 improved! Model saved to emotion_model.pth")
            no_improve_epochs = 0
            
            # Print detailed report for best model
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
        else:
            no_improve_epochs += 1
            
        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print("Training finished.")

if __name__ == "__main__":
    train()
