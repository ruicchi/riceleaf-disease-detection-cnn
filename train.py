"""
Training script for Rice Leaf Disease CNN
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import RiceLeafCNN


# ──────────────────────────────────────
# Configuration
# ──────────────────────────────────────
DATA_DIR = "dataset"          # Root folder with subfolders per class
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
IMAGE_SIZE = 128
NUM_CLASSES = 4               # Bacterial Blight, Brown Spot, Leaf Smut, Healthy
MODEL_SAVE_PATH = "rice_leaf_cnn.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (should match subfolder names in dataset/)
CLASS_NAMES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]


# ──────────────────────────────────────
# Data Loading & Augmentation
# ──────────────────────────────────────
def get_data_loaders():
    """Create training and validation data loaders."""

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "train"),
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "val"),
        transform=val_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    return train_loader, val_loader


# ─────────��────────────────────────────
# Training Loop
# ──────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def validate(model, loader, criterion):
    """Validate and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


# ──────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    print("=" * 50)

    # Data
    train_loader, val_loader = get_data_loaders()

    # Model
    model = RiceLeafCNN(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training
    best_val_acc = 0.0
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()

        print(
            f"Epoch [{epoch + 1:2d}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✅ Best model saved ({val_acc:.2f}%)")

    print("\n" + "=" * 50)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()