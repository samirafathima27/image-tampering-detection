"""
Starter Training Script (convert to a Colab/Kaggle notebook)
----------------------------------------------------------------
This is a plain-Python starting point for training FusionTamperNet.
Copy each section into its own cell in a Colab/Kaggle notebook.

Prerequisites:
  - Prepared dataset with images + 3-class labels + ground-truth masks
  - Directory structure expected (adjust to your actual layout):
        dataset/
          images/xxx.jpg
          masks/xxx.png        (0/255 binary mask, same filename as image)
          labels.csv            (filename, label)  label in {0,1,2}
"""

# ---- Cell 1: Setup (Colab) ----
# !pip install torch torchvision opencv-python-headless pillow numpy scikit-learn -q
# from google.colab import drive
# drive.mount('/content/drive')

# ---- Cell 2: Imports ----
import os
import sys
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

sys.path.append("/content/tampering-detection/src")  # adjust path after cloning your repo
from features.fusion import build_fused_input
from model.fusion_model import FusionTamperNet


# ---- Cell 3: Dataset class ----
class TamperDataset(Dataset):
    def __init__(self, root_dir, labels_csv, target_size=(256, 256)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.samples = []  # list of (filename, label)

        with open(labels_csv) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                self.samples.append((row[0], int(row[1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        image_path = os.path.join(self.root_dir, "images", filename)
        mask_path = os.path.join(self.root_dir, "masks", filename.rsplit(".", 1)[0] + ".png")

        fused = build_fused_input(image_path, target_size=self.target_size)  # HxWx10
        fused_tensor = torch.from_numpy(fused).float().permute(2, 0, 1) / 255.0

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.target_size)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        else:
            # "Real" images have no tampering — all-zero mask
            mask_tensor = torch.zeros(1, *self.target_size)

        return fused_tensor, torch.tensor(label, dtype=torch.long), mask_tensor


# ---- Cell 4: Training loop ----
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DATASET_ROOT = "/content/drive/MyDrive/tampering_dataset"  # adjust
    LABELS_CSV = os.path.join(DATASET_ROOT, "labels.csv")

    dataset = TamperDataset(DATASET_ROOT, LABELS_CSV)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    model = FusionTamperNet(in_channels=10, num_classes=3).to(device)
    class_criterion = nn.CrossEntropyLoss()
    mask_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 20
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for fused, labels, masks in train_loader:
            fused, labels, masks = fused.to(device), labels.to(device), masks.to(device)

            optimizer.zero_grad()
            class_logits, mask_logits = model(fused)

            loss_class = class_criterion(class_logits, labels)
            loss_mask = mask_criterion(mask_logits, masks)
            loss = loss_class + loss_mask  # equal weighting; tune if needed

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for fused, labels, masks in val_loader:
                fused, labels, masks = fused.to(device), labels.to(device), masks.to(device)
                class_logits, mask_logits = model(fused)
                loss = class_criterion(class_logits, labels) + mask_criterion(mask_logits, masks)
                val_loss += loss.item()

                preds = torch.argmax(class_logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={avg_train_loss:.4f} "
              f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print("  -> saved new best checkpoint")


if __name__ == "__main__":
    train()
