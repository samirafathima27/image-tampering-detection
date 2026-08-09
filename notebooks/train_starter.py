r"""
Training Script for FusionTamperNet (Colab-ready)
-------------------------------------------------------
Trains the fusion model on your merged CASIA + AutoSplice dataset.

COLAB SETUP (run these in separate cells first):
    !git clone https://github.com/<your-username>/<your-repo>.git
    %cd <your-repo>
    !pip install -q -r requirements.txt

    from google.colab import drive
    drive.mount('/content/drive')

USAGE:
    Quick test (recommended first — confirms the pipeline runs, ~2 min):
        python notebooks/train_starter.py --dataset_root "/content/drive/MyDrive/merged_training_data" --quick_test

    Full training run:
        python notebooks/train_starter.py --dataset_root "/content/drive/MyDrive/merged_training_data" --epochs 20

Expects dataset_root to contain:
    images/       <- .jpg images
    masks/         <- matching .png masks
    labels.csv      <- filename,label,mask_filename   (label: 0=Real,1=Photoshop,2=AI-tampered)
"""

import os
import sys
import csv
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

# Make src/ importable regardless of where this script is run from
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "..", "src")
sys.path.append(_SRC_DIR)

from features.fusion import build_fused_input
from model.fusion_model import FusionTamperNet


class TamperDataset(Dataset):
    def __init__(self, root_dir, labels_csv, target_size=(256, 256), subset_size=None):
        self.root_dir = root_dir
        self.target_size = target_size
        self.samples = []  # list of (filename, label, mask_filename)

        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filename"], int(row["label"]), row["mask_filename"]))

        if subset_size is not None:
            # Deterministic subset for quick testing (not random, so it's reproducible)
            step = max(1, len(self.samples) // subset_size)
            self.samples = self.samples[::step][:subset_size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label, mask_filename = self.samples[idx]
        image_path = os.path.join(self.root_dir, "images", filename)
        mask_path = os.path.join(self.root_dir, "masks", mask_filename)

        fused = build_fused_input(image_path, target_size=self.target_size)  # HxWx10
        fused_tensor = torch.from_numpy(fused).float().permute(2, 0, 1) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.target_size)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        return fused_tensor, torch.tensor(label, dtype=torch.long), mask_tensor


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected. In Colab, go to Runtime > Change runtime type > GPU.")

    labels_csv = os.path.join(args.dataset_root, "labels.csv")
    subset_size = 200 if args.quick_test else None
    dataset = TamperDataset(args.dataset_root, labels_csv, subset_size=subset_size)
    print(f"Dataset loaded: {len(dataset)} samples"
          f"{' (QUICK TEST SUBSET)' if args.quick_test else ''}")

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    num_workers = 0 if args.quick_test else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = FusionTamperNet(in_channels=10, num_classes=3).to(device)
    class_criterion = nn.CrossEntropyLoss()
    mask_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 2 if args.quick_test else args.epochs
    best_val_loss = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        for batch_idx, (fused, labels, masks) in enumerate(train_loader):
            fused, labels, masks = fused.to(device), labels.to(device), masks.to(device)

            optimizer.zero_grad()
            class_logits, mask_logits = model(fused)

            loss_class = class_criterion(class_logits, labels)
            loss_mask = mask_criterion(mask_logits, masks)
            loss = loss_class + loss_mask

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if args.quick_test:
                print(f"  batch {batch_idx+1}/{len(train_loader)} - loss: {loss.item():.4f}")

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
        elapsed = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{epochs} | train_loss={avg_train_loss:.4f} "
              f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved new best checkpoint to {ckpt_path}")

    if args.quick_test:
        print("\nQuick test complete. If this ran without errors, the pipeline works.")
        print("Now run again WITHOUT --quick_test for full training.")
    else:
        print(f"\nTraining complete. Best checkpoint: {args.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Path to merged_training_data folder")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Where to save best_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--quick_test", action="store_true",
                         help="Run on a small subset for 2 epochs to verify the pipeline works")
    args = parser.parse_args()
    train(args)