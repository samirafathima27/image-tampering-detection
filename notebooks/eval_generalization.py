r"""
Generalization Evaluation Script
--------------------------------------
Loads the trained checkpoint and evaluates it on a dataset it was NEVER
trained on (Columbia or COVERAGE), reporting how well it generalizes.

This is the core research contribution of the project: honestly measuring
the accuracy drop when moving from the training distribution (CASIA +
AutoSplice) to completely unseen data.

Note: Columbia and COVERAGE only have 2 classes (0=Real, 1=Photoshop-
tampered) — they don't contain AI-tampered examples. The model still
outputs 3-class predictions; a prediction of class 2 (AI-tampered) on
these datasets is counted as simply wrong (not a special case), which is
the correct, honest way to score it.

Usage:
    python eval_generalization.py --checkpoint checkpoints/best_model.pt --dataset_root "/path/to/Columbia/prepared" --dataset_name Columbia
    python eval_generalization.py --checkpoint checkpoints/best_model.pt --dataset_root "/path/to/coverage/prepared" --dataset_name COVERAGE
"""

import os
import sys
import csv
import argparse

import numpy as np
import torch
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "..", "src")
sys.path.append(_SRC_DIR)

from features.fusion import build_fused_input
from model.fusion_model import FusionTamperNet

CLASS_NAMES = ["Real", "Photoshop-tampered", "AI-tampered"]


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """IoU between predicted (sigmoid probability) mask and ground-truth binary mask."""
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to trained best_model.pt")
    parser.add_argument("--dataset_root", required=True, help="Path to the prepared/ folder (must contain images/, masks/, labels.csv)")
    parser.add_argument("--dataset_name", required=True, help="Display name for this dataset (e.g. Columbia, COVERAGE)")
    parser.add_argument("--target_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FusionTamperNet(in_channels=10, num_classes=3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    labels_csv = os.path.join(args.dataset_root, "labels.csv")
    images_dir = os.path.join(args.dataset_root, "images")
    masks_dir = os.path.join(args.dataset_root, "masks")

    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        samples = [(row["filename"], int(row["label"]), row["mask_filename"]) for row in reader]

    print(f"Evaluating on {len(samples)} samples from {args.dataset_name}...")

    # 3x3 confusion matrix: rows = ground truth, cols = predicted
    confusion = np.zeros((3, 3), dtype=int)
    total_confidence = 0.0
    ious = []

    with torch.no_grad():
        for i, (filename, true_label, mask_filename) in enumerate(samples):
            image_path = os.path.join(images_dir, filename)
            fused = build_fused_input(image_path, target_size=(args.target_size, args.target_size))
            tensor = torch.from_numpy(fused).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            class_logits, mask_logits = model(tensor)
            probs = torch.softmax(class_logits, dim=1)[0]
            pred_label = int(torch.argmax(probs))
            confidence = float(probs[pred_label])

            confusion[true_label, pred_label] += 1
            total_confidence += confidence

            # Mask IoU only meaningful for actually-tampered ground truth
            if true_label != 0 and mask_filename:
                pred_mask = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
                gt_mask_path = os.path.join(masks_dir, mask_filename)
                gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    gt_mask = cv2.resize(gt_mask, (args.target_size, args.target_size))
                    ious.append(compute_mask_iou(pred_mask, gt_mask))

            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(samples)} processed")

    total = len(samples)
    exact_match_correct = int(np.trace(confusion))
    exact_accuracy = exact_match_correct / total

    # Binary accuracy: collapse "Photoshop-tampered" and "AI-tampered" into
    # one "Tampered" class, since Columbia/COVERAGE ground truth only
    # distinguishes Real vs Tampered — this is a fairer generalization
    # metric than exact 3-class match.
    binary_correct = 0
    for true_label in range(3):
        for pred_label in range(3):
            true_is_real = (true_label == 0)
            pred_is_real = (pred_label == 0)
            if true_is_real == pred_is_real:
                binary_correct += confusion[true_label, pred_label]
    binary_accuracy = binary_correct / total

    avg_confidence = total_confidence / total
    avg_iou = float(np.mean(ious)) if ious else float("nan")

    print(f"\n{'=' * 60}")
    print(f"GENERALIZATION RESULTS — {args.dataset_name} (never trained on)")
    print(f"{'=' * 60}")
    print(f"Total samples evaluated:      {total}")
    print(f"Exact 3-class accuracy:       {exact_accuracy:.4f} ({exact_match_correct}/{total})")
    print(f"Binary Real-vs-Tampered acc:  {binary_accuracy:.4f} ({binary_correct}/{total})")
    print(f"Average prediction confidence: {avg_confidence:.4f}")
    if ious:
        print(f"Average mask IoU (tampered samples only): {avg_iou:.4f}  (n={len(ious)})")
    else:
        print(f"Average mask IoU: N/A (no tampered samples with masks found)")

    print(f"\nConfusion matrix (rows=ground truth, cols=predicted):")
    print(f"{'':>20}", end="")
    for name in CLASS_NAMES:
        print(f"{name:>20}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>20}", end="")
        for j in range(3):
            print(f"{confusion[i, j]:>20}", end="")
        print()

    print(f"\nNote: {args.dataset_name} has no AI-tampered ground truth samples, so the")
    print(f"'AI-tampered' row will be all zeros — this is expected, not an error.")


if __name__ == "__main__":
    main()
