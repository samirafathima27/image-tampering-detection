r"""
CASIA v2 Standardization Script
------------------------------------
Reads casia_labels.csv (produced by pair_casia.py), resizes every image
and mask to a fixed size, converts everything to consistent formats
(.jpg for images, .png for masks), and writes them into a clean
"prepared" folder ready for training.

Usage:
    python standardize_casia.py --casia_root "D:\client 3\archive\CASIA2" --target_size 256

Output structure created:
    <casia_root>/prepared/
        images/   <- all resized images (both Au and Tp), renamed by index
        masks/    <- resized masks for Tp images; Au images get an all-black mask
        labels.csv  <- filename, label, mask_filename (matches train_starter.py format)
"""

import os
import csv
import argparse
from PIL import Image
import numpy as np


def load_image_safe(path):
    """Load an image, handling .tif and other formats via PIL."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"  WARNING: could not open {path}: {e}")
        return None


def load_mask_safe(path):
    try:
        mask = Image.open(path).convert("L")  # grayscale
        return mask
    except Exception as e:
        print(f"  WARNING: could not open mask {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--casia_root", required=True)
    parser.add_argument("--gt_folder_name", default="CASIA 2 Groundtruth")
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--labels_csv", default="casia_labels.csv")
    args = parser.parse_args()

    au_dir = os.path.join(args.casia_root, "Au")
    tp_dir = os.path.join(args.casia_root, "Tp")
    gt_dir = os.path.join(args.casia_root, args.gt_folder_name)
    labels_path = os.path.join(args.casia_root, args.labels_csv)

    out_root = os.path.join(args.casia_root, "prepared")
    out_images = os.path.join(out_root, "images")
    out_masks = os.path.join(out_root, "masks")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    size = (args.target_size, args.target_size)

    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} entries from {labels_path}...")

    output_rows = []
    skipped = 0

    for i, row in enumerate(rows):
        filename = row["filename"]
        label = row["label"]
        mask_filename = row["mask_filename"]
        source_folder = row["source_folder"]

        src_dir = au_dir if source_folder == "Au" else tp_dir
        src_path = os.path.join(src_dir, filename)

        img = load_image_safe(src_path)
        if img is None:
            skipped += 1
            continue

        img_resized = img.resize(size, Image.BILINEAR)
        out_filename = f"{i:06d}.jpg"
        img_resized.save(os.path.join(out_images, out_filename), "JPEG", quality=92)

        # Handle mask
        out_mask_filename = ""
        if mask_filename:
            mask_path = os.path.join(gt_dir, mask_filename)
            mask = load_mask_safe(mask_path)
            if mask is not None:
                mask_resized = mask.resize(size, Image.NEAREST)  # NEAREST preserves binary edges
                out_mask_filename = f"{i:06d}.png"
                mask_resized.save(os.path.join(out_masks, out_mask_filename))
        else:
            # Authentic image -> all-black (zero) mask, since nothing is tampered
            blank_mask = Image.fromarray(np.zeros(size[::-1], dtype=np.uint8))
            out_mask_filename = f"{i:06d}.png"
            blank_mask.save(os.path.join(out_masks, out_mask_filename))

        output_rows.append([out_filename, label, out_mask_filename])

        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(rows)} processed")

    out_labels_path = os.path.join(out_root, "labels.csv")
    with open(out_labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "mask_filename"])
        writer.writerows(output_rows)

    print(f"\nDone. {len(output_rows)} images prepared, {skipped} skipped (unreadable files).")
    print(f"Output folder: {out_root}")
    print(f"  images/   -> {len(output_rows)} resized {args.target_size}x{args.target_size} JPGs")
    print(f"  masks/    -> matching resized PNG masks (all-black for authentic images)")
    print(f"  labels.csv -> ready for train_starter.py")


if __name__ == "__main__":
    main()
