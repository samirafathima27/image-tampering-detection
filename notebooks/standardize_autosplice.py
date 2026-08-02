r"""
AutoSplice Dataset Standardization Script
----------------------------------------------
Prepares the AutoSplice AI-generated (DALL-E2) tampering dataset for
training as the "AI-tampered" class (label 2) in your 3-class system.

Expected input structure:
    <autosplice_root>/
        Authentic/            <- authentic images, named <id>.jpg
        Forged_JPEG100/         <- (or 90/75) forged images, named <id>.jpg
        Mask/                    <- masks, named <id>_mask.png

Usage:
    python standardize_autosplice.py --root "D:\client 3\AutoSplice\AutoSplice" --forged_folder Forged_JPEG100 --target_size 256

Output:
    <autosplice_root>/prepared/
        images/
        masks/
        labels.csv   (filename, label, mask_filename)
            0 = Real, 2 = AI-tampered  (2, to match the 3-class scheme:
            0=Real, 1=Photoshop-tampered, 2=AI-tampered)
"""

import os
import re
import csv
import argparse
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to the AutoSplice folder containing Authentic/, Forged_JPEGxx/, Mask/")
    parser.add_argument("--forged_folder", default="Forged_JPEG100",
                         help="Which forged variant to use (Forged_JPEG100, Forged_JPEG90, or Forged_JPEG75)")
    parser.add_argument("--target_size", type=int, default=256)
    args = parser.parse_args()

    auth_dir = os.path.join(args.root, "Authentic")
    forged_dir = os.path.join(args.root, args.forged_folder)
    mask_dir = os.path.join(args.root, "Mask")

    for path, name in [(auth_dir, "Authentic"), (forged_dir, args.forged_folder), (mask_dir, "Mask")]:
        if not os.path.isdir(path):
            print(f"ERROR: could not find folder: {path}")
            return

    out_root = os.path.join(args.root, "prepared")
    out_images = os.path.join(out_root, "images")
    out_masks = os.path.join(out_root, "masks")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    size = (args.target_size, args.target_size)
    rows = []
    idx = 0
    skipped = 0

    # ---- Authentic images (label 0, no mask) ----
    auth_files = [f for f in os.listdir(auth_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Processing {len(auth_files)} authentic images...")
    for fname in auth_files:
        try:
            img = Image.open(os.path.join(auth_dir, fname)).convert("RGB")
            img = img.resize(size, Image.BILINEAR)
            out_name = f"{idx:06d}.jpg"
            img.save(os.path.join(out_images, out_name), "JPEG", quality=92)

            blank_mask = Image.fromarray(np.zeros(size[::-1], dtype=np.uint8))
            out_mask_name = f"{idx:06d}.png"
            blank_mask.save(os.path.join(out_masks, out_mask_name))

            rows.append([out_name, 0, out_mask_name])
            idx += 1
        except Exception as e:
            print(f"  WARNING: skipped {fname}: {e}")
            skipped += 1

    # ---- AI-forged images (label 2, with mask) ----
    forged_files = [f for f in os.listdir(forged_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Processing {len(forged_files)} AI-forged images from {args.forged_folder}...")
    for fname in forged_files:
        base = os.path.splitext(fname)[0]  # e.g. "39406_0"
        # Forged filenames have a "_0"/"_1"/"_2" variant suffix; the mask
        # is shared across all variants of the same source image, so we
        # strip that suffix to get the source id, e.g. "39406_0" -> "39406"
        match = re.match(r"^(\d+)_\d+$", base)
        source_id = match.group(1) if match else base
        mask_fname = f"{source_id}_mask.png"
        mask_path = os.path.join(mask_dir, mask_fname)

        if not os.path.exists(mask_path):
            print(f"  WARNING: no mask found for {fname} (expected {mask_fname}), skipping")
            skipped += 1
            continue

        try:
            img = Image.open(os.path.join(forged_dir, fname)).convert("RGB")
            img = img.resize(size, Image.BILINEAR)
            out_name = f"{idx:06d}.jpg"
            img.save(os.path.join(out_images, out_name), "JPEG", quality=92)

            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(size, Image.NEAREST)
            out_mask_name = f"{idx:06d}.png"
            mask.save(os.path.join(out_masks, out_mask_name))

            rows.append([out_name, 2, out_mask_name])
            idx += 1
        except Exception as e:
            print(f"  WARNING: skipped {fname}: {e}")
            skipped += 1

    labels_path = os.path.join(out_root, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "mask_filename"])
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} images prepared, {skipped} skipped.")
    print(f"Output folder: {out_root}")
    print(f"labels.csv ready — label 0=Real, 2=AI-tampered")
    print(f"\nNEXT: merge this labels.csv with your CASIA labels.csv (label 1=Photoshop-tampered)")
    print(f"to build the full 3-class training set.")


if __name__ == "__main__":
    main()
