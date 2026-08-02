r"""
Columbia Dataset Standardization Script
-------------------------------------------
Prepares the Columbia Uncompressed Splicing dataset for generalization
TESTING (this dataset is never trained on — only used to evaluate how
well a model trained on CASIA generalizes to unseen data).

Expected input structure:
    <columbia_root>/
        4cam_auth/4cam_auth/       <- authentic .tif images
        4cam_splc/4cam_splc/       <- spliced .tif images
        4cam_splc/4cam_splc/edgemask/   <- masks, named <image_base>_edgemask.jpg

Usage:
    python standardize_columbia.py --columbia_root "D:\client 3\Columbia Uncompressed Image Splicing Detection" --target_size 256

Output:
    <columbia_root>/prepared/
        images/
        masks/
        labels.csv   (filename, label, mask_filename)  0=Real, 1=Photoshop-tampered
"""

import os
import csv
import argparse
from PIL import Image
import numpy as np


def find_subfolder(base_dir, name):
    """Handle the common double-nested folder pattern (e.g. 4cam_auth/4cam_auth)."""
    direct = os.path.join(base_dir, name)
    nested = os.path.join(direct, name)
    if os.path.isdir(nested):
        return nested
    return direct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--columbia_root", required=True)
    parser.add_argument("--target_size", type=int, default=256)
    args = parser.parse_args()

    auth_dir = find_subfolder(args.columbia_root, "4cam_auth")
    splc_dir = find_subfolder(args.columbia_root, "4cam_splc")
    mask_dir = os.path.join(splc_dir, "edgemask")

    for path, name in [(auth_dir, "4cam_auth"), (splc_dir, "4cam_splc"), (mask_dir, "edgemask")]:
        if not os.path.isdir(path):
            print(f"ERROR: could not find folder: {path}")
            return

    out_root = os.path.join(args.columbia_root, "prepared")
    out_images = os.path.join(out_root, "images")
    out_masks = os.path.join(out_root, "masks")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    size = (args.target_size, args.target_size)
    rows = []
    idx = 0
    skipped = 0

    # ---- Authentic images (label 0, no mask) ----
    auth_files = [f for f in os.listdir(auth_dir)
                  if f.lower().endswith((".tif", ".tiff", ".bmp")) and f.lower() != "thumbs.db"]

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

    # ---- Spliced images (label 1, with mask) ----
    splc_files = [f for f in os.listdir(splc_dir)
                  if f.lower().endswith((".tif", ".tiff", ".bmp")) and f.lower() != "thumbs.db"]

    print(f"Processing {len(splc_files)} spliced images...")
    for fname in splc_files:
        base = os.path.splitext(fname)[0]
        mask_fname = f"{base}_edgemask.jpg"
        mask_path = os.path.join(mask_dir, mask_fname)

        if not os.path.exists(mask_path):
            print(f"  WARNING: no mask found for {fname}, skipping")
            skipped += 1
            continue

        try:
            img = Image.open(os.path.join(splc_dir, fname)).convert("RGB")
            img = img.resize(size, Image.BILINEAR)
            out_name = f"{idx:06d}.jpg"
            img.save(os.path.join(out_images, out_name), "JPEG", quality=92)

            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(size, Image.NEAREST)
            out_mask_name = f"{idx:06d}.png"
            mask.save(os.path.join(out_masks, out_mask_name))

            rows.append([out_name, 1, out_mask_name])
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
    print(f"labels.csv ready for generalization testing (train on CASIA, test on this).")


if __name__ == "__main__":
    main()
