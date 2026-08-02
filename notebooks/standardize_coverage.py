r"""
COVERAGE Dataset Standardization Script
--------------------------------------------
Prepares the COVERAGE copy-move dataset for generalization TESTING
(never trained on — used to evaluate how well a model trained on CASIA
generalizes to unseen copy-move tampering).

Naming convention in COVERAGE:
    image/N.tif      -> authentic image
    image/Nt.tif      -> tampered version of image N
    mask/Nforged.tif   -> binary ground-truth mask for Nt.tif
    (mask/Ncopy.tif and mask/Npaste.tif are auxiliary, not used here)

Usage:
    python standardize_coverage.py --coverage_root "D:\client 3\coverage" --target_size 256

Output:
    <coverage_root>/prepared/
        images/
        masks/
        labels.csv   (filename, label, mask_filename)  0=Real, 1=Photoshop-tampered
"""

import os
import re
import csv
import argparse
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage_root", required=True)
    parser.add_argument("--target_size", type=int, default=256)
    args = parser.parse_args()

    image_dir = os.path.join(args.coverage_root, "image")
    mask_dir = os.path.join(args.coverage_root, "mask")

    for path, name in [(image_dir, "image"), (mask_dir, "mask")]:
        if not os.path.isdir(path):
            print(f"ERROR: could not find folder: {path}")
            return

    out_root = os.path.join(args.coverage_root, "prepared")
    out_images = os.path.join(out_root, "images")
    out_masks = os.path.join(out_root, "masks")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    size = (args.target_size, args.target_size)

    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".tif", ".tiff"))]

    # Separate authentic (e.g. "12.tif") from tampered (e.g. "12t.tif")
    # using a regex so "12.tif" and "12t.tif" aren't confused.
    authentic_files = []
    tampered_files = []
    for f in all_files:
        base = os.path.splitext(f)[0]
        if re.fullmatch(r"\d+t", base):
            tampered_files.append(f)
        elif re.fullmatch(r"\d+", base):
            authentic_files.append(f)
        else:
            print(f"  NOTE: unrecognized filename pattern, skipping: {f}")

    print(f"Found {len(authentic_files)} authentic images, {len(tampered_files)} tampered images.")

    rows = []
    idx = 0
    skipped = 0

    # ---- Authentic images ----
    for fname in authentic_files:
        try:
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
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

    # ---- Tampered images ----
    for fname in tampered_files:
        base = os.path.splitext(fname)[0]  # e.g. "12t"
        number = base[:-1]  # strip trailing "t" -> "12"
        mask_fname = f"{number}forged.tif"
        mask_path = os.path.join(mask_dir, mask_fname)

        if not os.path.exists(mask_path):
            print(f"  WARNING: no mask found for {fname} (expected {mask_fname}), skipping")
            skipped += 1
            continue

        try:
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
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
