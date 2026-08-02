r"""
Merge Training Datasets Script
------------------------------------
Combines the prepared CASIA v2 (Real + Photoshop-tampered) and AutoSplice
(Real + AI-tampered) datasets into ONE unified training dataset with
correct 3-class labels, avoiding filename collisions.

Label scheme (matches the rest of the project):
    0 = Real
    1 = Photoshop-tampered  (from CASIA)
    2 = AI-tampered          (from AutoSplice)

Usage:
    python merge_training_data.py ^
        --casia_prepared "D:\client 3\archive\CASIA2\prepared" ^
        --autosplice_prepared "D:\client 3\AutoSplice\AutoSplice\prepared" ^
        --output "D:\client 3\merged_training_data"

Output:
    <output>/images/    <- all images, renamed to avoid collisions
    <output>/masks/      <- all matching masks
    <output>/labels.csv   <- filename, label, mask_filename (ready for train_starter.py)
"""

import os
import csv
import shutil
import argparse


def copy_dataset(prepared_dir, out_images, out_masks, start_idx, rows, source_name):
    """Copy one prepared dataset's images/masks into the merged output, renumbering filenames."""
    labels_path = os.path.join(prepared_dir, "labels.csv")
    images_dir = os.path.join(prepared_dir, "images")
    masks_dir = os.path.join(prepared_dir, "masks")

    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        entries = list(reader)

    print(f"Merging {len(entries)} entries from {source_name}...")

    idx = start_idx
    for entry in entries:
        src_img = os.path.join(images_dir, entry["filename"])
        src_mask = os.path.join(masks_dir, entry["mask_filename"])

        if not os.path.exists(src_img) or not os.path.exists(src_mask):
            print(f"  WARNING: missing file for entry {entry}, skipping")
            continue

        out_img_name = f"{idx:06d}.jpg"
        out_mask_name = f"{idx:06d}.png"

        shutil.copy2(src_img, os.path.join(out_images, out_img_name))
        shutil.copy2(src_mask, os.path.join(out_masks, out_mask_name))

        rows.append([out_img_name, entry["label"], out_mask_name])
        idx += 1

    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--casia_prepared", required=True, help="Path to CASIA's prepared/ folder")
    parser.add_argument("--autosplice_prepared", required=True, help="Path to AutoSplice's prepared/ folder")
    parser.add_argument("--output", required=True, help="Where to write the merged dataset")
    args = parser.parse_args()

    out_images = os.path.join(args.output, "images")
    out_masks = os.path.join(args.output, "masks")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    rows = []
    idx = 0
    idx = copy_dataset(args.casia_prepared, out_images, out_masks, idx, rows, "CASIA v2 (Real + Photoshop-tampered)")
    idx = copy_dataset(args.autosplice_prepared, out_images, out_masks, idx, rows, "AutoSplice (Real + AI-tampered)")

    labels_path = os.path.join(args.output, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "mask_filename"])
        writer.writerows(rows)

    # Print class breakdown so you can sanity-check the merge
    from collections import Counter
    label_counts = Counter(r[1] for r in rows)
    print(f"\nDone. {len(rows)} total images merged into: {args.output}")
    print(f"Class breakdown:")
    print(f"  0 (Real):               {label_counts.get('0', 0)}")
    print(f"  1 (Photoshop-tampered): {label_counts.get('1', 0)}")
    print(f"  2 (AI-tampered):         {label_counts.get('2', 0)}")
    print(f"\nThis merged dataset is ready to plug into train_starter.py as DATASET_ROOT.")


if __name__ == "__main__":
    main()
