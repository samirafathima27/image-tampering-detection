"""
CASIA v2 Dataset Pairing & Verification Script
--------------------------------------------------
Matches each tampered image in Tp/ to its ground-truth mask in
CASIA 2 Groundtruth/ (mask filename = image base name + "_gt").

Run this BEFORE training to catch any mismatches early, and to
generate a clean labels.csv your training script can use directly.

Usage:
    python pair_casia.py --casia_root "/path/to/CASIA2"

Expected folder structure:
    CASIA2/
      Au/                     -> authentic images
      Tp/                     -> tampered images
      CASIA 2 Groundtruth/    -> masks, named <image_base>_gt.<ext>
"""

import os
import argparse
import csv


def get_base_name(filename: str) -> str:
    """Strip extension from a filename."""
    return os.path.splitext(filename)[0]


def pair_tp_with_masks(tp_dir: str, gt_dir: str):
    """
    Match each tampered image to its ground-truth mask.

    Returns:
        matched: list of (image_filename, mask_filename)
        unmatched_images: list of image filenames with no mask found
    """
    tp_files = [f for f in os.listdir(tp_dir) if not f.startswith(".")]
    gt_files = [f for f in os.listdir(gt_dir) if not f.startswith(".")]

    # Build a lookup: base_name_without_gt_suffix -> actual gt filename
    gt_lookup = {}
    for gt_file in gt_files:
        base = get_base_name(gt_file)
        if base.endswith("_gt"):
            base = base[: -len("_gt")]
        gt_lookup[base] = gt_file

    matched = []
    unmatched_images = []

    for tp_file in tp_files:
        tp_base = get_base_name(tp_file)
        if tp_base in gt_lookup:
            matched.append((tp_file, gt_lookup[tp_base]))
        else:
            unmatched_images.append(tp_file)

    return matched, unmatched_images


def collect_authentic_images(au_dir: str):
    """List all authentic (real) images — no masks needed for these."""
    return [f for f in os.listdir(au_dir) if not f.startswith(".")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--casia_root", required=True,
                         help="Path to the CASIA2 folder containing Au/, Tp/, and the groundtruth folder")
    parser.add_argument("--gt_folder_name", default="CASIA 2 Groundtruth",
                         help="Name of the groundtruth subfolder (adjust if yours differs)")
    parser.add_argument("--output_csv", default="casia_labels.csv",
                         help="Where to save the final labels.csv")
    args = parser.parse_args()

    au_dir = os.path.join(args.casia_root, "Au")
    tp_dir = os.path.join(args.casia_root, "Tp")
    gt_dir = os.path.join(args.casia_root, args.gt_folder_name)

    for path, name in [(au_dir, "Au"), (tp_dir, "Tp"), (gt_dir, args.gt_folder_name)]:
        if not os.path.isdir(path):
            print(f"ERROR: Could not find folder: {path}")
            return

    print("Scanning folders...")
    au_files = collect_authentic_images(au_dir)
    matched, unmatched = pair_tp_with_masks(tp_dir, gt_dir)

    print(f"\n--- Summary ---")
    print(f"Authentic (Au) images found: {len(au_files)}")
    print(f"Tampered (Tp) images found:  {len(matched) + len(unmatched)}")
    print(f"  -> Successfully matched to a mask: {len(matched)}")
    print(f"  -> NO matching mask found:          {len(unmatched)}")

    if unmatched:
        print(f"\nFirst few unmatched files (these will be SKIPPED):")
        for f in unmatched[:10]:
            print(f"  - {f}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    # Write labels.csv
    # label: 0 = Real, 1 = Photoshop-tampered (we'll add AI-tampered=2 later
    # once you add the AI-generated dataset)
    out_path = os.path.join(args.casia_root, args.output_csv)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "mask_filename", "source_folder"])

        for fname in au_files:
            writer.writerow([fname, 0, "", "Au"])

        for img_fname, mask_fname in matched:
            writer.writerow([img_fname, 1, mask_fname, "Tp"])

    print(f"\nWrote {len(au_files) + len(matched)} total labeled rows to: {out_path}")
    print("Columns: filename, label (0=Real, 1=Photoshop-tampered), mask_filename, source_folder")
    print("\nNext step: run the resize/standardization script before training.")


if __name__ == "__main__":
    main()
