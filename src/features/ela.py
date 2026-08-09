"""
Error Level Analysis (ELA)
---------------------------
Catches traditional (Photoshop-style) tampering by exposing inconsistent
JPEG compression levels between authentic and pasted/edited regions.

How it works:
1. Save the image at a known JPEG quality (e.g. 90).
2. Compare it pixel-by-pixel against the original.
3. Regions that were edited after the original compression will show a
   different (usually higher) error level than untouched regions.

Usage:
    from ela import generate_ela
    ela_map = generate_ela("sample.jpg")
"""

import os
import uuid
import numpy as np
from PIL import Image, ImageChops


def generate_ela(image_path: str, quality: int = 90, scale: int = 15) -> np.ndarray:
    """
    Generate an ELA map for a given image.

    Args:
        image_path: path to the input image
        quality: JPEG re-save quality (90 is the common default in ELA literature)
        scale: brightness amplification factor so differences are visible/learnable

    Returns:
        ela_map: HxWx3 numpy array (uint8), same spatial size as input image
    """
    original = Image.open(image_path).convert("RGB")

    # Use a unique temp filename per call (pid + uuid) so this is safe to call
    # from multiple parallel DataLoader worker processes at once — a shared
    # hardcoded filename causes race conditions (one worker deletes the file
    # while another is still writing/reading it).
    tmp_path = f"_ela_tmp_{os.getpid()}_{uuid.uuid4().hex}.jpg"
    original.save(tmp_path, "JPEG", quality=quality)
    try:
        resaved = Image.open(tmp_path)
        diff = ImageChops.difference(original, resaved)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    diff_arr = np.array(diff).astype(np.float32)

    # Amplify the (usually very small) pixel differences so they're visible
    # and so the model has a stronger learnable signal.
    max_diff = diff_arr.max()
    if max_diff == 0:
        max_diff = 1  # avoid divide-by-zero on a perfectly flat image
    amplify = min(255.0 / max_diff, scale)

    ela_arr = np.clip(diff_arr * amplify, 0, 255).astype(np.uint8)
    return ela_arr


def save_ela_visual(image_path: str, out_path: str, quality: int = 90, scale: int = 15):
    """Convenience function: generate ELA and save it as a viewable image."""
    ela_arr = generate_ela(image_path, quality=quality, scale=scale)
    Image.fromarray(ela_arr).save(out_path)


if __name__ == "__main__":
    # Quick manual test — replace with a real sample image path
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ela.py <image_path> [output_path]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "ela_output.png"
    save_ela_visual(img_path, out_path)
    print(f"ELA map saved to {out_path}")