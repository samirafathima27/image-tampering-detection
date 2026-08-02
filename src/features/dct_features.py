"""
DCT (Discrete Cosine Transform) Frequency-Domain Analysis
------------------------------------------------------------
Catches AI-generated tampering (diffusion inpainting). Diffusion models
tend to leave subtle, consistent artifacts in the frequency domain that
differ from genuine camera sensor output — this is currently one of the
strongest known signals for detecting AI-generated image regions.

How it works:
We split the image into fixed-size blocks (default 8x8, matching JPEG's
own block structure), run a 2D DCT on each block, and build a spatial
map of high-frequency energy. Unusual high-frequency patterns cluster
around AI-generated / inpainted regions.

Usage:
    from dct_features import generate_dct_map
    dct_map = generate_dct_map("sample.jpg")
"""

import numpy as np
import cv2


def _block_dct_energy(block: np.ndarray) -> float:
    """Compute high-frequency energy of a single 8x8 block via 2D DCT."""
    dct = cv2.dct(block.astype(np.float32))
    # Zero out the low-frequency (top-left) DC/near-DC components,
    # keep only mid/high frequency energy — this is where AI-generation
    # artifacts tend to concentrate.
    dct[:2, :2] = 0
    return float(np.sum(np.abs(dct)))


def generate_dct_map(image_path: str, block_size: int = 8) -> np.ndarray:
    """
    Generate a block-wise high-frequency energy map for a given image.

    Args:
        image_path: path to the input image
        block_size: DCT block size (8 is standard, matches JPEG blocks)

    Returns:
        dct_map: HxW numpy array (uint8), normalized 0-255, same spatial
                 size as the input image (upsampled from block-level map)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Pad so height/width are divisible by block_size
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

    ph, pw = padded.shape
    block_map = np.zeros((ph // block_size, pw // block_size), dtype=np.float32)

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            block = padded[i:i + block_size, j:j + block_size]
            block_map[i // block_size, j // block_size] = _block_dct_energy(block)

    # Upsample block-level map back to full image resolution
    full_map = cv2.resize(block_map, (w, h), interpolation=cv2.INTER_LINEAR)
    norm_map = cv2.normalize(full_map, None, 0, 255, cv2.NORM_MINMAX)
    return norm_map.astype(np.uint8)


def save_dct_visual(image_path: str, out_path: str, block_size: int = 8):
    """Convenience function: generate DCT map and save as a viewable image."""
    dct_map = generate_dct_map(image_path, block_size=block_size)
    cv2.imwrite(out_path, dct_map)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dct_features.py <image_path> [output_path]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "dct_output.png"
    save_dct_visual(img_path, out_path)
    print(f"DCT frequency map saved to {out_path}")
