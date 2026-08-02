"""
Multi-Modal Fusion Pipeline
-----------------------------
Combines RGB + ELA + SRM + DCT into a single multi-channel tensor that
gets fed into the model. This is the core "fusion" step of the project.

Channels produced (9 total):
  0-2: original RGB
  3-5: ELA map (3 channels)
  6-8: SRM noise residual (3 channels)
  9  : DCT frequency energy map (1 channel, appended)

Usage:
    from fusion import build_fused_input
    fused = build_fused_input("sample.jpg")   # -> HxWx10 numpy array
"""

import numpy as np
import cv2

try:
    from .ela import generate_ela
    from .srm_noise import generate_srm_residual
    from .dct_features import generate_dct_map
except ImportError:
    # Fallback for running this file directly (python fusion.py) rather
    # than as part of the `features` package.
    from ela import generate_ela
    from srm_noise import generate_srm_residual
    from dct_features import generate_dct_map


def build_fused_input(image_path: str, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Build the full multi-modal fused input for a single image.

    Args:
        image_path: path to input image
        target_size: (H, W) to resize all maps to before stacking

    Returns:
        fused: HxWx10 numpy array (uint8)
            channels 0-2 = RGB, 3-5 = ELA, 6-8 = SRM, 9 = DCT
    """
    rgb = cv2.imread(image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, target_size)

    ela = generate_ela(image_path)
    ela = cv2.resize(ela, target_size)

    srm = generate_srm_residual(image_path)
    srm = cv2.resize(srm, target_size)

    dct = generate_dct_map(image_path)
    dct = cv2.resize(dct, target_size)
    dct = dct[..., np.newaxis]  # HxWx1

    fused = np.concatenate([rgb, ela, srm, dct], axis=-1)  # HxWx10
    return fused


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fusion.py <image_path>")
        sys.exit(1)

    fused = build_fused_input(sys.argv[1])
    print(f"Fused input shape: {fused.shape}  (expected HxWx10)")
    print(f"dtype: {fused.dtype}, min: {fused.min()}, max: {fused.max()}")
