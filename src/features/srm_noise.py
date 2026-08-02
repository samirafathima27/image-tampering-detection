"""
SRM (Spatial Rich Model) Noise Residual Extraction
----------------------------------------------------
Catches splicing / copy-move tampering by exposing sensor-noise
inconsistencies. Every camera leaves a subtle, consistent noise
"fingerprint" across an image. Pasted content from a different image
(different camera / different processing) breaks that consistency.

How it works:
We apply a small set of high-pass SRM filter kernels (standard in image
forensics literature) that suppress image content and amplify noise
residuals, then stack the filtered outputs into a multi-channel map.

Usage:
    from srm_noise import generate_srm_residual
    srm_map = generate_srm_residual("sample.jpg")
"""

import numpy as np
import cv2

# Three commonly used SRM high-pass kernels (from Fridrich & Kodovsky,
# and widely reused in tampering-detection papers like RGB-N / ManTra-Net).
SRM_KERNELS = [
    # 1st order horizontal/vertical residual
    np.array([[0, 0, 0],
               [0, -1, 1],
               [0, 0, 0]], dtype=np.float32),

    # 2nd order (Laplacian-like) residual
    np.array([[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]], dtype=np.float32),

    # 3rd order / edge3x3 residual
    np.array([[-1, 2, -1],
               [2, -4, 2],
               [-1, 2, -1]], dtype=np.float32) / 4.0,
]


def generate_srm_residual(image_path: str) -> np.ndarray:
    """
    Generate a stacked SRM noise-residual map for a given image.

    Args:
        image_path: path to the input image

    Returns:
        srm_map: HxWx3 numpy array (uint8) — each channel is one filter's
                 residual response, normalized to 0-255
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    channels = []
    for kernel in SRM_KERNELS:
        filtered = cv2.filter2D(gray, -1, kernel)
        # Normalize each residual map independently to 0-255
        norm = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
        channels.append(norm.astype(np.uint8))

    srm_map = np.stack(channels, axis=-1)  # HxWx3
    return srm_map


def save_srm_visual(image_path: str, out_path: str):
    """Convenience function: generate SRM residual and save as a viewable image."""
    srm_map = generate_srm_residual(image_path)
    cv2.imwrite(out_path, srm_map)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python srm_noise.py <image_path> [output_path]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "srm_output.png"
    save_srm_visual(img_path, out_path)
    print(f"SRM residual map saved to {out_path}")
