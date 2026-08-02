"""
Explainability Layer
----------------------
Not a separate neural network — a rule-based logic layer that converts
raw model outputs (per-channel anomaly scores + mask region stats) into
a plain-English explanation of WHY the model flagged an image.

This works by looking at which of the three feature channels (ELA, SRM,
DCT) contributed most strongly to the masked/flagged region, and mapping
that to a human-readable reason.

Usage:
    from explainability import explain_prediction
    text = explain_prediction(class_label="AI-tampered",
                               confidence=0.91,
                               ela_score=0.22, srm_score=0.18, dct_score=0.81,
                               region_desc="top-right region")
"""

from dataclasses import dataclass


@dataclass
class ChannelScores:
    ela_score: float   # 0-1, how much the ELA channel contributed to the flagged region
    srm_score: float   # 0-1, how much the SRM noise channel contributed
    dct_score: float   # 0-1, how much the DCT frequency channel contributed


def _dominant_channel(scores: ChannelScores) -> str:
    values = {
        "ela": scores.ela_score,
        "srm": scores.srm_score,
        "dct": scores.dct_score,
    }
    return max(values, key=values.get)


def _region_label_from_bbox(bbox, image_size):
    """
    Convert a bounding box (x_min, y_min, x_max, y_max) into a human
    readable region label like 'top-right region'.
    """
    W, H = image_size
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    horiz = "left" if cx < W / 3 else ("right" if cx > 2 * W / 3 else "center")
    vert = "top" if cy < H / 3 else ("bottom" if cy > 2 * H / 3 else "middle")

    if horiz == "center" and vert == "middle":
        return "central region"
    if horiz == "center":
        return f"{vert} region"
    if vert == "middle":
        return f"{horiz} region"
    return f"{vert}-{horiz} region"


CHANNEL_REASON_TEMPLATES = {
    "ela": "compression-level inconsistency (typical of Photoshop-style edits)",
    "srm": "sensor-noise inconsistency (typical of splicing or copy-move edits)",
    "dct": "frequency-domain anomaly (typical of AI-generated/diffusion inpainting)",
}


def explain_prediction(class_label: str, confidence: float, scores: ChannelScores,
                        bbox=None, image_size=None) -> str:
    """
    Build a plain-English explanation string.

    Args:
        class_label: "Real" | "Photoshop-tampered" | "AI-tampered"
        confidence: model confidence, 0-1
        scores: ChannelScores dataclass with per-channel contribution scores
        bbox: optional (x_min, y_min, x_max, y_max) of the flagged region
        image_size: optional (W, H), required if bbox is given

    Returns:
        A human-readable explanation string.
    """
    if class_label == "Real":
        return f"No tampering indicators detected (confidence {confidence*100:.1f}%)."

    dominant = _dominant_channel(scores)
    reason = CHANNEL_REASON_TEMPLATES[dominant]

    region_str = ""
    if bbox is not None and image_size is not None:
        region_str = f" in the {_region_label_from_bbox(bbox, image_size)}"

    return (
        f"Flagged as {class_label} (confidence {confidence*100:.1f}%) due to "
        f"{reason}{region_str}."
    )


if __name__ == "__main__":
    # Example usage
    scores = ChannelScores(ela_score=0.15, srm_score=0.20, dct_score=0.85)
    text = explain_prediction(
        class_label="AI-tampered",
        confidence=0.91,
        scores=scores,
        bbox=(180, 10, 250, 90),
        image_size=(256, 256),
    )
    print(text)

    scores2 = ChannelScores(ela_score=0.78, srm_score=0.30, dct_score=0.10)
    text2 = explain_prediction(
        class_label="Photoshop-tampered",
        confidence=0.87,
        scores=scores2,
        bbox=(10, 150, 100, 240),
        image_size=(256, 256),
    )
    print(text2)
