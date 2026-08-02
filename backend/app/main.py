"""
FastAPI Backend — Image Tampering Detection API
--------------------------------------------------
Serves the trained fusion model. Receives an image (upload or URL from
the Chrome extension), runs it through the feature-fusion + model
pipeline, and returns classification + localization + explanation.

Run locally:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    POST /check-image       -> multipart file upload
    POST /check-image-url    -> JSON { "url": "..." }  (for the Chrome extension)
    GET  /health              -> simple health check
"""

import base64
import io
import os
import sys

import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Make src/ importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

app = FastAPI(title="Image Tampering Detection API", version="0.1.0")

# CORS: allow the Chrome extension (and localhost dev) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your extension ID in production
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Real", "Photoshop-tampered", "AI-tampered"]

# ---------------------------------------------------------------------------
# Model loading — placeholder until your team has a trained checkpoint.
# Once trained, load real weights here, e.g.:
#   model = FusionTamperNet()
#   model.load_state_dict(torch.load("checkpoints/best_model.pt"))
#   model.eval()
# ---------------------------------------------------------------------------
_model = None


def get_model():
    global _model
    if _model is None:
        try:
            import torch
            from model.fusion_model import FusionTamperNet
            _model = FusionTamperNet()
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "checkpoints", "best_model.pt"
            )
            if os.path.exists(checkpoint_path):
                _model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
                print("Loaded trained checkpoint.")
            else:
                print("WARNING: No trained checkpoint found — using untrained model "
                      "(random weights). Replace with real checkpoint before demo.")
            _model.eval()
        except Exception as e:
            print(f"Model load failed: {e}")
    return _model


class ImageURLRequest(BaseModel):
    url: str


class PredictionResponse(BaseModel):
    verdict: str
    confidence: float
    explanation: str
    heatmap_base64: str  # base64-encoded PNG of the tampered-region heatmap


def run_pipeline(image: Image.Image) -> PredictionResponse:
    """
    Full pipeline: fused features -> model -> classification + mask -> explanation.

    NOTE: this currently returns a structurally-correct but placeholder
    response so the rest of the team (extension, frontend) can integrate
    against the real API shape immediately. Swap in real inference once
    the model is trained (see build_fused_input + FusionTamperNet).
    """
    from features.fusion import build_fused_input
    import torch

    tmp_path = "_tmp_input.jpg"
    image.convert("RGB").save(tmp_path, quality=90)

    fused = build_fused_input(tmp_path)  # HxWx10
    os.remove(tmp_path)

    model = get_model()

    tensor = torch.from_numpy(fused).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        class_logits, mask_logits = model(tensor)
        probs = torch.softmax(class_logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        confidence = float(probs[pred_idx])
        mask = torch.sigmoid(mask_logits)[0, 0].numpy()

    verdict = CLASS_NAMES[pred_idx]

    # Build explanation (placeholder channel scores until real per-channel
    # attribution is wired in — e.g. via Grad-CAM per input channel group)
    from model.explainability import explain_prediction, ChannelScores
    scores = ChannelScores(ela_score=0.3, srm_score=0.3, dct_score=0.4)
    explanation = explain_prediction(verdict, confidence, scores)

    # Encode heatmap (the mask) as base64 PNG
    heatmap_img = Image.fromarray((mask * 255).astype(np.uint8))
    buf = io.BytesIO()
    heatmap_img.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return PredictionResponse(
        verdict=verdict,
        confidence=round(confidence, 4),
        explanation=explanation,
        heatmap_base64=heatmap_b64,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check-image", response_model=PredictionResponse)
async def check_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return run_pipeline(image)


@app.post("/check-image-url", response_model=PredictionResponse)
def check_image_url(req: ImageURLRequest):
    try:
        resp = requests.get(req.url, timeout=10)
        image = Image.open(io.BytesIO(resp.content))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not fetch image from URL")

    return run_pipeline(image)
