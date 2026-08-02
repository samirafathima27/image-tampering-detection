# AI-Based Image Tampering Detection and Localization System

Detects whether an image is **Real**, **Photoshop-tampered**, or **AI-tampered**,
localizes *where* it was edited (heatmap), and explains *why* in plain English —
combining ELA, SRM noise-residual, and DCT frequency-domain analysis in a single
fused deep learning model.

## Project Structure

```
tampering-detection/
├── data/                    # datasets (not committed to git — see .gitignore)
├── src/
│   ├── features/
│   │   ├── ela.py            # Error Level Analysis (catches Photoshop edits)
│   │   ├── srm_noise.py       # SRM noise residual (catches splicing/copy-move)
│   │   ├── dct_features.py    # DCT frequency analysis (catches AI inpainting)
│   │   └── fusion.py          # combines all three into one 10-channel input
│   └── model/
│       ├── fusion_model.py    # U-Net-style CNN, classification + localization heads
│       └── explainability.py  # rule-based plain-English explanation generator
├── backend/
│   ├── app/main.py            # FastAPI server exposing /check-image endpoints
│   └── requirements.txt
├── extension/                 # Chrome extension (Manifest V3)
│   ├── manifest.json
│   ├── background.js          # right-click context menu logic
│   ├── content.js              # renders the result overlay card on the page
│   ├── overlay.css
│   ├── popup.html / popup.js   # direct-upload popup
├── notebooks/                  # Colab/Kaggle training notebooks go here
└── docs/                        # paper drafts, diagrams, references
```

## Quick Start

### 1. Feature extraction (test individually)
```bash
cd src/features
python3 ela.py <image_path> [output_path]
python3 srm_noise.py <image_path> [output_path]
python3 dct_features.py <image_path> [output_path]
python3 fusion.py <image_path>          # combines all three -> 10-channel array
```

### 2. Model sanity check
```bash
cd src/model
python3 fusion_model.py     # runs a dummy batch through the network, prints shapes
python3 explainability.py    # prints example explanation strings
```

### 3. Run the backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Then test:
```bash
curl -X POST http://127.0.0.1:8000/check-image -F "file=@../data/test_sample.jpg"
```

**Note:** until the model is trained, the backend runs on randomly-initialized
weights (a warning is printed on startup). Predictions will be meaningless
until you train and save a checkpoint to `checkpoints/best_model.pt`.

### 4. Load the Chrome extension
1. Go to `chrome://extensions`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked" → select the `extension/` folder
4. Make sure the backend is running on `http://127.0.0.1:8000`
5. Right-click any image on any webpage → "Check Image Authenticity"

## What's Placeholder vs. Real Right Now

| Component | Status |
|---|---|
| ELA / SRM / DCT feature extraction | ✅ Working, tested on real images |
| Fusion pipeline (10-channel input) | ✅ Working |
| Model architecture (U-Net, 2 heads) | ✅ Working (verified output shapes), **not yet trained** |
| Explainability text generation | ✅ Working, logic is complete |
| Backend API | ✅ Working end-to-end, returns real response shape |
| Chrome extension | ✅ Working, connects to backend |
| **Trained model weights** | ❌ Not yet — this is the team's next big task |
| Datasets (CASIA/Columbia/COVERAGE/AutoSplice/etc.) | ❌ Not yet downloaded/standardized |

## Team Workstreams

1. **Dataset & standardization** — download CASIA v2, Columbia, COVERAGE, AutoSplice/CocoGlide/IMD2020; unify image sizes and mask formats. Save prepared data under `data/`.
2. **Feature extraction** — already scaffolded in `src/features/`. Tune parameters (ELA quality, SRM kernels, DCT block size) against real tampered samples.
3. **Model training** — use `src/model/fusion_model.py` + a Colab/Kaggle notebook (put it in `notebooks/`) to train on the prepared dataset. Save checkpoints to `checkpoints/`.
4. **Explainability + generalization** — wire real per-channel attribution (e.g. Grad-CAM per input channel group) into `explainability.py`; run cross-dataset generalization tests (train on CASIA, test on Columbia/COVERAGE).
5. **Backend + extension** — already scaffolded and tested end-to-end. Update `BACKEND_URL` in `background.js`/`popup.js` once you deploy to Render/Railway.
6. **Paper writing** — base paper reference: FakeShield (arXiv 2410.02761); IEEE journal reference: "Image Manipulation Localization Using Multi-Scale Feature Fusion and Adaptive Edge Supervision" (IEEE Xplore doc 9996125).

## Datasets

| Dataset | Purpose | Link to search for |
|---|---|---|
| CASIA v2 | Real + Photoshop splicing/copy-move | "CASIA v2 dataset download" |
| Columbia, COVERAGE | Cross-dataset generalization testing | "Columbia image splicing dataset", "COVERAGE dataset" |
| AutoSplice, CocoGlide, IMD2020 | AI-generated/diffusion tampering | search each name + "dataset download" |

## Tech Stack

Python, OpenCV, NumPy, PIL, SciPy · PyTorch, torchvision · Grad-CAM · FastAPI, Uvicorn · Chrome Extension (Manifest V3, JS/HTML/CSS) · Google Colab/Kaggle (training) · Render/Railway (hosting)
