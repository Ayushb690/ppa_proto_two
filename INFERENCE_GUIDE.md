# PPA Inference Guide

Complete instructions for running programmatic inference, starting the backend API service, launching the web UI, and verifying model predictions.

---

## 1. Quickstart: Launching the Application

The application consists of a Python inference backend and a React 18 web client.

### Step 1: Start the Backend API Server
Open a terminal in the project root:
```bash
# Start backend server on http://127.0.0.1:8000
PYTHONPATH=src uv run python scripts/serve.py
```
*Options:*
- `--port 8000`: Set server port (default: 8000).
- `--host 127.0.0.1`: Set bind address.
- `--checkpoint "Data Files/best.pt"`: Checkpoint path (defaults to `Data Files/best.pt`, automatically resolves `Data_files/best.pt`).
- `--device cuda`: Force device (`cuda` or `cpu`).

### Step 2: Start the Web Client
In a second terminal:
```bash
cd client
pnpm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.
The header will display a green indicator: **`Trained Model Live (best.pt / CUDA)`**, indicating that all uploaded photos and sample images are evaluated by the real trained PPA model.

---

## 2. Programmatic Python Inference

You can run model inference directly in your own scripts using `PPAInferenceEngine`:

```python
from pathlib import Path
from ppa import PPAInferenceEngine

# 1. Initialize engine (loads Data Files/best.pt and precomputes prompt vectors once)
engine = PPAInferenceEngine()

# 2. Score an image (accepts file path, PIL Image, or raw bytes)
result = engine.predict_image("data/PARA/imgs/session15/iaa_pub1000_.jpg")

# 3. Access aesthetic scores
print(f"Predicted Aesthetic Score: {result['rawScore']} / 5.00")
print(f"Normalized Score:          {result['normalizedScore']} ({result['percentile']}th percentile)")
print(f"Aesthetic Tier:            {result['tier']}")
print(f"Inference Latency:         {result['inferenceTimeMs']} ms on {result['device']}")

# 4. Access prompt explainability
print("\nTop Positive Aesthetic Drivers:")
for p in result["topPositivePrompts"][:3]:
    print(f"  + \"{p['posPrompt']}\" (Score: {p['score']}, Weight: {p['selectionWeight']})")

print("\nTop Negative Penalties:")
for p in result["topNegativePrompts"][:3]:
    print(f"  - \"{p['negPrompt']}\" (Score: {p['score']}, Weight: {p['selectionWeight']})")

# 5. Access 11-attribute radar breakdown
print("\nAesthetic Attributes:")
for attr in result["attributeBreakdown"]:
    print(f"  - {attr['name']}: {attr['score']}%")
```

---

## 3. REST API Reference

The backend runs an HTTP REST service with CORS enabled for port 3000:

### 3.1. Health Check
- **Endpoint:** `GET /api/health`
- **Response:**
  ```json
  {
    "online": true,
    "status": "healthy",
    "device": "CUDA:0 (NVIDIA RTX)",
    "checkpoint": "best.pt",
    "best_epoch": 3,
    "val_metrics": {
      "srcc": 0.9154677391052246,
      "plcc": 0.9077672958374023
    }
  }
  ```

### 3.2. Single Image Analysis
- **Endpoint:** `POST /api/analyze`
- **Content-Type:** `multipart/form-data`
- **Body:** `image` file field
- **Example using cURL:**
  ```bash
  curl -X POST http://127.0.0.1:8000/api/analyze \
    -F "image=@data/PARA/imgs/session15/iaa_pub1000_.jpg"
  ```
- **Response Structure:** Returns full `AnalysisResult` JSON containing:
  - `rawScore` ($1.0 - 5.0$)
  - `normalizedScore` ($0.0 - 1.0$)
  - `tier` and `tierColor`
  - `topPositivePrompts` (top 6 positive drivers)
  - `topNegativePrompts` (top 5 negative penalties)
  - `attributeBreakdown` (11 aesthetic attributes for radar chart)
  - `qualityBreakdown` (9 technical attributes)
  - `allPrompts` (all 260 prompt evaluations)
  - `categoryWeights`

### 3.3. Side-by-Side Photo Comparison
- **Endpoint:** `POST /api/compare`
- **Content-Type:** `multipart/form-data`
- **Body:** `imageA` and `imageB` file fields
- **Example using cURL:**
  ```bash
  curl -X POST http://127.0.0.1:8000/api/compare \
    -F "imageA=@photo1.jpg" \
    -F "imageB=@photo2.jpg"
  ```
- **Response:** Returns `{ "resultA": ..., "resultB": ..., "winner": "Photo A", "scoreDiff": 0.42, "isAWinner": true }`.

---

## 4. Running the Verification Suite

To verify the integration, execute the automated 8-check test suite:

```bash
PYTHONPATH=src uv run python scripts/verify_integration.py
```

### Verification Checks Performed:
1. **Checkpoint Loading:** Confirms presence of `Data Files/best.pt`, verifies 792.5 MB size, checks all 6 top-level dictionary keys, and confirms Epoch 3.
2. **Architecture Compatibility:** Confirms `strict=True` loading with 0 missing and 0 unexpected keys, and verifies 28.68M trainable parameters.
3. **Model Initialization:** Validates prompt pool buffer initialization and $L_2$ unit normalization.
4. **Image Preprocessing:** Verifies CLIP $3 \times 224 \times 224$ tensor shape and channel normalization.
5. **Single-Image Inference:** Validates ground-truth tracking, latency calculation, and explainability schemas.
6. **Multiple-Image Validation:** Evaluates batch validation images, asserting high correlation ($>0.70$ threshold; achieves $\approx 0.914$ SRCC).
7. **API Integration:** Validates `/api/analyze` and `/api/compare` request and response contracts.
8. **Frontend Build:** Validates compiled client production assets in `client/dist`.

---

## 5. Hardware & Latency Benchmarks

| Hardware | Warm-up Time (Startup) | Single Image Latency | Throughput |
| :--- | :---: | :---: | :---: |
| **NVIDIA RTX 5060 / 40-series (CUDA)** | ~2.5 s | **18 - 45 ms** | ~35 images/sec |
| **Apple Silicon (MPS)** | ~3.0 s | **45 - 90 ms** | ~14 images/sec |
| **Modern x86 CPU (Intel / AMD)** | ~4.5 s | **350 - 550 ms** | ~2.5 images/sec |

*Note: Model warm-up occurs once at server startup (pre-tokenizing and encoding the 260 prompt embeddings). All subsequent inference requests only execute the ViT-B/16 image encoder and lightweight MLPs.*
