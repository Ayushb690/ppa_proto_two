# Probabilistic Prompt Adaptation (PPA) for Unified Image Aesthetics and Quality Assessment

[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B%2F16-blue.svg)](https://github.com/mlfoundations/open_clip)
[![React](https://img.shields.io/badge/React-18-61dafb.svg?logo=react&logoColor=black)](https://react.dev/)
[![PARA Benchmark](https://img.shields.io/badge/PARA_Benchmark-GIAA-emerald.svg)](https://github.com/yangj1e/PARA)

An implementation and interactive application of **Probabilistic Prompt Adaptation (PPA)** for Unified Image Aesthetics and Quality Assessment, based on the CVPR 2026 paper:

> **Probabilistic Prompt Adaptation for Unified Image Aesthetics and Quality Assessment**  
> *Takayuki Hara and Yuya Otsuka* (CVPR 2026)  
> [Paper Link](https://openaccess.thecvf.com/content/CVPR2026/papers/Hara_Probabilistic_Prompt_Adaptation_for_Unified_Image_Aesthetics_and_Quality_Assessment_CVPR_2026_paper.pdf)

---

## 1. Overview & Architecture

PPA treats Generic Image Aesthetics Assessment (GIAA) not as an opaque black-box regressor, but as a **soft Concept Bottleneck Model (CBM)** over **260 human-interpretable antonym prompt pairs** across 11 aesthetic attributes (e.g. *"Consistent detail photo."* vs. *"Out-of-focus photo."*, *"Vivid colors"* vs. *"Faded colors"*).

```
                             ┌────────────── OpenCLIP (ViT-B/16) ─────────────┐
    Image x ─────────────────►│ Image Encoder:                                 │───┬──────────────┐
                             │   Blocks 0-7: FROZEN                           │   │              │
                             │   Blocks 8-11: Fine-tuned (theta)              │   │ I(x)         │ I(x)
                             └────────────────────────────────────────────────┘   ▼              ▼
    Prompt Pool T_samp                                                           ┌───────────┐  ┌─────────────────────┐
    (260 antonym pairs)                                                          │Score Model│  │ Selection Model     │
    t = (t_pos, t_neg) ──────►┌────────────────────────────────────────────────┐ │p_theta    │  │ p_phi               │◄── Task c
                             │ Text Encoder: FROZEN                           │─►(CLIP-IQA) │  │ (MLPs f_phi1, g_phi2)│
                             │ Precomputed static buffers:                    │ │s̄(x, t)    │  │ p(t | x, c)         │
                             │   pos_emb, neg_emb, text_diff [260, 512]       │ └─────┬─────┘  └──────────┬──────────┘
                             └────────────────────────────────────────────────┘       ▼                   ▼
                                                                                   p(s|x, t)           p(t|x, c)
                                                                                      └─────────┬─────────┘
                                                                                                ▼
                                                                                   Marginalized Prediction:
                                                                                   E[s | x, c] = sum_t s̄ * p
```

---

## 2. Integrated Trained Model Artifacts

The system is powered by the fully trained PARA model checkpoint located in `Data Files/best.pt`:
- **Selected Checkpoint:** `Data Files/best.pt` (Epoch 3)
- **Validation Metrics:** **SRCC: 0.9155**, **PLCC: 0.9078**
- **Test Metrics:** **SRCC: 0.9100**, **PLCC: 0.8982**
- **Parameters:** 28,681,988 trainable ($\theta \approx 28.35\text{M}$ visual backbone blocks 8–11; $\phi \approx 331.9\text{K}$ prompt selection MLPs & task embeddings).
- **Strict Verification:** 100% exact architectural match, `load_state_dict(strict=True)` with 0 missing and 0 unexpected keys.

For complete audit and integration details, see:
- [`MODEL_ARTIFACT_AUDIT.md`](MODEL_ARTIFACT_AUDIT.md)
- [`MODEL_INTEGRATION.md`](MODEL_INTEGRATION.md)
- [`INFERENCE_GUIDE.md`](INFERENCE_GUIDE.md)

---

## 3. Quick Start

### Running Verification Suite
Verify the checkpoint, architecture compatibility, image preprocessing, single-image inference, batch validation correlation, API pipelines, and frontend bindings:
```bash
uv run python scripts/verify_integration.py
```

### Running Standalone Inference
```python
from ppa import PPAInferenceEngine

engine = PPAInferenceEngine()
result = engine.predict_image("data/PARA/imgs/session15/iaa_pub1000_.jpg")

print(f"Predicted Aesthetic Score: {result['rawScore']} / 5.00 ({result['tier']})")
print(f"Top Positive Driver:       {result['topPositivePrompts'][0]['posPrompt']}")
```

### Running the Live Web Application
1. **Launch the API server:**
   ```bash
   PYTHONPATH=src uv run python scripts/serve.py --port 8000
   ```
2. **Launch the React frontend:**
   ```bash
   cd client
   pnpm run dev
   ```
3. Open **[http://localhost:3000](http://localhost:3000)** to analyze any image with live model scoring and prompt explainability.
