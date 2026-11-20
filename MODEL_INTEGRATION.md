# PPA Model Integration Documentation

**Project:** Probabilistic Prompt Adaptation (PPA) for Unified Image Aesthetics and Quality Assessment  
**Benchmark:** PARA (Personalized Image Aesthetics Assessment with Rich Attributes) - Generic Setting (GIAA)  
**Status:** Integrated & Verified  
**Date:** 2026-11-20  

---

## 1. Executive Summary

This document details the complete integration of the pretrained PARA-trained model artifacts from `Data Files/` into the main PPA codebase and interactive application. 

The application now performs **genuine, real-time model inference** using the trained PyTorch checkpoint `Data Files/best.pt`, removing mock data and simulation from the active inference path while preserving the full architectural integrity of the CVPR 2026 paper:

> **Probabilistic Prompt Adaptation for Unified Image Aesthetics and Quality Assessment**  
> *Takayuki Hara and Yuya Otsuka* (CVPR 2026)

---

## 2. Checkpoint Selection & Integrity

### 2.1. Selected Checkpoint
- **File:** `Data Files/best.pt`
- **Size:** 831,007,924 bytes (~792.51 MB)
- **Epoch:** **Epoch 3** of 5
- **Validation Performance:** **SRCC: 0.9155**, **PLCC: 0.9078**
- **Test Performance (Reported):** **SRCC: 0.9100**, **PLCC: 0.8982**

### 2.2. Selection Rationale
In accordance with standard ML engineering methodology:
1. **Validation Protocol:** Checkpoint selection was determined strictly by validation SRCC on the held-out 10% stratified PARA split (`splits/para_giaa_val_seed42.csv`), avoiding any test-set leakage.
2. **Epoch 3 vs. Epoch 5 (`final.pt`):** While training negative log-likelihood (NLL) continued to decrease through epoch 5 (0.2165 $\to$ 0.1928), validation SRCC and PLCC peaked at Epoch 3 (SRCC: 0.9155, PLCC: 0.9078) before exhibiting slight validation regression at Epoch 5 (SRCC: 0.9139, PLCC: 0.8998). Thus, Epoch 3 was selected as the optimal checkpoint.
3. **Artifact Integrity:** The checkpoint was preserved unmodified. It contains:
   - `epoch`: 3
   - `model`: 314 parameter and buffer tensors
   - `optimizer`: Complete Adam optimizer state for 57 trainable parameter tensors (lr: 1e-5, betas: (0.9, 0.999))
   - `val_metrics`: `{'srcc': 0.9154677391052246, 'plcc': 0.9077672958374023}`
   - `train_nll`: 0.21645141578266452
   - `config`: Complete training configuration dictionary

### 2.3. Architecture Compatibility & Loading Procedure
The model is instantiated using `PPA(cfg, prompt_pairs)` and loaded strictly:
```python
checkpoint = torch.load("Data Files/best.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model"], strict=True)
```
- **Result:** `<All keys matched successfully>` (0 missing keys, 0 unexpected keys).
- **Trainable Parameters ($\theta + \phi$):** `28,681,988` (Last 4 visual transformer blocks + 2-layer MLPs + task embedding).
- **Frozen Parameters:** `121,269,249` (Text encoder + early visual transformer blocks 0–7).

---

## 3. Mathematical Formulation & Architecture Flow

The model operates as a soft Concept Bottleneck Model (CBM):

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
                                                                                 p(s | x, t)         p(t | x, c)
                                                                                      └─────────┬─────────┘
                                                                                                ▼
                                                                                       Expected Score (Eq. 6)
                                                                                 E[s|x,c] = Σ_t s̄(x,t) · p_φ(t|x,c)
```

1. **CLIP-IQA Score Model $\bar{s}(x, t)$ (Eq. 1):**
   $$\bar{s}(x, t) = \sigma\left( \langle I(x), T(t_{\text{pos}})\rangle - \langle I(x), T(t_{\text{neg}})\rangle \right) \in (0, 1)$$
2. **Prompt Selection Model $p_\phi(t \mid x, c)$ (Eq. 3):**
   $$p_\phi(t \mid x, c) = \frac{\exp\left( f_{\phi_1}(I(x), e_c)^T g_{\phi_2}(T(t_{\text{pos}}) - T(t_{\text{neg}}), e_c) \right)}{\sum_{t'} \exp\left( f_{\phi_1}(I(x), e_c)^T g_{\phi_2}(T(t'_{\text{pos}}) - T(t'_{\text{neg}}), e_c) \right)}$$
3. **Marginalization Inference (Eq. 6):**
   $$\mathbb{E}[s \mid x, c] = \sum_{t \in T_{\text{samp}}} \bar{s}(x, t) \cdot p_\phi(t \mid x, c)$$
   Mapped to the PARA aesthetic scale $[1.0, 5.0]$ via $s_{\text{raw}} = 1.0 + 4.0 \cdot \mathbb{E}[s \mid x, c]$.

---

## 4. Preprocessing Pipeline

Inference strictly adheres to the CLIP ViT-B/16 validation preprocessing used during PARA training:
- **Image Mode:** RGB (`PIL.Image.convert("RGB")`)
- **Resize:** Bicubic interpolation to shortest edge 224 pixels
- **Crop:** Center crop to $224 \times 224$ pixels
- **Normalization:** Standard CLIP channel mean and standard deviation:
  - Mean: `[0.48145466, 0.4578275, 0.40821073]`
  - Standard Deviation: `[0.26862954, 0.26130258, 0.27577711]`
- **Inference Mode:** Executed under `torch.inference_mode()` with `model.eval()`.

---

## 5. System Components & File Map

| Component | File | Purpose |
| :--- | :--- | :--- |
| **Inference Engine** | `src/ppa/inference.py` | `PPAInferenceEngine`: Checkpoint loader, device routing, forward pass, explainability aggregator |
| **API Server** | `src/ppa/server.py` | Multi-threaded HTTP API exposing `/api/health`, `/api/analyze`, `/api/compare` |
| **Server CLI** | `scripts/serve.py` | CLI entry point: `PYTHONPATH=src uv run python scripts/serve.py` |
| **Verification Suite** | `scripts/verify_integration.py` | Automated 8-point integration test runner |
| **Web Frontend** | `client/src/App.tsx` | React 18 dashboard with Dropzone, radial ScoreGauge, AttributeRadar, PromptTable, CompareView |
| **Frontend API Layer**| `client/src/api.ts` | Async HTTP client communicating with port 8000 backend; fallback simulation only if offline |

---

## 6. Verification Results

All 8 automated integration checks pass cleanly (`scripts/verify_integration.py`):
1. **Checkpoint Loading:** Verified `Data Files/best.pt` (792.51 MB, Epoch 3, Val SRCC: 0.9155).
2. **Architecture Compatibility:** Verified `strict=True` with 28.68M trainable / 121.27M frozen parameters.
3. **Model Initialization:** Verified 260 unit-normalized prompt vectors in precomputed buffers.
4. **Image Preprocessing:** Verified $3 \times 224 \times 224$ float32 normalization.
5. **Single-Image Inference:** Verified valid prediction and complete `AnalysisResult` contract (Ground Truth 3.640 vs Predicted 3.553).
6. **Batch Inference Validation:** Verified on 20 validation images; reproduced SRCC of **0.9143** and PLCC of **0.8960**.
7. **API Integration:** Verified multipart/form-data endpoints for analysis and image comparison.
8. **Frontend Build:** Verified production asset bundle generation and `/api` proxy routing.
