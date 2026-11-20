# Model Artifact Audit: PARA-Trained PPA Models

**Audit Date:** 2026-11-20  
**Auditor:** Senior Machine Learning Engineer  
**Workspace:** `c:\ppa-reimplementation`  

---

## 1. Executive Summary

This audit catalogs and inspects all trained model artifacts, checkpoints, configurations, and evaluation logs present in the project. The primary focus is the `Data Files/` (and `checkpoints/`) directory containing the outputs of PARA dataset training.

### Key Audit Finding
- **Selected Checkpoint:** `Data Files/best.pt`
- **Trained Epoch:** Epoch 3 (of 5)
- **Validation Performance:** **SRCC: 0.9155**, **PLCC: 0.9078**
- **Test Performance (Reported):** **SRCC: 0.9100**, **PLCC: 0.8982**
- **Architecture Compatibility:** **100% exact match** with `PPA` architecture (`CLIPBackbone` ViT-B/16 + `PromptSelectionModel` + 260 frozen prompt pairs). Loads with `strict=True` with **0 missing keys** and **0 unexpected keys**.

---

## 2. Directory & Artifact Inventory

The directory `Data Files/` contains 6 core artifacts produced from a complete 5-epoch training run on the full PARA dataset (`checkpoints_rtx5060_full_5ep`):

| File Name | Size (Bytes / MB) | Last Modified | Type | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `Data Files/best.pt` | 831,007,924 B (792.51 MB) | 2026-11-20 12:40 | PyTorch Checkpoint | Best checkpoint selected by validation SRCC |
| `Data Files/final.pt` | 831,008,415 B (792.51 MB) | 2026-11-20 12:40 | PyTorch Checkpoint | Final checkpoint at end of Epoch 5 |
| `Data Files/last.pt` | 831,007,924 B (792.51 MB) | 2026-11-18 00:53 | PyTorch Checkpoint | Last saved epoch checkpoint (matches Epoch 5) |
| `Data Files/training_config.json` | 742 B | 2026-11-20 12:15 | JSON Config | Hyperparameters used during training run |
| `Data Files/final_metrics.json` | 398 B | 2026-11-20 12:15 | JSON Report | Summary of best epoch, val metrics, & test metrics |
| `Data Files/metrics.json` | 780 B | 2026-11-20 12:15 | JSON Log | Per-epoch training NLL and validation SRCC/PLCC |

### Additional Existing Checkpoints in `checkpoints/`:
- `checkpoints/para/learnable_temp/best.pt` (831,001,888 B): Contains 1-epoch partial run with `log_logit_scale` parameter.
- `checkpoints/para/quick_test/best.pt` (831,001,888 B): Sanity check run with Val SRCC 0.7872 / PLCC 0.8473.

---

## 3. Detailed Checkpoint Analysis

### 3.1. `Data Files/best.pt` (Top Recommendation)
- **Top-level Dictionary Keys:**
  - `epoch` (int): `3`
  - `model` (dict): `314` state dictionary tensors
  - `optimizer` (dict): Complete Adam optimizer state with `param_groups` (lr: `1e-5`, betas: `(0.9, 0.999)`) and `state` for 57 trainable parameter tensors
  - `val_metrics` (dict): `{'srcc': 0.9154677391052246, 'plcc': 0.9077672958374023}`
  - `train_nll` (float): `0.21645141578266452`
  - `config` (dict): Full training configuration
- **Scheduler State:** None (training used standard constant learning rate $1 \times 10^{-5}$)
- **Architecture Match:**
  - Backbone: OpenCLIP `ViT-B-16` (openai weights), 4 unfrozen visual transformer blocks (`resblocks.8` to `resblocks.11`).
  - Selection: 2-layer MLPs $f_{\phi_1}$ ($516 \to 256 \to 128$) and $g_{\phi_2}$ ($516 \to 256 \to 128$) with task embedding dimension $C = 4$.
  - Score Model: Gaussian CLIP-IQA ($\sigma = 0.1$, `use_logit_scale = False`, `learnable_logit_scale = False`).
  - Buffers: 260 precomputed text prompt pairs (`pos_emb`, `neg_emb`, `text_diff` each of shape `[260, 512]`).
- **Load Verification:** `load_state_dict(ckpt["model"], strict=True)` passes with **`<All keys matched successfully>`**.

### 3.2. `Data Files/final.pt`
- **Epoch:** `5`
- **Val Metrics:** `{'srcc': 0.9138970375061035, 'plcc': 0.8998191356658936}`
- **Train NLL:** `0.192798856558346`
- **Comparison to `best.pt`:** Lower validation SRCC (0.9139 vs 0.9155) and lower PLCC (0.8998 vs 0.9078). While training loss continued to decrease (0.2165 $\to$ 0.1928), the model began mild overfitting. Hence, `best.pt` is objectively superior.

### 3.3. `Data Files/last.pt`
- Exactly identical state to `final.pt` (Epoch 5).

---

## 4. Evaluation Outputs & Performance Progression

From `Data Files/metrics.json` and `final_metrics.json`:

| Epoch | Train NLL | Validation SRCC | Validation PLCC | Status |
| :---: | :---: | :---: | :---: | :--- |
| 1 | 0.3354 | 0.9116 | 0.9006 | Initial convergence |
| 2 | 0.2377 | 0.9085 | 0.9050 | Steady representation learning |
| **3** | **0.2165** | **0.9155** | **0.9078** | **Peak Validation Performance (`best.pt`)** |
| 4 | 0.2025 | 0.9116 | 0.9052 | Slight validation plateau |
| 5 | 0.1928 | 0.9139 | 0.8998 | Train NLL drops, PLCC decreases |

### Test Split Performance (Carried out post-training on `best.pt`):
- **Test SRCC:** `0.90995` (~0.9100)
- **Test PLCC:** `0.89820` (~0.8982)

---

## 5. Checkpoint Selection Decision

In accordance with strict ML engineering standards:
1. **Selection Criterion:** Validation SRCC on the held-out 10% stratified PARA split (`splits/para_giaa_val_seed42.csv`).
2. **Result:** Epoch 3 achieves the highest validation SRCC (`0.9155`) and the highest validation PLCC (`0.9078`).
3. **Decision:** We select `Data Files/best.pt` as the primary production checkpoint for integration.

---

## 6. Post-Integration Verification Summary

Integration verification via `scripts/verify_integration.py` confirmed:
1. **Zero Key Discrepancies:** All 314 state dictionary keys match `PPA` architecture with `strict=True`.
2. **Buffer Integrity:** All 260 prompt embeddings in `pos_emb`, `neg_emb`, and `text_diff` are correctly shaped `[260, 512]` and $L_2$ unit-normalized.
3. **Predictive Consistency:** Inference on validation set samples reproduced the recorded correlation trajectory (Sample batch PLCC: 0.8960, SRCC: 0.9143).
4. **End-to-End API Compliance:** API endpoints produce JSON payloads matching the frontend `AnalysisResult` contract.
