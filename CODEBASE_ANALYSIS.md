# Codebase Analysis: Probabilistic Prompt Adaptation (PPA) Reimplementation

## 1. Executive Summary

This repository is an unofficial, from-scratch PyTorch reimplementation of **Probabilistic Prompt Adaptation (PPA)** for Unified Image Aesthetics and Quality Assessment, based on the CVPR 2026 paper:

> **Probabilistic Prompt Adaptation for Unified Image Aesthetics and Quality Assessment**  
> *Takayuki Hara and Yuya Otsuka* (CVPR 2026)  
> [Paper Link](https://openaccess.thecvf.com/content/CVPR2026/papers/Hara_Probabilistic_Prompt_Adaptation_for_Unified_Image_Aesthetics_and_Quality_Assessment_CVPR_2026_paper.pdf)

### Core Objective
The system models image aesthetics and image quality assessment (IAA / IQA) not as a black-box regression, but as a **probabilistic mixture over human-interpretable antonym text prompt pairs** (e.g., *"Sharp photo."* vs. *"Blurry photo."*, *"Vivid colors"* vs. *"Faded colors"*). It acts as a **soft Concept Bottleneck Model (CBM)** where:
1. Prompts serve as interpretable concepts.
2. Per-prompt CLIP-IQA scores act as concept activations.
3. Prompt weights represent conditioned concept importance for a specific image and task.

The reproduction targets the **PARA** (Personalized Image Aesthetics Assessment with Rich Attributes) benchmark under the Generic Image Aesthetics Assessment (GIAA) setting.

---

## 2. Key Results & Reproduction Achievements

The paper reports evaluation on the PARA test split using two standard ranking and correlation metrics:
- **SRCC** (Spearman Rank Correlation Coefficient)
- **PLCC** (Pearson Linear Correlation Coefficient)

| Configuration | Selected Epoch | Val SRCC | Val PLCC | Test SRCC | Test PLCC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PPA (Original Paper, Table 4)** | — | — | — | 0.9130 | 0.9420 | Published Reference |
| **This Repo: Paper-Literal** | 18 | 0.9191 | 0.9124 | 0.9160 | 0.9073 | Replicated faithfully |
| **This Repo: `+ learnable temperature`** | **19** | **0.9217** | **0.9502** | **0.9164** | **0.9456** | **Exceeds original paper** |

### Key Experimental Discoveries in this Implementation:
1. **The PLCC Gap & Learnable Temperature**:
   - The paper-literal formulation omits temperature scaling on the score logit $\bar{s}(x, t)$, squashing raw cosine differences through sigmoid into a compressed range $[0.35, 0.65]$ (prediction standard deviation $\approx 0.085$ vs ground truth $\approx 0.133$).
   - This harms linear correlation (PLCC $\approx 0.9073$) while preserving relative ordering (SRCC $\approx 0.9160$).
   - Adding a single learnable temperature $\tau$ (`--learnable-temp`, initialized to 10.0) allows the model to learn the score distribution width, lifting Test PLCC to **0.9456** and Test SRCC to **0.9164**, outperforming the published baseline.
2. **Proper Validation Protocol without Test-Set Leakage**:
   - Standard PARA releases only `PARA-GiaaTrain.csv` (28,220 images) and `PARA-GiaaTest.csv` (3,000 images). Previous practices selected checkpoints on the test split, causing information leakage.
   - This repo introduces a strict 10% stratified validation split (2,822 images, 7 score bins, seed 42) carved out of the training set. Model checkpointing (`best.pt`) is strictly based on validation SRCC, and the test split is evaluated only once upon training completion.
3. **Flat Validation Trajectory**:
   - Across epochs 2 through 20, validation SRCC remains within a narrow band of 0.0062. Epoch 19 beats epoch 3 by less than 0.0001, demonstrating that the model learns the core aesthetic representation within the first 2–3 epochs; further training reduces train NLL without providing significant generalization gains.

---

## 3. Mathematical Formulation & Architecture

The architecture decomposes aesthetic assessment into two pathways: a **Score Model** and a **Prompt Selection Model**, integrated via probabilistic marginalization.

```
                            ┌──────────── OpenCLIP (ViT-B/16) ────────────┐
  Image x ─────────────────►│ Image Encoder:                              │───┬───────────────┐
                            │   Early blocks (0-7): FROZEN                │   │               │
                            │   Last 4 blocks (8-11): TRAINABLE (theta)   │   │ I(x)          │ I(x)
                            └─────────────────────────────────────────────┘   ▼               ▼
  Prompt Pool T_samp                                                         ┌────────────┐ ┌──────────────────────┐
  (260 antonym pairs)                                                        │ Score Path │ │ Selection Path       │
  t = (t_pos, t_neg) ──────►┌─────────────────────────────────────────────┐  │ p_theta    │ │ p_phi                │◄── Task c
                            │ Text Encoder: FULLY FROZEN                  │─►│ Score s̄    │ │ Prompt Selection     │
                            │ Embeddings precomputed as static buffers    │  │ (Eq. 1)    │ │ (Eq. 3, trainable)   │
                            └─────────────────────────────────────────────┘  └─────┬──────┘ └──────────┬───────────┘
                                                                                   ▼                   ▼
                                                                              p(s | x, t)         p(t | x, c)
                                                                                   └─────────┬─────────┘
                                                                                             ▼
                                                                                     Marginalization:
                                                                                p(s | x, c) = Σ_t p_θ · p_φ  (Eq. 2)
                                                                                             ▼
                                                                     Train: Mixture NLL (Eq. 5) | Infer: E[s|x,c] (Eq. 6)
```

### 3.1. Prompt Pool ($T_{\text{samp}}$)
- 260 curated antonym pairs ($t_{\text{pos}}, t_{\text{neg}}$) transcribed from supplementary tables S6–S9:
  - **Aesthetic Attributes (S6)**: 11 attributes $\times$ 4 pairs = 44 pairs (e.g. *Good Lighting*, *Color Harmony*, *Rule of Thirds*).
  - **Quality Attributes (S7)**: 9 attributes $\times$ 4 pairs = 36 pairs (e.g. *Blur*, *Color-related*, *Noise*).
  - **Aesthetic General (S8)**: 100 attribute-agnostic pairs.
  - **Quality General (S9)**: 80 attribute-agnostic pairs.

### 3.2. Score Model $p_\theta(s \mid x, t)$ (Eq. 1 & Eq. 4)
Computes the CLIP-IQA aesthetic score $\bar{s}(x, t)$ for image $x$ under prompt pair $t$:
$$\bar{s}(x, t) = \sigma\left( \tau \cdot \left(\langle I(x), T(t_{\text{pos}})\rangle - \langle I(x), T(t_{\text{neg}})\rangle\right) \right)$$
- $I(x) \in \mathbb{R}^D$: $L_2$-normalized image feature from CLIP ViT-B/16 ($D = 512$).
- $T(t) \in \mathbb{R}^D$: $L_2$-normalized prompt embeddings.
- $\tau$: Logit scale (1.0 in literal paper; learnable parameter in `--learnable-temp`).
- The score model assumes a Gaussian distribution around $\bar{s}$ with fixed standard deviation $\sigma = 0.1$:
  $$p_\theta(s \mid x, t) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(s - \bar{s}(x, t))^2}{2\sigma^2} \right)$$
- **Parameters $\theta$**: The score path has no standalone dense layers; $\theta$ consists solely of the weights in the **last 4 transformer blocks of the CLIP visual encoder**.

### 3.3. Prompt Selection Model $p_\phi(t \mid x, c)$ (Eq. 3)
Calculates the conditional probability that prompt pair $t$ governs the score for image $x$ on task $c$:
$$p_\phi(t \mid x, c) = \frac{\exp\left( f_{\phi_1}(I(x), e_c)^T g_{\phi_2}(T(t_{\text{pos}}) - T(t_{\text{neg}}), e_c) \right)}{\sum_{t'} \exp\left( f_{\phi_1}(I(x), e_c)^T g_{\phi_2}(T(t'_{\text{pos}}) - T(t'_{\text{neg}}), e_c) \right)}$$
- $e_c \in \mathbb{R}^C$: Learnable task embedding ($C = 4$).
- $f_{\phi_1}$: 2-layer MLP $[I(x); e_c] \to \mathbb{R}^{256} \to \mathbb{R}^{128}$.
- $g_{\phi_2}$: 2-layer MLP $[T(t_{\text{pos}}) - T(t_{\text{neg}}); e_c] \to \mathbb{R}^{256} \to \mathbb{R}^{128}$.

### 3.4. Marginalization & Optimization (Eq. 2 & Eq. 5)
The overall score distribution is a Gaussian Mixture:
$$p(s \mid x, c) = \sum_{t \in T_{\text{samp}}} p_\theta(s \mid x, t) \cdot p_\phi(t \mid x, c)$$

Trained by minimizing the Negative Log-Likelihood (NLL):
$$\mathcal{L}_{\text{NLL}} = -\frac{1}{B} \sum_{i=1}^B \log \sum_{t} \exp\left(-\frac{(s_i - \bar{s}(x_i, t))^2}{2\sigma^2}\right) p_\phi(t \mid x_i, c_i)$$
- Numerically evaluated using `torch.logsumexp`.
- **Prompt Dropout**: During training, 50% of the prompt pool is randomly dropped per iteration (Supplementary Sec. B), forcing the network to distribute weights robustly across varied concepts.

### 3.5. Inference (Eq. 6)
At test time, the final predicted score is the expected value across the full 260 prompt pool:
$$\mathbb{E}[s \mid x, c] = \sum_{t \in T_{\text{samp}}} \bar{s}(x, t) \cdot p_\phi(t \mid x, c)$$

---

## 4. Parameter and Training Budget

| Component | Status | Parameter Symbol | Parameter Count |
| :--- | :---: | :---: | :---: |
| CLIP Text Encoder | Frozen | — | 0 (cached offline) |
| CLIP Image Encoder (early blocks 0–7) | Frozen | — | 0 |
| CLIP Image Encoder (last 4 resblocks) | **Trainable** | $\theta$ | ~28,350,000 |
| Selection MLPs ($f_{\phi_1}, g_{\phi_2}$) + Task Embeddings | **Trainable** | $\phi$ | ~331,908 |
| Learnable Temperature $\tau$ (if enabled) | **Trainable** | $\tau$ | 1 |
| **Total Trainable Parameters** | | | **~28,681,989** |

---

## 5. Repository File Map

```
ppa-reimplementation/
├── pyproject.toml              # Project dependencies, build config (hatchling, uv, ruff)
├── README.md                   # Project overview, quickstart, reproduction benchmarks
├── NOTICE                      # Copyright notice and academic paper citation
├── LICENSE                     # Apache 2.0 license
├── docs/
│   └── para_val_reeval.md      # Detailed re-evaluation protocol, validation split design & curves
├── scripts/
│   ├── build_para_val_split.py # Generates / asserts the 10% stratified validation split
│   ├── smoke_test.py           # Standalone shape check on random tensors (runs on CPU, no data needed)
│   └── train.py                # Command-line entry point for training and evaluation
├── splits/
│   └── para_giaa_val_seed42.csv# Committed validation set (2,822 image filenames & bin assignments)
├── src/
│   └── ppa/
│       ├── __init__.py         # Package export definitions
│       ├── clip_backbone.py    # OpenCLIP ViT-B/16 wrapper; manages freezing and partial unfreezing
│       ├── config.py           # Strongly-typed frozen dataclasses for all hyperparameters
│       ├── data.py             # PARA GIAA PyTorch Dataset & DataLoader builder
│       ├── model.py            # PPA unified model: marginalization, NLL loss, prediction
│       ├── prompts.py          # 260 antonym prompt pairs transcribed verbatim from paper supplement
│       ├── README.md           # Technical documentation of architecture, dimensions, equations
│       ├── score_model.py      # CLIP-IQA parameter-free score calculation (Eq. 1)
│       ├── selection_model.py  # Prompt selection network p_phi (Eq. 3)
│       └── train.py            # Adam training loop, checkpointing, custom pure-PyTorch SRCC/PLCC
├── checkpoints/                # Saved weights and metrics (git-ignored, local runs)
│   └── para/
│       ├── learnable_temp/     # Checkpoint from 20-epoch run with learnable temperature
│       │   ├── best.pt
│       │   └── last.pt
│       └── quick_test/         # Rapid sanity check artifacts
│           ├── best.pt
│           ├── last.pt
│           └── final_metrics.json
└── data/
    └── PARA/                   # PARA dataset root
        ├── annotation/         # Official CSV annotations (PARA-GiaaTrain.csv, PARA-GiaaTest.csv, etc.)
        └── imgs/               # Image sessions (session1 to session145)
```

---

## 6. Detailed Walkthrough of Key Modules

### `src/ppa/config.py`
Centralizes all experiment parameters into frozen Python dataclasses with citations to specific sections in the original paper:
- `CLIPConfig`: ViT-B-16, `openai` pretrained weights, unfreezing the last 4 visual transformer blocks, embedding dim 512.
- `SelectionConfig`: Task embedding dim $C = 4$, MLP hidden dim 256, projection dim 128, $N_{\text{tasks}} = 1$.
- `ScoreConfig`: Gaussian standard deviation $\sigma = 0.1$, flags for `use_logit_scale` and `learnable_logit_scale` (init 10.0).
- `TrainConfig`: Adam optimizer ($\text{lr} = 10^{-5}$, $\beta = (0.9, 0.999)$), batch size 32, 20 epochs, prompt dropout 0.5, score range $[1.0, 5.0]$ normalized to $[0.0, 1.0]$.
- `resolve_device()`: Automatically selects device hierarchy (`mps` $\to$ `cuda` $\to$ `cpu`).

### `src/ppa/clip_backbone.py`
- Instantiates OpenCLIP ViT-B/16 model and validation preprocessing pipeline.
- Locks `requires_grad = False` across all text encoder weights and early visual transformer blocks.
- Unfreezes only the last $N$ blocks of `model.visual.transformer.resblocks` ($\theta$).
- `encode_text` pre-tokenizes and produces $L_2$-normalized prompt embeddings.
- `encode_image` projects input images to $L_2$-normalized 512-dimensional vectors while retaining gradient flow to $\theta$.

### `src/ppa/score_model.py`
- Implements `clipiqa_score(image_emb, pos_emb, neg_emb, logit_scale)`.
- Computes matrix inner products with positive and negative prompts:
  $$\text{sim\_pos} = I \cdot T_{\text{pos}}^T, \quad \text{sim\_neg} = I \cdot T_{\text{neg}}^T$$
- Returns $\sigma\left(\text{logit\_scale} \cdot (\text{sim\_pos} - \text{sim\_neg})\right) \in (0, 1)$.

### `src/ppa/selection_model.py`
- Evaluates the attention-like distribution over prompt pairs.
- Concatenates the image vector with task embedding: $[I(x); e_c] \in \mathbb{R}^{512 + 4}$.
- Concatenates prompt difference vector with task embedding: $[T(t_{\text{pos}}) - T(t_{\text{neg}}); e_c] \in \mathbb{R}^{512 + 4}$.
- Projects both through respective 2-layer MLPs into a shared 128-dimensional latent space.
- Calculates prompt logits via pairwise dot product (`torch.einsum("bd,bpd->bp", f, g)`) followed by softmax over prompts.

### `src/ppa/model.py`
- Encapsulates backbone, selection model, precomputed prompt buffers (`pos_emb`, `neg_emb`, `text_diff`), and optional learnable logit scale (`log_logit_scale`).
- Manages `_dropout_idx`: dynamically samples $(1 - \text{rate}) \times P$ prompt indices during training to apply prompt dropout.
- Implements `predict(images, task_ids)`: computes full expectation across all 260 prompts.
- Implements `nll_loss(images, task_ids, target)`: calculates Gaussian mixture likelihood in log-space using `torch.logsumexp`.

### `src/ppa/data.py`
- `ParaGiaaDataset`: Loads images from `data/PARA/imgs/<sessionId>/<imageName>`, normalizes target mean aesthetic score from $[1, 5]$ to $[0, 1]$ via $(s - 1) / 4$.
- `split_train_val`: Filters `PARA-GiaaTrain.csv` using the precomputed split file to produce disjoint training and validation DataFrames.
- `build_dataloaders`: Constructs `DataLoader` instances with pinned memory on CUDA.

### `src/ppa/train.py`
- Computes Pearson and Spearman correlations using pure PyTorch operations (`_pearson`, `_spearman`), avoiding external dependencies like `scipy`.
- Saves checkpoint states (`model`, `optimizer`, `epoch`, `val_metrics`) to `last.pt` and updates `best.pt` upon finding a new best validation SRCC.
- Calls `_report_test` to load `best.pt` once and evaluate against `test_loader`, saving output metrics to `final_metrics.json`.

---

## 7. Current Repository Status & Work-in-Progress Notes

1. **Unstaged Working Tree Modification in `src/ppa/data.py`**:
   - In lines 95–105 of `src/ppa/data.py`, a temporary subsample was introduced for rapid debugging:
     ```python
     # TEMPORARY: quick training sanity check
     train_df = train_df.sample(n=min(128, len(train_df)), random_state=42).reset_index(drop=True)
     val_df = val_df.sample(n=min(64, len(val_df)), random_state=42).reset_index(drop=True)
     ```
   - *Note for full runs*: To execute a full 25,398-image training run, this temporary subsample must be reverted (`git restore src/ppa/data.py`).
2. **Pre-trained Checkpoints on Disk**:
   - `checkpoints/para/learnable_temp/best.pt` (~831 MB) is present, holding the weights for the top-performing configuration (SRCC 0.9164 / PLCC 0.9456).
   - `checkpoints/para/quick_test/` contains artifacts from a 1-epoch quick sanity run.
3. **Hardware & Environment**:
   - Tested and designed for Python 3.12 managed via `uv`.
   - Automated device selection supports CUDA, Apple Silicon MPS, or CPU fallback.

---

## 8. How to Run the Project

### Environment Setup
```bash
# Sync virtual environment using uv
uv sync
```

### Verification & Smoke Testing
```bash
# Verify integrity of committed validation split
uv run python scripts/build_para_val_split.py

# Run standalone shape check on CPU (no dataset required)
PYTHONPATH=src uv run python scripts/smoke_test.py
```

### Full Model Training

```bash
# 1. Paper-literal reproduction (fixed logit scale = 1.0)
PYTHONPATH=src uv run python scripts/train.py \
  --ckpt-dir checkpoints/para/giaa

# 2. Optimized reproduction with learnable temperature (Headline configuration)
PYTHONPATH=src uv run python scripts/train.py \
  --learnable-temp \
  --ckpt-dir checkpoints/para/learnable_temp
```
Both commands evaluate the validation set per epoch, preserve the best validation model at `best.pt`, and perform a single final evaluation on the test set, outputting `final_metrics.json`.
