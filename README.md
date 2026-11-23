# Frame_IQ

**Frame_IQ** is a PyTorch-based implementation of a unified framework for **Image Aesthetics Assessment (IAA)** and **Image Quality Assessment (IQA)**, based on the CVPR 2026 research paper *Probabilistic Prompt Adaptation for Unified Image Aesthetics and Quality Assessment*.

## Overview

Frame_IQ leverages OpenCLIP and human-interpretable text prompts to assess the aesthetic appeal and perceptual quality of images.

The framework consists of two primary components:

- **Score Model:** Estimates image quality using image-text similarity with assessment prompts.
- **Prompt Selection Model:** Selects relevant prompts based on the input image and the assessment task.

The final prediction is generated through probabilistic aggregation of scores across the selected prompts.

## Key Features

- Unified image aesthetics and quality assessment.
- 260 antonymous prompt pairs covering 11 aesthetic attributes.
- Image- and task-conditioned probabilistic prompt selection.
- Fine-tuned OpenCLIP backbone (ViT-B/16).
- Probabilistic aggregation of prompt-based predictions.
- Interactive web application for image assessment.

## Model Performance

The model was evaluated using **Spearman Rank Correlation Coefficient (SRCC)** and **Pearson Linear Correlation Coefficient (PLCC)**.

| Evaluation Metric | Validation | Test |
|:---|---:|---:|
| SRCC | 0.9155 | 0.9100 |
| PLCC | 0.9078 | 0.8982 |

## Technology Stack

| Component | Technology |
|:---|:---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Vision-Language Model | OpenCLIP (ViT-B/16) |
| Frontend | React |
| Benchmark Dataset | PARA |
| Package Management | uv, pnpm |

## Installation and Usage

### Prerequisites

Ensure the following tools are installed:

- Python
- [uv](https://docs.astral.sh/uv/)
- Node.js and pnpm

### 1. Verify the Integration

Run the integration verification suite from the project root:

```bash
uv run python scripts/verify_integration.py
```

### 2. Start the Backend

On Windows, run:

```powershell
.\.venv\Scripts\python.exe scripts/serve.py --port 8000
```

The backend will start on port `8000`.

### 3. Start the Frontend

Open a separate terminal and navigate to the client directory:

```bash
cd client
pnpm install
pnpm run dev
```

Open the local URL displayed by the frontend development server.

## Model Checkpoint

The trained model checkpoint is available in the repository at:

```text
Data Files/best.pt
```

Ensure the checkpoint is present at the expected path before running model inference.
