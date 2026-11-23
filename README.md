# Frame_IQ

A PyTorch-based implementation of **Frame_IQ**, a framework for Unified Image Aesthetics and Quality Assessment, based on the CVPR 2026 research paper.

**Original Paper:** [Probabilistic Prompt Adaptation for Unified Image Aesthetics and Quality Assessment](https://openaccess.thecvf.com/content/CVPR2026/papers/Hara_Probabilistic_Prompt_Adaptation_for_Unified_Image_Aesthetics_and_Quality_Assessment_CVPR_2026_paper.pdf)

## Overview

Frame_IQ uses OpenCLIP and human-interpretable text prompts to evaluate image aesthetics and quality.

It consists of two main components:

- **Score Model:** Estimates image quality using text prompts.
- **Prompt Selection Model:** Selects relevant prompts based on the input image and assessment task.

The final prediction is obtained by combining scores across the selected prompts.

## Features

- Unified image aesthetics and quality assessment.
- 260 antonym prompt pairs covering 11 aesthetic attributes.
- Task-conditioned probabilistic prompt selection.
- Fine-tuned OpenCLIP (ViT-B/16) backbone.
- Interactive web application for image assessment.

## Model Performance

| Metric | Validation | Test |
|---|---:|---:|
| SRCC | 0.9155 | 0.9100 |
| PLCC | 0.9078 | 0.8982 |

## Technologies

- Python and PyTorch
- OpenCLIP (ViT-B/16)
- React
- PARA Benchmark

## Installation and Usage

**1. Run the verification suite**

```bash
uv run python scripts/verify_integration.py
```

**2. Start the backend**

```bash.\.venv\Scripts\python.exe scripts/serve.py --port 8000```

**3. Start the frontend**

```bash
cd client
pnpm run dev
```

## Model Checkpoint

The trained model checkpoint is available at:

`Data Files/best.pt`

## Reference

Hara, T. and Otsuka, Y. (2026). *Probabilistic Prompt Adaptation for Unified Image Aesthetics and Quality Assessment*. CVPR 2026.
