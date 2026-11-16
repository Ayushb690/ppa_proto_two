# Architecture & Implementation Plan: Interactive Interface for PPA

This document outlines the complete plan to build an interactive, user-facing web interface for the **PPA (Probabilistic Prompt Adaptation)** Image Aesthetics and Quality Assessment model.

---

## 1. Executive Overview & Goals

The current repository contains the PyTorch core, training scripts, and trained checkpoints (`checkpoints/para/learnable_temp/best.pt`) for PPA. While effective as an offline research framework, inspecting results currently requires running CLI scripts.

### Core Goals
1. **Interactive Image Scoring**: Upload any image and immediately receive an overall aesthetic score (on both $[1, 5]$ and normalized $[0, 1]$ scales).
2. **Concept Explainability Visualization**: Expose the 260 antonym prompt pairs to reveal *why* the image received its score (e.g., how much "Good Lighting" or "Blur" contributed via $p_\phi(t \mid x, c)$ and $\bar{s}(x, t)$).
3. **Comparative Analysis (A/B Testing)**: Upload two photos side-by-side to determine which has better aesthetic/technical quality and see contrasting criteria.
4. **Clean Decoupled Architecture**: Create a dedicated `client/` directory for the UI and a dedicated `server/` directory for model inference.

---

## 2. Why a Separate `server/` Folder is Necessary

While modern web browsers can execute lightweight ML models via WebAssembly or WebGPU (e.g., ONNX Runtime Web), a dedicated Python backend in `server/` is necessary for this project:

```
┌─────────────────────────────────┐           HTTP / REST           ┌─────────────────────────────────┐
│         client/ (UI)            │◄───────────────────────────────►│        server/ (API)            │
│  - Image Upload & Preview       │        (Multipart Form /        │  - FastAPI / Uvicorn Server     │
│  - Score & Attribute Cards      │             JSON)               │  - Checkpoint & Model In-Memory │
│  - Radar & Explainability Chart │                                 │  - PyTorch CUDA / MPS Pipeline  │
└─────────────────────────────────┘                                 └────────────────┬────────────────┘
                                                                                     │ Calls Python API
                                                                                     ▼
                                                                    ┌─────────────────────────────────┐
                                                                    │        src/ppa/ (Core)          │
                                                                    │  - ViT-B/16 OpenCLIP Backbone   │
                                                                    │  - checkpoints/best.pt (~831MB) │
                                                                    │  - 260 Antonym Prompt Pairs     │
                                                                    └─────────────────────────────────┘
```

### Key Reasons for a Dedicated Server:
1. **Model Footprint and Memory (~831 MB Checkpoint)**:
   - The model checkpoint alone is ~831 MB, plus the OpenCLIP ViT-B/16 visual encoder and precomputed text token buffers.
   - Forcing client browsers to download an ~850MB+ bundle over the internet would create unusable load times and crash on mobile or low-memory devices.
2. **PyTorch & OpenCLIP Ecosystem Dependencies**:
   - PPA relies directly on `torch`, `torchvision`, `open-clip-torch`, and custom CUDA/MPS acceleration kernels.
   - A Python backend directly imports `from ppa.model import PPA` and `from ppa.config import Config` without fragile weight conversions (such as ONNX or TensorFlow.js export, which often fail on custom multi-branch architectures like prompt-dropout and einsum operations).
3. **Hardware Acceleration (GPU / MPS / CPU)**:
   - Inference requires CLIP image transformations and a 12-layer Vision Transformer forward pass. Running this on a dedicated server with GPU/MPS ensures sub-100ms response times.
4. **Decoupling of Concerns & API Reusability**:
   - A standalone backend allows any frontend (web client, mobile app, desktop app, or CLI tool) to query the same inference engine via standard REST/JSON endpoints.
5. **Stateful Model Warm-Up**:
   - The server loads the model, weights, and precomputes the 260 text prompt embeddings **once** at server startup. Subsequent inference requests only require passing the image through the visual encoder and MLPs, ensuring minimal latency.

---

## 3. Tech Stack Options for `client/`

Below are the three primary options for structuring the `client/` interface, along with their pros, cons, and requirements.

### Option A (Recommended): Next.js 15 (React 19) + TypeScript + Tailwind CSS + Lucide + Recharts
*Best for: A modern, production-grade, highly responsive web application with rich charts.*

- **Why it fits**: Provides fast server-side rendering for landing pages, instant client reactivity for image drag-and-drop, and strong charting libraries (Recharts / Chart.js) to display prompt distributions and radar graphs.
- **Requirements & Dependencies**:
  - Node.js $\ge$ 20.x, pnpm/npm
  - Core dependencies:
    ```json
    {
      "dependencies": {
        "react": "^19.0.0",
        "react-dom": "^19.0.0",
        "next": "^15.0.0",
        "lucide-react": "^0.460.0",
        "recharts": "^2.13.0",
        "clsx": "^2.1.1",
        "tailwind-merge": "^2.5.4",
        "axios": "^1.7.7"
      },
      "devDependencies": {
        "typescript": "^5.6.0",
        "tailwindcss": "^3.4.14",
        "postcss": "^8.4.47"
      }
    }
    ```

---

### Option B: Vite + Vue 3 / Svelte 5 + Tailwind CSS + Chart.js
*Best for: Ultra-lightweight, lightning-fast Single Page Application (SPA).*

- **Why it fits**: Simpler mental model with minimal overhead. Vue's reactive bindings or Svelte's compile-time reactivity make handling image dropzones and reactive sliders very fast without full-stack framework overhead.
- **Requirements & Dependencies**:
  - Node.js $\ge$ 20.x, Vite $\ge$ 5.x
  - Core dependencies:
    ```json
    {
      "dependencies": {
        "vue": "^3.5.0",
        "chart.js": "^4.4.6",
        "vue-chartjs": "^5.3.1",
        "lucide-vue-next": "^0.460.0",
        "axios": "^1.7.7"
      },
      "devDependencies": {
        "vite": "^5.4.0",
        "typescript": "^5.6.0",
        "tailwindcss": "^3.4.14"
      }
    }
    ```

---

### Option C (Data Science Alternative): Monolithic Streamlit / Gradio
*Best for: 1-day internal research demo with zero frontend JavaScript code.*

- **Why it fits**: Can be implemented entirely in Python without separate `client/` and `server/` codebases.
- **Drawbacks**: Limited UI customizability, slow state management, clunky mobile layout, and cannot easily be converted into a consumer-facing web product.
- **Verdict**: Useful as an initial prototype, but keeping decoupled `client/` and `server/` directories provides far greater control over explainability visualizations and responsive UI.

---

## 4. Proposed Client Interface Design & Features

The `client/` application will feature three main views:

### 1. Single Image Analysis & Explainability Dashboard
- **Image Dropzone**: Drag-and-drop or file selector with instant local thumbnail preview.
- **Score Meter**:
  - Large radial gauge displaying Predicted Aesthetic Score (e.g., `3.84 / 5.00` or `76.8%`).
  - Score classification badge (e.g., "High Quality / Exhibition Grade", "Good Everyday Photo", "Low / Blurry").
- **Top Concept Drivers (Explainability)**:
  - **Positive Drivers**: Prompts with high selection weight $p_\phi$ and high positive score $\bar{s}$ (e.g., `+ "Good Lighting." (Weight: 14.2%, Score: 0.89)`).
  - **Negative Penalties**: Prompts that reduced the score (e.g., `- "Blurry photo." (Weight: 8.5%, Score: 0.12)`).
- **Attribute Category Breakdown**:
  - Tabbed or accordion view showing the 4 prompt pool subsets:
    1. *Aesthetic Attributes* (11 attributes, e.g. Rule of Thirds, Symmetry, Depth of Field).
    2. *Quality Attributes* (9 attributes, e.g. Blur, Noise, Color-related).
    3. *General Aesthetic* (100 pairs).
    4. *General Quality* (80 pairs).
  - Radar chart of the 11 aesthetic attributes to give a quick visual fingerprint of composition, lighting, and color harmony.

### 2. A/B Photo Comparison (Photo Shoot Curation)
- Upload two photos side-by-side (e.g., two burst shots of the same subject).
- Highlights the winning photo with green border and shows a differential analysis:
  - Why Photo A scored higher than Photo B (e.g., +15% sharper focus, +8% better subject emphasis).

### 3. Batch Image Gallery Mode
- Upload a zip or multiple images.
- Displays a sortable grid ordered by aesthetic score, allowing photographers to quickly filter out unflattering or blurry shots.

---

## 5. Server Specification (`server/`)

### Backend Stack
- **Framework**: `FastAPI` (asynchronous, high performance, automatic OpenAPI documentation).
- **Server Engine**: `uvicorn` with `gunicorn` process management.
- **Data Validation**: `pydantic v2`.
- **Image Processing**: `Pillow`, `torchvision`.

### Server Requirements (`pyproject.toml` or `server/requirements.txt`)
```toml
fastapi >= 0.115.0
uvicorn[standard] >= 0.32.0
python-multipart >= 0.0.12
pydantic >= 2.9.0
pillow >= 10.0.0
```

### Proposed API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Verifies server status, active device (CUDA/MPS/CPU), and loaded checkpoint. |
| `GET` | `/api/prompts` | Returns the list of 260 antonym prompt pairs grouped by category and attribute. |
| `POST` | `/api/analyze` | Accepts an image file (`multipart/form-data`) and optional `task_id`. Returns predicted score, top drivers, and category aggregates. |
| `POST` | `/api/compare` | Accepts two image files. Returns scores, delta comparison, and dominant differences. |
| `POST` | `/api/batch` | Accepts up to 50 images and returns sorted ranking and summary scores. |

### API Data Contract Sample (`/api/analyze` Response)
```json
{
  "score": {
    "raw_1_to_5": 3.842,
    "normalized_0_to_1": 0.7105,
    "confidence_std": 0.10
  },
  "top_positive_prompts": [
    {
      "pos_prompt": "Good lighting.",
      "neg_prompt": "Poor lighting.",
      "category": "aesthetic_attribute",
      "attribute": "Good Lighting",
      "selection_weight": 0.1245,
      "score": 0.8921
    }
  ],
  "top_negative_prompts": [
    {
      "pos_prompt": "Sharp focus photo.",
      "neg_prompt": "Out of focus blurry photo.",
      "category": "quality_attribute",
      "attribute": "Blur",
      "selection_weight": 0.0812,
      "score": 0.2150
    }
  ],
  "attribute_breakdown": {
    "Good Lighting": 0.84,
    "Color Harmony": 0.78,
    "Rule of Thirds": 0.65,
    "Depth of Field": 0.72
  },
  "device": "cuda:0",
  "inference_time_ms": 42.8
}
```

---

## 6. Target Directory Layout

```
ppa-reimplementation/
├── client/                     # [NEW] Web Frontend (Next.js / Vite SPA)
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── public/
│   └── src/
│       ├── components/
│       │   ├── Dropzone.tsx
│       │   ├── ScoreMeter.tsx
│       │   ├── PromptDriversTable.tsx
│       │   ├── AttributeRadarChart.tsx
│       │   └── CompareViewer.tsx
│       ├── lib/
│       │   ├── api.ts          # Axios client communicating with server
│       │   └── types.ts        # TypeScript schemas matching Pydantic response
│       └── app/ (or views/)
│           └── page.tsx        # Main dashboard
│
├── server/                     # [NEW] FastAPI Backend
│   ├── __init__.py
│   ├── main.py                 # FastAPI application definition & CORS
│   ├── routes.py               # Endpoint handlers (/analyze, /compare)
│   ├── schemas.py              # Pydantic request/response models
│   ├── service.py              # Model caching, warm-up, and inference runner
│   └── requirements.txt        # Backend dependencies
│
├── src/ppa/                    # Core Model Package (unchanged)
│   ├── model.py
│   ├── clip_backbone.py
│   ├── prompts.py
│   └── ...
├── checkpoints/                # Trained weights (best.pt)
│   └── para/learnable_temp/best.pt
├── scripts/                    # Existing training & smoke test scripts
├── docs/                       # Existing documentation
├── CODEBASE_ANALYSIS.md        # Technical analysis of the codebase
└── plan.md                     # This plan file
```

---

## 7. Step-by-Step Implementation Roadmap

### Phase 1: Server Implementation (`server/`)
1. Create `server/` directory and configure `main.py` with FastAPI and CORS middleware.
2. Implement `service.py`:
   - Initialize `Config(score=ScoreConfig(learnable_logit_scale=True))` and load `checkpoints/para/learnable_temp/best.pt`.
   - Precompute frozen prompt embeddings and warm up the CLIP image encoder.
3. Add inference methods:
   - Extract raw scores $\bar{s}(x, t)$ and selection probabilities $p_\phi(t \mid x, c)$.
   - Sort prompts by contribution ($p_\phi \times \bar{s}$) to extract top drivers.
4. Expose `/api/health`, `/api/analyze`, and `/api/compare`.
5. Test endpoints with curl/Postman.

### Phase 2: Client Setup & Scaffolding (`client/`)
1. Initialize the client using Next.js (or Vite + React/TypeScript) inside `client/`.
2. Configure Tailwind CSS and design system (clean modern dark/light mode layout).
3. Set up API client in `client/src/lib/api.ts` pointing to `http://localhost:8000`.

### Phase 3: Core UI Development
1. Build `Dropzone.tsx`: Image upload with drag-and-drop, validation (JPEG/PNG, max 20MB), and preview.
2. Build `ScoreMeter.tsx`: Animated circular gauge displaying score on both $[1, 5]$ and $0–100\%$ scale.
3. Build `PromptDriversTable.tsx`: Display top 5 positive and top 5 negative prompt criteria with interactive tooltips.
4. Build `AttributeRadarChart.tsx`: Radar visualization using Recharts to display attribute breakdown.

### Phase 4: A/B Comparison & Advanced Modes
1. Build side-by-side comparison screen for two images.
2. Calculate score delta and highlight distinguishing prompt weights.
3. Add preset demo images from the PARA dataset (`iaa_pub*.jpg`) so users can test immediately without uploading their own photos.

### Phase 5: Verification, Packaging & Dockerization
1. Write a verification script or integration test (`tests/test_api.py`).
2. Provide a single launch script (e.g., `scripts/start_demo.py` or `docker-compose.yml`) that boots both the FastAPI server and the UI.
3. Update root `README.md` with instructions on how to start and use the interface.
