# PPA Aesthetics Studio (Client Interface)

An interactive, explainable Image Aesthetics and Quality Assessment web application powered by **Probabilistic Prompt Adaptation (PPA)** (CVPR 2026).

---

## Features

1. **Single Image Scoring & Confidence**:
   - Radial score gauge displaying both the PARA $[1.0, 5.0]$ scale and normalized percentage $[0, 100\%]$.
   - Aesthetic Tier classification (Masterpiece, High Aesthetic, Good Everyday, Sub-optimal, Low/Flawed).
   - Inference latency and parameter metrics.

2. **Concept Bottleneck Explainability**:
   - **Top Positive Drivers**: Explains which antonym prompt pairs boosted the image's score (e.g. *"Good lighting"*, *"Rule of thirds"*), displaying individual selection weights $p_\phi(t \mid x, c)$ and CLIP-IQA scores $\bar{s}(x, t)$.
   - **Negative Penalties & Flaws**: Identifies technical defects dragging down the score (e.g. *"Blurry photo"*, *"High sensor noise"*, *"Heavy compression artifacts"*).

3. **Multidimensional Radar Fingerprint**:
   - Interactive 8-axis SVG radar chart mapping aesthetic attributes (*Good Lighting*, *Color Harmony*, *Rule of Thirds*, *Subject Emphasis*, *Depth of Field*, *Vivid Colors*, *Balance & Symmetry*, *Interesting Content*).
   - Technical Image Quality Assessment (IQA) diagnostic progress bars (*Blur*, *Noise*, *Color-related*, *Contrast*, *Compression*).

4. **A/B Photo Comparison (Curation Mode)**:
   - Upload two photos side-by-side to determine which image won and view differential scores.

5. **260 Prompt Pool Inspector**:
   - Searchable, filterable, sortable database displaying all 260 antonym prompt pairs from CVPR 2026 Supplementary Tables S6–S9.

6. **Dual Mode (Live API & Demo Simulation)**:
   - If the FastAPI server is running at `http://localhost:8000`, the client queries the live PyTorch backend.
   - If the server is offline, the client runs an instant local demo simulation with curated benchmark presets.

---

## Getting Started

### Prerequisites
- Node.js $\ge$ 20.x
- `pnpm` (or `npm`)

### Installation & Launch

```bash
# Navigate to the client directory
cd client

# Install dependencies
pnpm install

# Start development server
pnpm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build
```bash
pnpm run build
pnpm run preview
```
