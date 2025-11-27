# Modeling Linguistic Change Over Time on Reddit

Transformer models that guess when a Reddit comment was written just by reading the text. We analyze 18 years of language drift (2006 → 2024) and present it in a polished Next.js + FastAPI demo for the MAIS Machine Learning Bootcamp.

## Snapshot
- **85.3%** accuracy with a RoBERTa-base 2-bin classifier (2008-2010 vs 2020-2022)
- **Regression fallback** (MAE ≈ 4.4 years) keeps the `/predict` endpoint alive even without classifier weights
- **Next.js frontend** for the showcase, **FastAPI backend** for inference
- **Visualizations + storytelling** assets ready for a project fair booth

## Architecture
| Layer | Stack | Notes |
| --- | --- | --- |
| Frontend | Next.js 14, Tailwind, shadcn/ui | `frontend/` contains the entire site (hero, methodology, demo, visualizations) |
| Backend | FastAPI, Uvicorn | `backend/api/main.py` exposes `/predict`, prioritizing the classifier and falling back to regression |
| Models | Hugging Face Transformers | `backend/models/reddit_year_model` (2-bin classifier) and `backend/models/reddit_year_model_regressor` (exact year) |
| Training | Python scripts + PyTorch | `backend/training/2binclassifier.py`, `backend/training/train_classifier.py`, `backend/training/train_regressor.py` |
| Assets | Static PNG/SVG | Drop plots and logos in `frontend/public/` |

## Repository Layout
```
MAIS202-project/
├── backend/                      # Backend API and ML models
│   ├── api/                     # FastAPI server
│   ├── core/                    # Core prediction and model logic
│   ├── training/                 # Training scripts
│   ├── models/                   # Model artifacts (drop trained weights here)
│   ├── scripts/                  # Utility scripts (visualizations, etc.)
│   └── config/                   # Configuration files
├── frontend/                     # Next.js 14 site
│   ├── app/                     # Pages + layout
│   ├── components/              # React components for each section
│   └── public/                  # Logos, favicon, visualization PNGs
├── data/                        # Dataset files
├── docs/                        # Documentation and guides
├── scripts/                     # Startup and utility scripts
└── README.md                    # This file
```

## Quick Start

### 1. Install Dependencies

**Backend:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
```

**Frontend:**
```bash
cd frontend && npm install && cd ..
```

See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md) for detailed setup instructions.

### 2. Provide Model Weights

- **Classifier (preferred demo experience)** → place Hugging Face files (config, tokenizer, `model.safetensors`, etc.) inside `./backend/models/reddit_year_model/`.
- **Regression fallback** → optional, but keep `./backend/models/reddit_year_model_regressor/` around so `/predict` keeps working even if classifier weights are missing.

See [docs/COPY_MODEL_FILES.md](docs/COPY_MODEL_FILES.md) for detailed instructions.

### 3. Start Services

**Option A: Use the startup script**
```bash
./start_all.sh
```

**Option B: Start manually**

Backend:
```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:
```bash
cd frontend
npm run dev
```

Open http://localhost:3000 and use the Interactive Demo section.

## Project Structure

- **[backend/README.md](backend/README.md)** - Backend setup, training, API documentation
- **[frontend/README.md](frontend/README.md)** - Frontend setup, development guide
- **[docs/](docs/)** - Additional documentation
  - `demo-guide.md` - Step-by-step playbook for running the demo
  - `testing-checklist.md` - Pre-presentation verification checklist
  - `COPY_MODEL_FILES.md` - How to add trained model weights
  - `GENERATE_VISUALIZATIONS.md` - Visualization generation guide

## Results

| Model Type            | # bins  | Accuracy | Comments |
|----------------------|---------|----------|----------|
| Linear Regression     |  None   | MSE = 19 | Too slow + requires a lot of data |
| Multi-class classifier|   2     | ~ 85%    | Best in terms of accuracy and speed |
| Multi-class classifier|   3     | ~ 60%    | Needs more epochs for better accuracy |
| Multi-class classifier|   4     | ~ 53%    | Needs more data for better accuracy |

**Final Model:**
- RoBERTa-base
- 2-bin classifier (2008-2010, 2020-2022)
- Epochs = 3
- Learning rate = 2e-5
- Accuracy: 85.3%
- Precision: 87.3%
- F1: 85.5%

## Contact

Questions or issues? Reach out to the MAIS Machine Learning Bootcamp Fall 2025 team.
