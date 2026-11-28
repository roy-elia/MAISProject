
# Modeling Linguistic Change Over Time on Reddit

Transformer-based NLP models that estimate **when a Reddit comment was written using only its text**.  
We analyze **18 years of linguistic drift (2006 → 2024)** and present the results through a **production-style web application** built with **Next.js (frontend)** and **FastAPI (backend)**.

This project was developed for the **MAIS Machine Learning Bootcamp (Fall 2025)** and focuses on **language evolution, temporal generalization, and real-world ML deployment**.



## Live Deployment

The project is deployed online using a **modern split-stack architecture**, mirroring real-world ML systems.

**Frontend (Vercel)**  
https://mais-frontend-psi.vercel.app/

**Backend API (Render)**  
https://maisproject.onrender.com

The frontend sends user-submitted Reddit-style comments to the backend API, which performs inference using fine-tuned transformer models and returns a predicted year (or time bin).

*Note:* The backend may take a few seconds to respond on the first request due to Render cold starts.



## Snapshot

- **85.3% accuracy** using a **RoBERTa-base 2-bin classifier** (2008–2010 vs 2020–2022)
- **Regression model (live demo)**: test-split MAE 4.4 years, RMSE 5.2 (228 comments); always guessing 2015 gives 4.7 and 5.5
- **Next.js frontend** for visualization and interaction
- **FastAPI backend** for inference
- **Visualizations and storytelling assets** ready for a project fair or demo booth



## Architecture

| Layer | Stack | Notes |
|------|------|------|
| Frontend | Next.js 14, Tailwind, shadcn/ui | `frontend/` contains the entire site (hero, methodology, demo, visualizations) |
| Backend | FastAPI, Uvicorn | `backend/api/main.py` exposes `/predict`, prioritizing classification and falling back to regression |
| Models | Hugging Face Transformers | `reddit_year_model` (2-bin classifier) and `reddit_year_model_regressor` (exact year) |
| Training | Python + PyTorch | Training scripts in `backend/training/` |
| Assets | Static PNG / SVG | Stored in `frontend/public/` |



## Repository Layout

```
MAIS202-project/
|
|-- backend/                       Backend API and ML models
|   |-- api/                       FastAPI server
|   |-- core/                      Core prediction and model logic
|   |-- training/                  Training scripts
|   |-- models/                    Model artifacts (trained weights)
|   |-- scripts/                   Utility scripts (visualizations, analysis)
|   |-- config/                    Configuration files
|
|-- frontend/                      Next.js 14 site
|   |-- app/                       Pages and layout
|   |-- components/                React components
|   |-- public/                    Logos, favicon, visualization PNGs
|
|-- data/                          Dataset files
|-- docs/                          Documentation and guides
|-- scripts/                       Startup and utility scripts
|-- README.md                      Project overview
```



## Quick Start

### 1. Install Dependencies

**Backend**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

**Frontend**
```bash
cd frontend
npm install
cd ..
```

See `backend/README.md` and `frontend/README.md` for detailed setup instructions.



### 2. Provide Model Weights

To enable inference, trained model weights must be added to the backend.

**Classifier (preferred demo experience)**  
Place all Hugging Face model files (`config.json`, tokenizer files, `model.safetensors`, etc.) inside:
```
backend/models/reddit_year_model/
```

**Regression fallback (optional)**  
Keep the following directory available so the `/predict` endpoint remains functional even if classifier weights are missing:
```
backend/models/reddit_year_model_regressor/
```

For detailed instructions, see:  
`docs/copy-model-files.md`



### 3. Start Services

Once dependencies and model weights are in place, start the backend and frontend services.

#### Option A: Use the Startup Script
```bash
./start_all.sh
```

This will automatically start both the backend API and the frontend web application.

#### Option B: Start Manually

**Backend**
```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend**
```bash
cd frontend
npm run dev
```

Open the application in your browser at:  
http://localhost:3000



## Project Structure

- **backend/README.md** — Backend setup, training, and API documentation  
- **frontend/README.md** — Frontend setup and development guide  
- **docs/** — Additional documentation:
  - `demo-guide.md` — Step-by-step demo walkthrough  
  - `testing-checklist.md` — Pre-presentation checklist  
  - `copy-model-files.md` — How to add trained model weights  
  - `generate-visualizations.md` — Visualization generation guide  



## Results

| Model Type | # Bins | Accuracy | Comments |
|-----------|--------|----------|----------|
| Linear Regression | — | MSE ≈ 19 | Too slow and data-hungry |
| Multi-class Classifier | 2 | ~85% | Best balance of speed and accuracy |
| Multi-class Classifier | 3 | ~60% | Needs more epochs |
| Multi-class Classifier | 4 | ~53% | Needs more data |



## Final Model

- **Architecture:** RoBERTa-base  
- **Task:** 2-bin classification (2008–2010 vs 2020–2022)  
- **Epochs:** 3  
- **Learning rate:** 2e-5  
- **Accuracy:** 85.3%  
- **Precision:** 87.3%  
- **F1 score:** 85.5%  



## Project Purpose

This project demonstrates that **language alone contains strong temporal signals**.

By modeling linguistic drift, we show how NLP systems can:
- Detect cultural and slang evolution
- Reveal temporal bias in language models
- Motivate retraining strategies for long-lived NLP systems



## Contact

For questions, feedback, or collaboration inquiries, please reach out to:

**roy.elia@mail.mcgill.ca**

