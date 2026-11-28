# Backend - Reddit Year Prediction API

FastAPI backend for predicting Reddit comment years using transformer models.

## Overview

The backend provides a REST API endpoint (`/predict`) that uses a RoBERTa-base model to classify Reddit comments into time periods. It prioritizes a 2-bin classifier (2008-2010 vs 2020-2022) and falls back to a regression model if the classifier is unavailable.

## Structure

```
backend/
├── api/                    # FastAPI server
│   └── main.py            # API endpoints and server configuration
├── core/                   # Core ML logic
│   ├── predict.py        # YearPredictor class for inference
│   ├── models.py         # Model loading utilities
│   ├── dataset.py        # Data loading and preprocessing (classifiers)
│   ├── regression_data.py # Cleaning, sampling and splits for the regressor
│   └── visualize.py      # Visualization utilities
├── training/              # Training scripts
│   ├── 2binclassifier.py # 2-bin classifier training
│   ├── train_classifier.py # Generic classifier training
│   ├── train_regressor.py  # Regression model training
│   ├── export_fp16.py      # Half-precision copy for serving
│   └── legacy/             # Legacy scripts kept for reference
│       ├── redditmain.py     # Legacy classifier (reference)
│       └── redditregression.py # Legacy regression (reference)
├── models/                # Model artifacts
│   ├── reddit_year_model/ # 2-bin classifier weights
│   └── reddit_year_model_regressor/ # Regression model weights
├── scripts/               # Utility scripts
│   ├── generate_visualizations.py # Generate evaluation plots
│   └── generate_wordclouds.py     # Generate word clouds
├── config/                # Configuration files
│   ├── config.json
│   └── merges.txt
└── requirements.txt       # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Provide Model Weights

Place your trained model files in the appropriate directories:

- **2-bin Classifier** (primary): `backend/models/reddit_year_model/`
  - Required: `model.safetensors`, `config.json`, `tokenizer.json`, `vocab.json`, etc.
  
- **Regression Model** (fallback): `backend/models/reddit_year_model_regressor/`
  - Same structure as above

See [../docs/copy-model-files.md](../docs/copy-model-files.md) for detailed instructions.

### 3. Start the API Server

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (Swagger UI)
- **Health Check**: http://localhost:8000/health

## API Endpoints

### POST `/predict`

Predict the time period for a Reddit comment.

**Request:**
```json
{
  "text": "This meme is fire fr fr no cap",
  "task_type": "classification"
}
```

**Response:**
```json
{
  "predicted_year": 2015,
  "predicted_year_range": "2008-2010",
  "confidence": 0.92,
  "class_probabilities": {
    "2008-2010": 0.08,
    "2020-2022": 0.92
  }
}
```

**Parameters:**
- `text` (string, required): The Reddit comment text to analyze
- `task_type` (string, optional): `"classification"` or `"regression"` (default: `"classification"`)

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Environment Variables

- `MODEL_PATH_CLASSIFICATION` (default: `./backend/models/reddit_year_model`)
  - Path to the 2-bin classifier model directory
  
- `MODEL_PATH_REGRESSION` (default: `./backend/models/reddit_year_model_regressor_fp16`, a half-precision copy of `reddit_year_model_regressor` that fits a 512 MB host)
  - Path to the regression model directory

## Training Models

### 2-bin Classifier (2008-2010 vs 2020-2022)

```bash
python backend/training/2binclassifier.py
```

This script:
- Loads data from `data/sampled_comments/RC_YYYY-MM.csv`
- Trains a RoBERTa-base model for binary classification
- Saves weights to `./backend/models/reddit_year_model/`

### Regression Model (2006-2024)

Check the data first. This loads no model and writes nothing:

```bash
python -m backend.training.train_regressor --dry_run
```

The data is cleaned (deleted and removed comments and self-declared bots are
dropped, some bots remain; dump formatting such as HTML escapes is undone),
sampled at random and balanced by year. March of every year is held out for
validation and September for test; training uses the other ten months.

The sample is narrower than "every month" suggests. `data/commentcleaner` kept
only the first 20,000 comments of each monthly dump, so from 2009 a file covers
less than a day, and from 2012 typically under an hour, at the start of the
month. The test score says little about comments from other days or hours.

Training needs a GPU. Run it from the repository root. In a GPU notebook install
these versions, not the root `requirements.txt`, which pins CPU-only PyTorch:

```bash
pip install transformers==4.57.1 accelerate==1.12.0 scikit-learn pandas
```

Trial run first (about 100,000 comments):

```bash
python -m backend.training.train_regressor \
  --train_per_year 5300 --num_epochs 2 --output_dir output/regressor-trial
```

Full run (760,000 comments, 3 epochs):

```bash
python -m backend.training.train_regressor
```

A free notebook can disconnect during a long run. Point `--output_dir` at storage
that outlives the notebook (about 4 GB, for example a mounted drive). To continue,
run the same command again with `--resume`. Stopping and resuming was checked with
a tiny stand-in model on a CPU, not yet with `roberta-base` on a GPU.

Each run writes to its output folder: checkpoints, `final/` (full precision),
`fp16/` (the copy to serve) and `metrics.json`. The gate is in `metrics.json`:
`passes_gate` is true only if the model's test error is lower than that of a
TF-IDF baseline fitted on the same training comments and scored on the same test
comments. On the full data the baseline reached a test error of 3.61 years on a
laptop, where it took about 6 minutes. Deploy only a full run that passes, by
copying `fp16/` over `backend/models/reddit_year_model_regressor_fp16/` and
`final/` over `backend/models/reddit_year_model_regressor/` in a pull request.
Do not add checkpoint folders to Git.

### Generic Classifier Training

```bash
python -m backend.training.train_classifier \
  --start_year 2006 \
  --end_year 2024 \
  --num_bins 2 \
  --output_dir ./backend/models/reddit_year_model_classifier
```

## Model Loading Strategy

The backend uses a **lazy loading** approach with automatic fallback:

1. **Primary Model**: On first request, attempts to load the 2-bin classifier from `./backend/models/reddit_year_model/`
2. **Fallback Model**: If classifier weights are missing, automatically falls back to the regression model in `./backend/models/reddit_year_model_regressor/`
3. **Caching**: Once loaded, the model stays in memory for subsequent requests

Check the startup logs to see which model was loaded:
```
Loaded 2-bin classifier model from ./backend/models/reddit_year_model
```

## Generating Visualizations

After training, generate evaluation plots:

```bash
python backend/scripts/generate_visualizations.py
```

This creates:
- `confusion-matrix.png`
- `prediction-vs-actual.png`
- `error-distribution.png`
- `temporal-accuracy.png`

Save these to `frontend/public/` for display on the website.

See [../docs/generate-visualizations.md](../docs/generate-visualizations.md) for more details.

## Development

### Running Tests

```bash
# Test the API endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "test comment", "task_type": "classification"}'
```

### Code Structure

- **`api/main.py`**: FastAPI application, request/response models, endpoint handlers
- **`core/predict.py`**: `YearPredictor` class - handles model loading, tokenization, inference
- **`core/models.py`**: Model loading utilities for different architectures
- **`core/dataset.py`**: Data loading, preprocessing, label preparation
- **`core/visualize.py`**: Plotting utilities for model evaluation

## Troubleshooting

**Model not loading?**
- Check that `model.safetensors` exists in the model directory
- Verify all tokenizer files are present (`tokenizer.json`, `vocab.json`, etc.)
- Check startup logs for error messages

**API returns 500 error?**
- Ensure the model files are in the correct directory
- Check that the backend has read permissions for the model directory
- Verify the model architecture matches the config.json

**Predictions always return the same value?**
- The backend may be using the regression fallback model
- Ensure classifier weights are in `./backend/models/reddit_year_model/`
- Restart the API server after adding model files

## Dependencies

See `requirements.txt` for the full list. Key dependencies:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` - PyTorch for model inference
- `transformers` - Hugging Face transformers library
- `pydantic` - Data validation

