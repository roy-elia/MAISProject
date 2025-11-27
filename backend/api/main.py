"""
FastAPI backend for Reddit Year Prediction Web Demo.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add backend directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.predict import YearPredictor

app = FastAPI(
    title="Reddit Year Prediction API",
    description="API for predicting the year of Reddit comments using transformer models",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[YearPredictor] = None


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    text: str
    task_type: Optional[str] = "classification"  # "classification" or "regression" - default to 2-bin classifier


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    predicted_year: int
    predicted_year_range: Optional[str] = None  # For 2-bin classifier: "2008-2010" or "2020-2022"
    confidence: Optional[float] = None
    class_probabilities: Optional[dict] = None
    year_index: Optional[float] = None
    raw_prediction: Optional[float] = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global predictor
    
    # Default model paths - adjust as needed
    model_path_regression = os.getenv(
        "MODEL_PATH_REGRESSION",
        "./backend/models/reddit_year_model_regressor"
    )
    model_path_classification = os.getenv(
        "MODEL_PATH_CLASSIFICATION",
        "./backend/models/reddit_year_model"  # Default to 2-bin classifier model
    )
    
    # Try to load 2-bin classifier model first (final model)
    if os.path.exists(model_path_classification):
        try:
            predictor = YearPredictor(
                model_path=model_path_classification,
                task_type="classification"
            )
            print(f"Loaded 2-bin classifier model from {model_path_classification}")
        except Exception as e:
            print(f"Error loading classification model: {e}")
            # Fallback to regression if classification fails
            if os.path.exists(model_path_regression):
                try:
                    predictor = YearPredictor(
                        model_path=model_path_regression,
                        task_type="regression",
                        start_year=2006,
                        end_year=2024
                    )
                    print(f"Loaded regression model from {model_path_regression}")
                except Exception as e2:
                    print(f"Error loading regression model: {e2}")
    elif os.path.exists(model_path_regression):
        try:
            predictor = YearPredictor(
                model_path=model_path_regression,
                task_type="regression",
                start_year=2006,
                end_year=2024
            )
            print(f"Loaded regression model from {model_path_regression}")
        except Exception as e:
            print(f"Error loading regression model: {e}")
    else:
        print(f"Warning: Models not found at {model_path_classification} or {model_path_regression}")
        print("Please train a model first or set MODEL_PATH_CLASSIFICATION environment variable")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Reddit Year Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict year for a comment",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_year(request: PredictionRequest):
    """
    Predict the year for a Reddit comment.
    
    Args:
        request: Prediction request with text and optional task_type
    
    Returns:
        Prediction response with predicted year and metadata
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure a model is trained and available."
        )
    
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    try:
        result = predictor.predict(request.text)
        
        return PredictionResponse(
            predicted_year=result["predicted_year"],
            predicted_year_range=result.get("predicted_year_range"),
            confidence=result.get("confidence"),
            class_probabilities=result.get("class_probabilities"),
            year_index=result.get("year_index"),
            raw_prediction=result.get("raw_prediction")
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(texts: list[str]):
    """
    Predict years for multiple comments.
    
    Args:
        texts: List of comment texts
    
    Returns:
        List of prediction results
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        results = predictor.predict_batch(texts)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

