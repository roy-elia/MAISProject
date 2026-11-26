"""
Inference script to predict year for Reddit-style comments.
"""

import os
import sys
import threading
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, BertForSequenceClassification

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import load_classification_model, load_regression_model

# The free host has a fraction of one CPU. PyTorch otherwise starts one thread
# per visible host core, and those threads fight over that fraction.
# Set TORCH_NUM_THREADS on a host with real cores.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))


class YearPredictor:
    """Predictor class for Reddit comment year prediction."""
    
    def __init__(self, model_path: str, task_type: str = "regression", start_year: int = 2006, end_year: int = 2024):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model directory
            task_type: "classification" or "regression"
            start_year: Starting year (for regression)
            end_year: Ending year (for regression)
        """
        self.model_path = model_path
        self.task_type = task_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        if task_type == "classification":
            # For classification, we need to infer num_labels from config
            config_path = os.path.join(model_path, "config.json")
            config = {}
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    num_labels = config.get('num_labels', 2)
            else:
                num_labels = 2
            
            if "distilbert" in model_path.lower() or (config and "distilbert" in str(config.get('model_type', ''))):
                self.model = BertForSequenceClassification.from_pretrained(model_path)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Infer years from model config
            # Check if it's the 2-bin classifier (2008-2010 vs 2020-2022)
            if config.get('problem_type') == 'single_label_classification' and config.get('num_labels') == 2:
                # 2-bin classifier: 2008-2010 vs 2020-2022
                self.years = ["2008-2010", "2020-2022"]
            else:
                # Default for binary classification (2013 vs 2024)
                self.years = [2013, 2024]
            
        else:  # regression
            config = AutoConfig.from_pretrained(model_path)
            config.num_labels = 1
            config.problem_type = "regression"
            # Half precision by default: the fp32 weights (about 500 MB) do not fit
            # in a 512 MB free instance next to PyTorch. Scores match fp32 within
            # 0.01 on our checks. Set MODEL_DTYPE=float32 where memory allows.
            dtype = getattr(torch, os.environ.get("MODEL_DTYPE", "float16"))
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, dtype=dtype)
            self.year_range = list(range(start_year, end_year + 1))
        
        self.model.to(self.device)
        self.model.eval()
        # One prediction at a time. The API runs requests in a thread pool, and a
        # request the client gave up on keeps running; without the lock those
        # pile up and share the CPU until every request times out.
        self._lock = threading.Lock()
        print(f"Model loaded from {model_path} on device: {self.device}")
    
    def predict(self, text: str, max_length: int = 256) -> dict:
        """
        Predict year for a single text.
        
        Args:
            text: Input text (Reddit comment)
            max_length: Maximum sequence length
        
        Returns:
            Dictionary with prediction results
        """
        # Tokenize
        # No padding: a single sequence needs none, and padding every comment to
        # max_length made each request cost as much as a 256-token comment.
        # Outputs are identical (checked on the regressor, max diff 0.0).
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Predict
        with self._lock, torch.inference_mode():
            outputs = self.model(**encoding)
            logits = outputs.logits
        
        if self.task_type == "classification":
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()
            predicted_year_range = self.years[pred_class]
            
            # For year ranges, return the midpoint year for compatibility
            if isinstance(predicted_year_range, str) and "-" in predicted_year_range:
                # Parse range like "2008-2010" or "2020-2022"
                start, end = map(int, predicted_year_range.split("-"))
                predicted_year = (start + end) // 2  # Midpoint
            else:
                predicted_year = predicted_year_range
            
            return {
                "predicted_year": predicted_year,
                "predicted_year_range": predicted_year_range,
                "confidence": confidence,
                "class_probabilities": {
                    year: prob.item() for year, prob in zip(self.years, probs[0])
                }
            }
        else:  # regression
            pred_index = logits.squeeze(-1).item()
            pred_index_clipped = np.clip(pred_index, 0, len(self.year_range) - 1)
            predicted_year = self.year_range[int(round(pred_index_clipped))]
            
            return {
                "predicted_year": predicted_year,
                "year_index": pred_index_clipped,
                "raw_prediction": pred_index
            }
    
    def predict_batch(self, texts: list, max_length: int = 256) -> list:
        """
        Predict years for multiple texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
        
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text, max_length) for text in texts]


def load_predictor(model_path: str, task_type: str = "regression", start_year: int = 2006, end_year: int = 2024) -> YearPredictor:
    """
    Convenience function to load a predictor.
    
    Args:
        model_path: Path to saved model directory
        task_type: "classification" or "regression"
        start_year: Starting year (for regression)
        end_year: Ending year (for regression)
    
    Returns:
        YearPredictor instance
    """
    return YearPredictor(model_path, task_type, start_year, end_year)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict year for Reddit comments")
    parser.add_argument("--model_path", type=str, default="./reddit_year_model_regressor",
                       help="Path to saved model directory")
    parser.add_argument("--task_type", type=str, default="regression", choices=["classification", "regression"],
                       help="Task type: classification or regression")
    parser.add_argument("--start_year", type=int, default=2006, help="Start year (for regression)")
    parser.add_argument("--end_year", type=int, default=2024, help="End year (for regression)")
    parser.add_argument("--text", type=str, help="Text to predict (if not provided, uses interactive mode)")
    
    args = parser.parse_args()
    
    # Load predictor
    predictor = load_predictor(
        model_path=args.model_path,
        task_type=args.task_type,
        start_year=args.start_year,
        end_year=args.end_year
    )
    
    # Predict
    if args.text:
        result = predictor.predict(args.text)
        print(f"\nPredicted Year: {result['predicted_year']}")
        if 'confidence' in result:
            print(f"Confidence: {result['confidence']:.2%}")
    else:
        print("\nInteractive mode. Enter Reddit comments (type 'quit' to exit):\n")
        while True:
            text = input("Comment: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if not text:
                continue
            
            result = predictor.predict(text)
            print(f"Predicted Year: {result['predicted_year']}")
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.2%}")
            print()

