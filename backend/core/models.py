"""
Model definitions and PyTorch dataset classes.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    BertForSequenceClassification
)


class RedditDataset(Dataset):
    """PyTorch Dataset for Reddit comment classification/regression."""
    
    def __init__(self, encodings: Dict, labels: List, is_regression: bool = False):
        """
        Args:
            encodings: Tokenizer encodings dictionary
            labels: List of labels (integers for classification, floats for regression)
            is_regression: Whether this is a regression task
        """
        self.encodings = encodings
        self.labels = labels
        self.is_regression = is_regression
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        
        if self.is_regression:
            item['labels'] = torch.tensor(float(self.labels[idx]), dtype=torch.float)
        else:
            item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        
        return item


def load_classification_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    device: Optional[str] = None
) -> tuple:
    """
    Load a model and tokenizer for classification.
    
    Args:
        model_name: HuggingFace model name
        num_labels: Number of classification classes
        device: Device to load model on ('cuda' or 'cpu'). If None, auto-detects.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use BertForSequenceClassification for DistilBERT compatibility
    if "distilbert" in model_name.lower():
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    print(f"Model loaded on device: {device}")
    
    return model, tokenizer


def load_regression_model(
    model_name: str = "roberta-base",
    start_year: int = 2006,
    end_year: int = 2024,
    device: Optional[str] = None
) -> tuple:
    """
    Load a model and tokenizer for regression.
    
    Args:
        model_name: HuggingFace model name
        start_year: Starting year (for label range)
        end_year: Ending year (for label range)
        device: Device to load model on ('cuda' or 'cpu'). If None, auto-detects.
    
    Returns:
        Tuple of (model, tokenizer, year_range)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1
    config.problem_type = "regression"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    print(f"Model loaded on device: {device}")
    
    year_range = list(range(start_year, end_year + 1))
    
    return model, tokenizer, year_range

