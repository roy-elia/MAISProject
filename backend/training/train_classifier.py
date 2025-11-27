"""
Train a binary/multi-class classifier to predict Reddit comment year.
"""

import os
import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import (
    load_reddit_data,
    prepare_classification_labels,
    split_data,
    tokenize_texts
)
from core.models import RedditDataset, load_classification_model


def compute_classification_metrics(eval_pred):
    """Compute metrics for classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "macro_f1": f1}


def train_classifier(
    data_dir: str = "data/sampled_comments",
    years: list = [2013, 2024],
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./backend/models/reddit_year_model_classifier",
    max_samples_per_month: int = 1000,
    max_length: int = 128,
    num_epochs: int = 3,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    random_state: int = 42
):
    """
    Train a classifier to predict Reddit comment year.
    
    Args:
        data_dir: Directory containing CSV files
        years: List of years to classify
        model_name: HuggingFace model name
        output_dir: Directory to save model
        max_samples_per_month: Max samples per month file
        max_length: Maximum sequence length
        num_epochs: Number of training epochs
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        random_state: Random seed
    """
    print("="*60)
    print("Training Reddit Comment Year Classifier")
    print("="*60)
    print(f"Years: {years}")
    print(f"Model: {model_name}\n")
    
    # Load data
    print("Loading data...")
    df = load_reddit_data(
        data_dir=data_dir,
        specific_years=years,
        max_samples_per_month=max_samples_per_month
    )
    
    # Prepare labels
    df = prepare_classification_labels(df, years)
    print(f"Total samples: {len(df):,}\n")
    
    # Split data
    train_df, val_df, test_df = split_data(
        df,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        stratify_col='label',
        random_state=random_state
    )
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_classification_model(
        model_name=model_name,
        num_labels=len(years)
    )
    
    # Tokenize
    print("Tokenizing texts...")
    train_texts = train_df['body'].tolist()
    val_texts = val_df['body'].tolist()
    test_texts = test_df['body'].tolist()
    
    train_enc = tokenize_texts(train_texts, tokenizer, max_length=max_length, padding='max_length')
    val_enc = tokenize_texts(val_texts, tokenizer, max_length=max_length, padding='max_length')
    test_enc = tokenize_texts(test_texts, tokenizer, max_length=max_length, padding='max_length')
    
    # Create datasets
    train_dataset = RedditDataset(train_enc, train_df['label'].tolist(), is_regression=False)
    val_dataset = RedditDataset(val_enc, val_df['label'].tolist(), is_regression=False)
    test_dataset = RedditDataset(test_enc, test_df['label'].tolist(), is_regression=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        seed=random_state
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_classification_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    
    print(f"\nAccuracy: {accuracy_score(test_df['label'], preds):.4f}")
    print(f"Macro F1: {f1_score(test_df['label'], preds, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(
        test_df['label'],
        preds,
        target_names=[str(y) for y in years]
    ))
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        from core.visualize import plot_confusion_matrix
        os.makedirs("../frontend/public", exist_ok=True)
        plot_confusion_matrix(
            y_true=test_df['label'].values,
            y_pred=preds,
            class_names=[str(y) for y in years],
            output_path="../frontend/public/confusion-matrix.png",
            title="Confusion Matrix: Classification (2013 vs 2024)"
        )
        print("Confusion matrix saved to frontend/public/confusion-matrix.png")
    except Exception as e:
        print(f"Note: Could not generate visualizations: {e}")
        print("You can generate them manually using src/visualize.py")
    
    print("Training complete!")
    
    return trainer, test_df, preds


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Reddit comment year classifier")
    parser.add_argument("--data_dir", type=str, default="data/sampled_comments",
                       help="Directory containing CSV files")
    parser.add_argument("--years", type=int, nargs="+", default=[2013, 2024],
                       help="Years to classify")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                       help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="./reddit_year_model_classifier",
                       help="Output directory for model")
    parser.add_argument("--max_samples_per_month", type=int, default=1000,
                       help="Max samples per month file")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    
    args = parser.parse_args()
    
    train_classifier(
        data_dir=args.data_dir,
        years=args.years,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_samples_per_month=args.max_samples_per_month,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )

