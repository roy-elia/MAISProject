"""
Train a regression model to predict Reddit comment year (2006-2024).
"""

import os
import sys
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import (
    load_reddit_data,
    prepare_regression_labels,
    split_data,
    tokenize_texts
)
from core.models import RedditDataset, load_regression_model


def compute_regression_metrics(eval_pred, year_range):
    """Compute metrics for regression."""
    logits, labels = eval_pred
    preds = logits.squeeze(-1)  # shape (batch,)
    preds_clipped = np.clip(preds, 0, len(year_range) - 1)
    
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)
    
    off_by_1 = np.mean(np.abs(preds_clipped - labels) <= 1)
    off_by_2 = np.mean(np.abs(preds_clipped - labels) <= 2)
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "off_by_1_acc": off_by_1,
        "off_by_2_acc": off_by_2
    }


def train_regressor(
    data_dir: str = "data/sampled_comments",
    start_year: int = 2006,
    end_year: int = 2024,
    model_name: str = "roberta-base",
    output_dir: str = "./backend/models/reddit_year_model_regressor",
    max_samples_per_month: int = 10,
    max_length: int = 256,
    num_epochs: int = 3,
    batch_size: int = 8,
    eval_batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    random_state: int = 42
):
    """
    Train a regressor to predict Reddit comment year.
    
    Args:
        data_dir: Directory containing CSV files
        start_year: Starting year (inclusive)
        end_year: Ending year (inclusive)
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
    print("Training Reddit Comment Year Regressor")
    print("="*60)
    print(f"Year range: {start_year}-{end_year}")
    print(f"Model: {model_name}\n")
    
    # Load data
    print("Loading data...")
    df = load_reddit_data(
        data_dir=data_dir,
        start_year=start_year,
        end_year=end_year,
        max_samples_per_month=max_samples_per_month
    )
    
    # Prepare labels
    df = prepare_regression_labels(df, start_year, end_year)
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
    model, tokenizer, year_range = load_regression_model(
        model_name=model_name,
        start_year=start_year,
        end_year=end_year
    )
    
    # Tokenize
    print("Tokenizing texts...")
    train_texts = train_df['body'].tolist()
    val_texts = val_df['body'].tolist()
    test_texts = test_df['body'].tolist()
    
    train_enc = tokenize_texts(train_texts, tokenizer, max_length=max_length, padding='longest')
    val_enc = tokenize_texts(val_texts, tokenizer, max_length=max_length, padding='longest')
    test_enc = tokenize_texts(test_texts, tokenizer, max_length=max_length, padding='longest')
    
    # Create datasets
    train_dataset = RedditDataset(train_enc, train_df['label'].tolist(), is_regression=True)
    val_dataset = RedditDataset(val_enc, val_df['label'].tolist(), is_regression=True)
    test_dataset = RedditDataset(test_enc, test_df['label'].tolist(), is_regression=True)
    
    # Create metrics function with year_range
    def compute_metrics(eval_pred):
        return compute_regression_metrics(eval_pred, year_range)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
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
        compute_metrics=compute_metrics,
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
    preds = preds_output.predictions.squeeze(-1)
    preds_clipped = np.clip(preds, 0, len(year_range) - 1)
    
    # Convert back to actual years
    actual_years = test_df['year'].values
    predicted_years = [year_range[int(round(p))] for p in preds_clipped]
    
    print(f"\nMAE:  {mean_absolute_error(test_df['label'], preds_clipped):.3f} years")
    print(f"RMSE: {np.sqrt(mean_squared_error(test_df['label'], preds_clipped)):.3f} years")
    print(f"R²:   {r2_score(test_df['label'], preds_clipped):.3f}")
    
    off_by_1 = np.mean(np.abs(preds_clipped - test_df['label'].values) <= 1)
    off_by_2 = np.mean(np.abs(preds_clipped - test_df['label'].values) <= 2)
    print(f"\nOff-by-1 year accuracy: {off_by_1:.1%}")
    print(f"Off-by-2 year accuracy: {off_by_2:.1%}")
    
    # Sample predictions
    print("\n" + "="*60)
    print("Sample Predictions (first 10)")
    print("="*60)
    for i in range(min(10, len(test_df))):
        actual = actual_years[i]
        pred = predicted_years[i]
        text = test_texts[i][:80] + "..." if len(test_texts[i]) > 80 else test_texts[i]
        print(f"\nActual: {actual} | Predicted: {pred}")
        print(f"Text: {text}")
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        from core.visualize import create_evaluation_plots
        os.makedirs("../frontend/public", exist_ok=True)
        create_evaluation_plots(
            y_true=actual_years,
            y_pred=predicted_years,
            task_type="regression",
            output_dir="../frontend/public",
            year_range=year_range
        )
        print("Visualizations saved to frontend/public/")
    except Exception as e:
        print(f"Note: Could not generate visualizations: {e}")
        print("You can generate them manually using src/visualize.py")
    
    print("Training complete!")
    
    return trainer, test_df, preds_clipped, year_range, predicted_years


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Reddit comment year regressor")
    parser.add_argument("--data_dir", type=str, default="data/sampled_comments",
                       help="Directory containing CSV files")
    parser.add_argument("--start_year", type=int, default=2006,
                       help="Starting year")
    parser.add_argument("--end_year", type=int, default=2024,
                       help="Ending year")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                       help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="./reddit_year_model_regressor",
                       help="Output directory for model")
    parser.add_argument("--max_samples_per_month", type=int, default=10,
                       help="Max samples per month file")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    
    args = parser.parse_args()
    
    train_regressor(
        data_dir=args.data_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_samples_per_month=args.max_samples_per_month,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )

