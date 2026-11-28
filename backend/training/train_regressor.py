"""
Train a regression model to predict Reddit comment year (2006-2024).

    python -m backend.training.train_regressor --dry_run     # data only, no model
    python -m backend.training.train_regressor               # full run, needs a GPU
    python -m backend.training.train_regressor --resume      # continue a stopped run

Everything is written under --output_dir: checkpoints, the final model, a
half-precision copy for serving and metrics.json. The served model folders are
never touched; copy a model there only if metrics.json says it passed the gate.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_union
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import tokenize_texts
from core.models import RedditDataset, load_regression_model
from core.regression_data import load_regression_splits
from training.export_fp16 import export_fp16

# The comments that exposed the weak model; their scores go into metrics.json.
EXAMPLES = [
    "lol",
    "This meme is fire fr fr no cap",
    "Just upgraded to Windows Vista, loving the new Aero look",
]
EVAL_EVERY_STEPS = 2000


def compute_regression_metrics(eval_pred, max_index: int) -> dict:
    """Score predictions clipped to the year range, as the API serves them."""
    logits, labels = eval_pred
    preds = np.clip(np.asarray(logits).reshape(-1), 0, max_index)
    mse = mean_squared_error(labels, preds)
    errors = np.abs(preds - labels)
    return {
        "mse": float(mse),
        "mae": float(mean_absolute_error(labels, preds)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(labels, preds)),
        "off_by_1_acc": float(np.mean(errors <= 1)),
        "off_by_2_acc": float(np.mean(errors <= 2)),
    }


def build_dataset(df, tokenizer, max_length: int) -> RedditDataset:
    """Tokenize without padding; the collator pads each batch on its own."""
    encodings = tokenize_texts(df["body"].tolist(), tokenizer, max_length=max_length, padding=False)
    return RedditDataset(encodings, df["label"].tolist(), is_regression=True)


def init_output_bias(model, value: float) -> None:
    """Start predictions at the average label so training need not learn it first."""
    with torch.no_grad():
        model.classifier.out_proj.bias.fill_(value)


def build_training_args(
    output_dir: str,
    train_rows: int,
    num_epochs: int = 3,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    random_state: int = 42,
) -> TrainingArguments:
    """Evaluate every 2,000 steps, or every epoch when an epoch is shorter."""
    # batch_size is per GPU, so two GPUs halve the number of steps in an epoch.
    rows_per_step = batch_size * max(1, torch.cuda.device_count())
    interval = min(EVAL_EVERY_STEPS, math.ceil(train_rows / rows_per_step))
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        group_by_length=True,
        eval_strategy="steps",
        eval_steps=interval,
        save_strategy="steps",
        save_steps=interval,
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        seed=random_state,
        report_to="none",
    )


class EvaluateLastStep(TrainerCallback):
    """
    Evaluate and save after the final step.

    Otherwise the steps after the last 2,000-step mark are never scored, and
    reloading the best checkpoint throws them away.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


def build_trainer(model, tokenizer, args, train_dataset, eval_dataset, max_index: int) -> Trainer:
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda eval_pred: compute_regression_metrics(eval_pred, max_index),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), EvaluateLastStep()],
    )


def tfidf_baseline(train_df, test_df, max_index: int) -> np.ndarray:
    """
    Predict the test years with word and character TF-IDF plus ridge regression.

    The transformer has to beat this to be worth deploying, so it is fitted on
    the same training rows. Single precision keeps memory low: 760,000 rows
    peaked under 4 GB.
    """
    shared = dict(sublinear_tf=True, max_features=200_000, dtype=np.float32)
    features = make_union(
        TfidfVectorizer(min_df=2, ngram_range=(1, 2), **shared),
        TfidfVectorizer(analyzer="char_wb", min_df=3, ngram_range=(3, 5), **shared),
    )
    model = Ridge(alpha=2.0, solver="sparse_cg")
    model.fit(features.fit_transform(train_df["body"]), train_df["label"])
    return np.clip(model.predict(features.transform(test_df["body"])), 0, max_index)


def evaluation_report(test_df, raw_preds: np.ndarray, train_mean: float, max_index: int) -> dict:
    """Test scores overall, by year and by comment length."""
    labels = test_df["label"].to_numpy(dtype=float)
    report = compute_regression_metrics((raw_preds, labels), max_index)
    errors = np.abs(np.clip(raw_preds, 0, max_index) - labels)
    lengths = test_df["body"].str.len().to_numpy()
    years = test_df["year"].to_numpy()
    buckets = {
        "under_50_chars": lengths < 50,
        "50_to_300_chars": (lengths >= 50) & (lengths <= 300),
        "over_300_chars": lengths > 300,
    }
    report.update({
        "constant_mae": float(np.abs(labels - train_mean).mean()),
        "mae_by_year": {str(y): float(errors[years == y].mean()) for y in sorted(set(years))},
        "mae_by_length": {name: float(errors[m].mean()) for name, m in buckets.items() if m.any()},
        # A ceiling far below the last year was the symptom of the weak model.
        "prediction_min": float(np.min(raw_preds)),
        "prediction_max": float(np.max(raw_preds)),
    })
    return report


def passes_gate(model_mae: float, baseline_mae: float) -> bool:
    return model_mae < baseline_mae


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Reddit comment year regressor")
    parser.add_argument("--data_dir", default="data/sampled_comments", help="Directory containing CSV files")
    parser.add_argument("--output_dir", default="output/regressor", help="Where checkpoints, models and metrics go")
    parser.add_argument("--start_year", type=int, default=2006)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--model_name", default="roberta-base", help="HuggingFace model name")
    parser.add_argument("--train_per_year", type=int, default=40_000, help="Training comments per year")
    parser.add_argument("--eval_per_year", type=int, default=500, help="Validation and test comments per year")
    parser.add_argument("--val_month", type=int, default=3, help="Month held out for validation")
    parser.add_argument("--test_month", type=int, default=9, help="Month held out for test")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length (the API uses 256)")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the latest checkpoint in --output_dir; keep the other settings the same")
    parser.add_argument("--dry_run", action="store_true", help="Prepare the data, print a summary and stop")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    max_index = args.end_year - args.start_year

    print("Loading data...")
    train_df, val_df, test_df = load_regression_splits(
        data_dir=args.data_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        train_per_year=args.train_per_year,
        eval_per_year=args.eval_per_year,
        val_month=args.val_month,
        test_month=args.test_month,
        seed=args.seed,
    )
    rows = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    print(f"Rows: {rows}")
    if args.dry_run:
        print("Dry run: no model was loaded, nothing was trained or written.")
        return

    train_mean = float(train_df["label"].mean())
    print("Fitting the TF-IDF baseline (several minutes on the full data)...")
    baseline = tfidf_baseline(train_df, test_df, max_index)
    baseline_mae = float(mean_absolute_error(test_df["label"], baseline))
    print(f"Baseline test MAE: {baseline_mae:.3f} years")

    print("Loading model...")
    model, tokenizer, _ = load_regression_model(
        model_name=args.model_name, start_year=args.start_year, end_year=args.end_year
    )
    init_output_bias(model, train_mean)

    print("Tokenizing texts...")
    train_data, val_data, test_data = (
        build_dataset(df, tokenizer, args.max_length) for df in (train_df, val_df, test_df)
    )
    training_args = build_training_args(
        args.output_dir, len(train_df), num_epochs=args.num_epochs,
        batch_size=args.batch_size, random_state=args.seed,
    )
    trainer = build_trainer(model, tokenizer, training_args, train_data, val_data, max_index)

    checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        checkpoint = get_last_checkpoint(args.output_dir)
    print(f"Resuming from {checkpoint}..." if checkpoint else "Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save before evaluating, so a later failure cannot lose the trained model.
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    export_fp16(final_dir, os.path.join(args.output_dir, "fp16"))

    print("Evaluating on the test months...")
    raw = trainer.predict(test_data).predictions.reshape(-1)
    report = evaluation_report(test_df, raw, train_mean, max_index)

    examples = build_dataset(pd.DataFrame({"body": EXAMPLES, "label": 0}), tokenizer, args.max_length)
    example_preds = trainer.predict(examples).predictions.reshape(-1)

    passed = passes_gate(report["mae"], baseline_mae)
    metrics = {
        "settings": vars(args),
        "rows": rows,
        "model": report,
        "baseline_tfidf_mae": baseline_mae,
        "passes_gate": passed,
        "examples": {text: float(p) for text, p in zip(EXAMPLES, example_preds)},
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nModel test MAE {report['mae']:.3f} | TF-IDF {baseline_mae:.3f} | constant {report['constant_mae']:.3f}")
    print("PASSED the gate." if passed else "FAILED the gate: do not deploy this model.")
    print(f"Results are in {args.output_dir}")


if __name__ == "__main__":
    main()
