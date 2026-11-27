"""
Write a half-precision copy of a trained model: the copy the API serves.

    python -m backend.training.export_fp16 output/regressor/final output/regressor/fp16
"""

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def export_fp16(source_dir: str, target_dir: str) -> None:
    """Save the model and tokenizer in `source_dir` to `target_dir` as float16."""
    model = AutoModelForSequenceClassification.from_pretrained(source_dir, dtype=torch.float16)
    model.save_pretrained(target_dir)
    AutoTokenizer.from_pretrained(source_dir).save_pretrained(target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a half-precision copy of a model")
    parser.add_argument("source_dir", help="Directory of the trained full-precision model")
    parser.add_argument("target_dir", help="Directory to write the float16 copy to")
    args = parser.parse_args()
    export_fp16(args.source_dir, args.target_dir)
    print(f"Half-precision model written to {args.target_dir}")
