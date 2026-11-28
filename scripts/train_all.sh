#!/bin/bash
# Script to train both classification and regression models

echo "Training Classification Model (2013 vs 2024)..."
python -m backend.training.train_classifier \
    --years 2013 2024 \
    --model_name distilbert-base-uncased \
    --max_samples_per_month 1000 \
    --output_dir ./backend/models/reddit_year_model_classifier

echo ""
echo "Training Regression Model (2006-2024)..."
# Needs a GPU. Writes to output/regressor, never to the served model folders
# (see backend/README.md).
python -m backend.training.train_regressor

echo ""
echo "Training complete!"

