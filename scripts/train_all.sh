#!/bin/bash
# Script to train both classification and regression models

echo "Training Classification Model (2013 vs 2024)..."
python -m src.train_classifier \
    --years 2013 2024 \
    --model_name distilbert-base-uncased \
    --max_samples_per_month 1000 \
    --output_dir ./backend/models/reddit_year_model_classifier

echo ""
echo "Training Regression Model (2006-2024)..."
python -m src.train_regressor \
    --start_year 2006 \
    --end_year 2024 \
    --model_name roberta-base \
    --max_samples_per_month 10 \
    --output_dir ./backend/models/reddit_year_model_regressor

echo ""
echo "Training complete!"

