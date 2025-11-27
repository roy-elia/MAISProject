# How to Generate Visualizations

This guide shows you how to create visualization images for your website.

## Quick Steps

1. **After training your models**, you'll have predictions from the test set
2. **Use the visualization functions** to create images
3. **Save images** to `frontend/public/` folder
4. **Images will automatically appear** on the website

## Example: Generate All Visualizations

### For 2-Bin Classification Model (2008-2010 vs 2020-2022)

```python
from src.visualize import plot_confusion_matrix
import numpy as np

# Load your classification results (0 = 2008-2010, 1 = 2020-2022)
y_true = test_df['label'].values
y_pred = predictions

# Generate confusion matrix
plot_confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    class_names=['2008-2010', '2020-2022'],
    output_path='frontend/public/confusion-matrix.png',
    title='Confusion Matrix: 2-Bin Classification'
)
```

### For Regression Model

```python
from src.visualize import (
    plot_prediction_vs_actual,
    plot_error_distribution,
    plot_temporal_accuracy
)

# Load your regression results
# Assuming you have test_df and predicted_years from training
y_true = test_df['year'].values  # Actual years
y_pred = predicted_years  # Predicted years

# Generate all regression visualizations
plot_prediction_vs_actual(
    y_true=y_true,
    y_pred=y_pred,
    output_path='frontend/public/prediction-vs-actual.png',
    year_range=list(range(2006, 2025))
)

plot_error_distribution(
    y_true=y_true,
    y_pred=y_pred,
    output_path='frontend/public/error-distribution.png'
)

plot_temporal_accuracy(
    y_true=y_true,
    y_pred=y_pred,
    output_path='frontend/public/temporal-accuracy.png'
)
```

## Or Use the Convenience Function

```python
from src.visualize import create_evaluation_plots

# For regression
create_evaluation_plots(
    y_true=test_df['year'].values,
    y_pred=predicted_years,
    task_type="regression",
    output_dir="./frontend/public",
    year_range=list(range(2006, 2025))
)
```

## Required Images

Place these in `frontend/public/`:

- ✅ `mcgill-logo.png` - McGill University logo
- ✅ `confusion-matrix.png` - Classification confusion matrix
- ✅ `prediction-vs-actual.png` - Regression predictions plot
- ✅ `error-distribution.png` - Error histogram
- ✅ `temporal-accuracy.png` - Accuracy by year
- ⚪ `word-cloud-*.png` - Optional word clouds

## Image Requirements

- **Format**: PNG (recommended) or JPG
- **Size**: Any size (will be scaled automatically)
- **Aspect Ratio**: 
  - Confusion matrix: Square or 4:3
  - Prediction plots: 16:9 or 4:3
  - Word clouds: Square

## Notes

- If an image doesn't exist, a placeholder will show with instructions
- Images are automatically optimized by Next.js
- Use high-resolution images for best quality (at least 1200px wide for plots)

