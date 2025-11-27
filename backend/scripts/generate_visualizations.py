"""
Script to generate all visualizations for the project website.
Run this after training your models to create all visualization images.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.visualize import (
    plot_confusion_matrix,
    plot_prediction_vs_actual,
    plot_error_distribution,
    plot_temporal_accuracy
)

def generate_classification_visualizations():
    """Generate visualizations for the 2-bin classifier (2008-2010 vs 2020-2022)."""
    print("Generating classification visualizations...")
    print("Load your saved test labels and predictions, then uncomment the code below.")
    print("Example:")
    print("  y_true = test_df['label'].values  # 0 = 2008-2010, 1 = 2020-2022")
    print("  y_pred = predictions")
    print("  plot_confusion_matrix(")
    print("      y_true=y_true,")
    print("      y_pred=y_pred,")
    print(\"      class_names=['2008-2010', '2020-2022'],\")
    print(\"      output_path='frontend/public/confusion-matrix.png',\")
    print(\"      title='Confusion Matrix: 2-Bin Classification'\" )
    print("  )")

def generate_regression_visualizations():
    """Generate visualizations for regression model."""
    print("Generating regression visualizations...")
    
    # Load your regression results here
    # Example: Load from saved predictions or test results
    # y_true = [2006, 2007, 2008, ...]  # actual years
    # y_pred = [2007, 2008, 2009, ...]  # predicted years
    
    # For now, this is a template - you'll need to load your actual results
    print("Note: Load your regression test results to generate plots")
    print("Example:")
    print("  y_true = test_df['year'].values")
    print("  y_pred = predicted_years")
    print("  plot_prediction_vs_actual(y_true, y_pred, 'frontend/public/prediction-vs-actual.png')")
    print("  plot_error_distribution(y_true, y_pred, 'frontend/public/error-distribution.png')")
    print("  plot_temporal_accuracy(y_true, y_pred, 'frontend/public/temporal-accuracy.png')")

if __name__ == "__main__":
    print("="*60)
    print("Visualization Generator")
    print("="*60)
    print("\nThis script helps generate visualizations for your project.")
    print("You'll need to load your model predictions first.\n")
    
    # Create output directory
    os.makedirs("frontend/public", exist_ok=True)
    
    print("To generate visualizations:")
    print("1. Train your models (classification and regression)")
    print("2. Save predictions from test set")
    print("3. Load predictions in this script")
    print("4. Run the visualization functions")
    print("\nSee src/visualize.py for available functions.")
    print("\nOutput directory: frontend/public/")

