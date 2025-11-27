"""
Visualization utilities for model evaluation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6)
):
    """
    Plot confusion matrix for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (e.g., ['2013', '2024'])
        output_path: Path to save figure (if None, displays)
        title: Plot title
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Year', fontsize=12)
    plt.xlabel('Predicted Year', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_vs_actual(
    y_true: List[float],
    y_pred: List[float],
    output_path: Optional[str] = None,
    title: str = "Predicted vs Actual Year",
    figsize: tuple = (10, 8),
    year_range: Optional[List[int]] = None
):
    """
    Plot predicted vs actual years for regression.
    
    Args:
        y_true: True year values
        y_pred: Predicted year values
        output_path: Path to save figure (if None, displays)
        title: Plot title
        figsize: Figure size
        year_range: Optional list of years for axis limits
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    if year_range:
        min_year, max_year = min(year_range), max(year_range)
    else:
        min_year, max_year = int(min(min(y_true), min(y_pred))), int(max(max(y_true), max(y_pred)))
    
    ax1.plot([min_year, max_year], [min_year, max_year], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Year', fontsize=12)
    ax1.set_ylabel('Predicted Year', fontsize=12)
    ax1.set_title(f"{title}\nMAE: {mae:.2f} years, RMSE: {rmse:.2f} years", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2 = axes[1]
    residuals = y_pred - y_true
    ax2.scatter(y_true, residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Actual Year', fontsize=12)
    ax2.set_ylabel('Residual (Predicted - Actual)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_distribution(
    y_true: List[float],
    y_pred: List[float],
    output_path: Optional[str] = None,
    title: str = "Prediction Error Distribution",
    figsize: tuple = (10, 6)
):
    """
    Plot distribution of prediction errors.
    
    Args:
        y_true: True year values
        y_pred: Predicted year values
        output_path: Path to save figure (if None, displays)
        title: Plot title
        figsize: Figure size
    """
    errors = np.array(y_pred) - np.array(y_true)
    
    plt.figure(figsize=figsize)
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.axvline(x=np.mean(errors), color='g', linestyle='--', lw=2, label=f'Mean Error: {np.mean(errors):.2f}')
    plt.xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_temporal_accuracy(
    y_true: List[float],
    y_pred: List[float],
    output_path: Optional[str] = None,
    title: str = "Accuracy by Year",
    figsize: tuple = (12, 6),
    window_size: int = 1
):
    """
    Plot accuracy metrics over time.
    
    Args:
        y_true: True year values
        y_pred: Predicted year values
        output_path: Path to save figure (if None, displays)
        title: Plot title
        figsize: Figure size
        window_size: Window size for calculating off-by-N accuracy
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get unique years
    unique_years = sorted(set(y_true))
    
    # Calculate metrics per year
    mae_per_year = []
    accuracy_per_year = []
    
    for year in unique_years:
        mask = y_true == year
        if np.sum(mask) > 0:
            year_true = y_true[mask]
            year_pred = y_pred[mask]
            mae_per_year.append(np.mean(np.abs(year_pred - year_true)))
            accuracy_per_year.append(np.mean(np.abs(year_pred - year_true) <= window_size))
        else:
            mae_per_year.append(0)
            accuracy_per_year.append(0)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # MAE per year
    ax1 = axes[0]
    ax1.plot(unique_years, mae_per_year, marker='o', linewidth=2, markersize=6)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error', fontsize=12)
    ax1.set_title(f'MAE by Year', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy per year
    ax2 = axes[1]
    ax2.plot(unique_years, accuracy_per_year, marker='o', linewidth=2, markersize=6, color='green')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel(f'Off-by-{window_size} Accuracy', fontsize=12)
    ax2.set_title(f'Accuracy by Year (within {window_size} year)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Temporal accuracy plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_evaluation_plots(
    y_true: List,
    y_pred: List,
    task_type: str = "regression",
    output_dir: str = "./plots",
    class_names: Optional[List[str]] = None,
    year_range: Optional[List[int]] = None
):
    """
    Create all evaluation plots.
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        task_type: "classification" or "regression"
        output_dir: Directory to save plots
        class_names: Class names for classification
        year_range: Year range for regression
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if task_type == "classification":
        if class_names is None:
            class_names = [str(i) for i in sorted(set(y_true))]
        
        plot_confusion_matrix(
            y_true, y_pred, class_names,
            output_path=os.path.join(output_dir, "confusion_matrix.png")
        )
    else:  # regression
        plot_prediction_vs_actual(
            y_true, y_pred,
            output_path=os.path.join(output_dir, "prediction_vs_actual.png"),
            year_range=year_range
        )
        plot_error_distribution(
            y_true, y_pred,
            output_path=os.path.join(output_dir, "error_distribution.png")
        )
        plot_temporal_accuracy(
            y_true, y_pred,
            output_path=os.path.join(output_dir, "temporal_accuracy.png")
        )
    
    print(f"\nAll plots saved to {output_dir}/")

