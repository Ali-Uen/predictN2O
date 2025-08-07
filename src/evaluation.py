"""Evaluation utilities for regression models.

Includes standard regression metrics and basic visualization functions:
- R², RMSE, MAE
- Residual plots
- True vs. Predicted plots
"""

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

def evaluate_regression(y_true, y_pred):
    """
    Computes standard regression metrics and returns them as a dictionary.

    Args:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted values.

    Returns:
        dict: Dictionary containing R2, RMSE, and MAE.
    """
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

def print_metrics(metrics, prefix=""):
    """
    Prints regression metrics in a nicely formatted way.

    Args:
        metrics: Dictionary of metric names and values.
        prefix: Optional prefix string for each line.
    """
    for key, value in metrics.items():
        print(f"{prefix}{key}: {value:.3f}")

def plot_residuals(y_true, y_pred, title=None):
    """
    Plots the residuals (difference between true and predicted values).
    Args:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted values.
        title: Plot title (optional).
    """ 
    residuals = y_true - y_pred
    plt.figure(figsize=(10,4))
    plt.plot(residuals)
    plt.title(title or "Residuen")
    plt.axhline(0, color='red', linestyle='--')


def plot_true_vs_pred(y_true, y_pred, title=None):
    """
    Creates a scatter plot of true versus predicted values.
    Args:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted values.
        title: Plot title (optional).
    """
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title or "True vs. Predicted")