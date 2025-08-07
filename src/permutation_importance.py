"""Permutation importance analysis for trained regression models.

This module computes and visualizes the importance of features by measuring the
impact of permuting each feature on model performance.
"""

import os
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def run_permutation_importance(
    model, X, y, feature_names, model_name, period_name, output_dir="results/figures/feature importance", scoring="r2"
):
    """
    Computes and saves permutation importance for a fitted model.

    Args:
        model: Fitted model object (must have predict()).
        X: Test set features (array or DataFrame, scaled if required).
        y: Test set targets.
        feature_names: List of feature names (column order must match X).
        model_name: Model name (str).
        period_name: Period label (str).
        output_dir: Where to save plot and table.
        scoring: Scoring function (default: "r2").

    Returns:
        pd.Series with permutation importances.
    """
    os.makedirs(output_dir, exist_ok=True)
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, scoring=scoring, n_jobs=-1
    )
    importances = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)

    # Save table
    csv_path = os.path.join(output_dir, f"{model_name}_{period_name}_permutation_importance.csv")
    importances.to_csv(csv_path)
    logger.info(f"Permutation importance saved: {csv_path}")

    # Plot
    plt.figure(figsize=(7, 0.5 + 0.4*len(feature_names)))
    importances.iloc[:20].plot(kind="barh")  # top 20 or fewer
    plt.xlabel(f"Mean Importance (scoring: {scoring})")
    plt.title(f"Permutation Importance: {model_name} ({period_name})")
    plt.gca().invert_yaxis()
    fig_path = os.path.join(output_dir, f"{model_name}_{period_name}_permutation_importance.png")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Permutation importance plot saved: {fig_path}")

    return importances