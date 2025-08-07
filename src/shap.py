"""SHAP Analysis Utilities.

This module provides functions to compute and visualize SHAP values for different model types.
Supports TreeExplainer (e.g., RandomForest, XGBoost) and DeepExplainer (for small DNNs).
"""

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.inspection import PartialDependenceDisplay
logger = logging.getLogger(__name__)

def run_shap_analysis(model, X, feature_names, model_name, period_name, output_dir="results/figures"):
    """
    Compute SHAP values and generate summary plots for a given model and dataset.

    Supports tree-based models and DNNs (with limitations).

    Args:
        model: Trained machine learning model.
        X (np.ndarray): Feature matrix (scaled if applicable).
        feature_names (List[str]): List of feature names corresponding to X.
        model_name (str): Model identifier (e.g., "RandomForest", "DNN").
        period_name (str): Name of the time period for analysis (e.g., "Mai–Oktober").
        output_dir (str): Directory where outputs (plots and CSVs) will be saved.

    Returns:
        pd.Series or None: Series with feature importance (only for tree-based models).
    """
    os.makedirs(output_dir, exist_ok=True)

    if model_name in ["RandomForest", "XGBoost", "DecisionTree", "AdaBoost"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # Feature importance as a CSV
        feature_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)
        csv_path = os.path.join(output_dir, f"{model_name}_{period_name}_shap_importance.csv")
        feature_importance.to_csv(csv_path)
        logger.info(f"SHAP feature importance saved: {csv_path}")

        # SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        fig_path = os.path.join(output_dir, f"{model_name}_{period_name}_shap_summary.png")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"SHAP plot saved: {fig_path}")

        return feature_importance

    elif model_name == "DNN":
        # Note: DeepExplainer only works for small datasets and with TF/keras models
        try:
            explainer = shap.DeepExplainer(model, X[:100])
            shap_values = explainer.shap_values(X[:100])
            plt.figure()
            shap.summary_plot(shap_values, X[:100], feature_names=feature_names, show=False)
            fig_path = os.path.join(output_dir, f"{model_name}_{period_name}_shap_summary.png")
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            logger.info(f"SHAP plot for DNN saved: {fig_path}")
        except Exception as e:
            logger.warning(f"SHAP DeepExplainer failed: {e}")

    else:
        logger.info(f"SHAP analysis not implemented for model type '{model_name}'.")

    return None


def run_shap_analysis_selected_features(
    model, X_full, feature_names_full, selected_features,
    model_name, period_name, output_dir="results/figures/shap"
):
    """Run SHAP analysis for a subset of selected features.

    Args:
        model: Trained model (tree-based).
        X_full (np.ndarray): Full feature matrix.
        feature_names_full (List[str]): All feature names in correct order.
        selected_features (List[str]): Subset of features for which SHAP should be computed.
        model_name (str): Name of the model (e.g., "RandomForest").
        period_name (str): Time period label (e.g., "Mai–Oktober").
        output_dir (str): Directory to save output files.

    Returns:
        pd.Series or None: SHAP importances for selected features.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        if model_name in ["RandomForest", "XGBoost", "DecisionTree", "AdaBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_full)

            # Filter selected features
            sel_indices = [feature_names_full.index(f) for f in selected_features if f in feature_names_full]
            sel_names = [f for f in selected_features if f in feature_names_full]

            # Plot
            plt.figure()
            shap.summary_plot(
                shap_values[:, sel_indices],
                X_full[:, sel_indices],
                feature_names=sel_names,
                show=False
            )
            fig_path = os.path.join(output_dir, f"{model_name}_{period_name}_shap_summary_ops.png")
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            logger.info(f"SHAP selected feature plot saved: {fig_path}")

            # Save importance CSV
            importances = np.abs(shap_values).mean(axis=0)
            all_importance = pd.Series(importances, index=feature_names_full)
            op_importance = all_importance.loc[sel_names]
            csv_path = os.path.join(output_dir, f"{model_name}_{period_name}_shap_importance_ops.csv")
            op_importance.to_csv(csv_path)
            return op_importance

        else:
            logger.info(f"SHAP selected feature analysis skipped for model '{model_name}'.")
            return None

    except Exception as e:
        logger.warning(f"SHAP selected feature analysis failed for '{model_name}': {e}")
        return None

def plot_pdp_sklearn(model, X, feature, feature_names, model_name, period_name, output_dir="results/figures/pdp"):
    """Plot a partial dependence plot (PDP) for a given feature using scikit-learn.

    Args:
        model: Trained scikit-learn-compatible model.
        X (np.ndarray): Feature matrix.
        feature (str): Name of the feature for which the PDP should be generated.
        feature_names (List[str]): Names of all features.
        model_name (str): Name of the model (e.g., "XGBoost").
        period_name (str): Time period for which the plot is generated.
        output_dir (str): Directory to save the PDP plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    feature_idx = feature_names.index(feature)

    fig, ax = plt.subplots()
    PartialDependenceDisplay.from_estimator(
        model, X, [feature_idx], feature_names=feature_names, ax=ax
    )

    pdp_path = os.path.join(output_dir, f"{model_name}_{period_name}_pdp_{feature}.png")
    plt.savefig(pdp_path)
    plt.close(fig)
    print(f"PDP für Feature {feature} gespeichert: {pdp_path}")