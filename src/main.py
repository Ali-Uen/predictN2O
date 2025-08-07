"""Main script to train and evaluate N2O emission models.

This script loads data, preprocesses features, performs augmentation, trains models,
evaluates performance, saves results, and allows flexible configuration via command line.

Usage examples:
    python main.py
    python main.py --model RandomForest --split 0.7 --augment 4 --noise 0.2
"""
import argparse
import os
import numpy as np
import pandas as pd
import time

from src.data_loader import load_and_prepare_data
from src.feature_engineering import preprocess_features, model_period_lags, model_period_rolling
from src.models import train_model, save_model, append_results_to_csv, train_dnn_model
from src.param_grids import param_grids
from src.resampling import resampling
from src.evaluation import evaluate_regression, print_metrics
from src.augmentation import add_noise
from src.config import (
    DATA_PATH, PERIODS, DEFAULT_MODEL, DEFAULT_SPLIT_RATIO, RANDOM_SEED,
    AUGMENTATION_N, AUGMENTATION_NOISE, RESULTS_DIR, SAVE_RESULTS
)
from src.shap import run_shap_analysis_selected_features
from src.permutation_importance import run_permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments to override config values.

    Returns:
        argparse.Namespace: Parsed arguments with possible overrides.
    """
    parser = argparse.ArgumentParser(description="N2O model pipeline")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Name of the model (XGBoost, RandomForest, KNN)")
    parser.add_argument("--split", type=float, default=DEFAULT_SPLIT_RATIO,
                        help="Train/test split ratio")
    parser.add_argument("--augment", type=int, default=AUGMENTATION_N,
                        help="Number of augmented samples per original sample")
    parser.add_argument("--noise", type=float, default=AUGMENTATION_NOISE,
                        help="Noise level to apply during augmentation")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def split_train_test_time(df: pd.DataFrame, split_ratio: float = 0.8):
    """Splits a DataFrame into train and test by time.

    Args:
        df: Input DataFrame, sorted by time.
        split_ratio: Fraction for the train set.

    Returns:
        Tuple of (train_df, test_df)
    """
    n = len(df)
    split_idx = int(n * split_ratio)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test

def get_period_mask(df: pd.DataFrame, start_mmdd: str, end_mmdd: str):
    """Creates a boolean mask for a period in the year.

    Args:
        df: DataFrame with a TIME column.
        start_mmdd: Start date as MM-DD string.
        end_mmdd: End date as MM-DD string.

    Returns:
        Boolean mask for selection.
    """
    timestr = df['TIME'].dt.strftime('%m-%d')
    return (timestr >= start_mmdd) & (timestr <= end_mmdd)

def main():
    """Main pipeline: load data, train models on time periods, evaluate and save results.

    Parses command line arguments, loads and filters data by time periods,
    applies feature engineering, augmentation, trains model with cross-validation,
    evaluates on train and test sets, saves predictions and models, and logs metrics.
    """
    args = parse_args()
    np.random.seed(args.seed)

    model_name = args.model
    split_ratio = args.split
    n_augment = args.augment
    noise_level = args.noise

    logger.info(f"\n*** Current Parameters: Model={model_name} | Split={split_ratio} | Augment={n_augment} | Noise={noise_level} | Seed={args.seed}\n")

    results_list = []

    # Load the full dataset and apply initial preprocessing.
    df_full = load_and_prepare_data(DATA_PATH)

    for name, start_mmdd, end_mmdd in PERIODS:
        logger.info(f"\n=== {name} ({start_mmdd} bis {end_mmdd}) ===")

        # Filter the dataset to only include the current time period.
        mask = get_period_mask(df_full, start_mmdd, end_mmdd)
        df_bereich = df_full[mask].copy()

        # Resample the data according to the model and period configuration.
        resample_rule = resampling[model_name][name]
        if resample_rule is not None:
            df_period = df_bereich.resample(resample_rule, on="TIME").mean().reset_index()
        else:
            df_period = df_bereich.copy()

        # Skip this period if the dataset is too small to be reliable.
        if len(df_period) < 50:
            logger.warning("Zu wenig Daten – überspringe...")
            continue

        # Train/test split by time
        df_train, df_test = split_train_test_time(df_period, split_ratio)
        if len(df_test) < 10:
            logger.warning("Zu wenig Testdaten – überspringe...")
            continue

        # Generate lag and rolling statistics as features for time-series modeling.
        my_lags = model_period_lags[model_name][name]
        my_rolling = model_period_rolling[model_name][name]
        df_train_proc = preprocess_features(df_train, lags_dict=my_lags, rolling_dict=my_rolling)
        df_test_proc = preprocess_features(df_test, lags_dict=my_lags, rolling_dict=my_rolling)

        # Select all columns except time and target as features
        feature_cols = [col for col in df_train_proc.columns if col not in ["TIME", "N2O"]]
        print("Features (alle Spalten außer TIME & N2O):", feature_cols)

        X_train = df_train_proc[feature_cols]
        y_train = df_train_proc['N2O']
        X_test = df_test_proc[feature_cols]
        y_test = df_test_proc['N2O']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Augment training data by adding Gaussian noise to increase robustness.
        X_train_aug, y_train_aug = add_noise(
            X_train_scaled, y_train.values,
            n_augment=n_augment,
            noise_level=noise_level,
            random_state=args.seed
        )

        # Train the model using cross-validation and select the best hyperparameters.
        train_start = time.time()
        param_grid = param_grids[model_name][name]

        if model_name == "DNN":
            dnn_param_dict = {k: v[0] if isinstance(v, list) else v for k, v in param_grid.items()}
            model, history, best_params = train_dnn_model(
                X_train_aug, y_train_aug, dnn_param_dict
            )
        else:
            model, best_params = train_model(
                X_train_aug, y_train_aug, model_name,
                param_grid=param_grid,
                cv=TimeSeriesSplit(n_splits=5),
                scoring='r2'
            )
        train_duration = time.time() - train_start

        # Make predictions on the test set and measure the time taken.
        predict_start = time.time()
        if model_name == "DNN":
            y_test_pred = model.predict(X_test_scaled).flatten()
            prediction_duration = time.time() - predict_start
            y_train_pred = model.predict(X_train_scaled).flatten()
        else:
            y_test_pred = model.predict(X_test_scaled)
            prediction_duration = time.time() - predict_start
            y_train_pred = model.predict(X_train_scaled)
        """ 
        # SHAP analysis
        operation_cols = ["T", "DO", "Q_in"]
        run_shap_analysis_selected_features(
            model=model,
            X_full=X_test_scaled,
            feature_names_full=feature_cols,
            selected_features=operation_cols,
            model_name=model_name,
            period_name=name
        )

        # Permutation Importance
        importances = run_permutation_importance(
            model=model,
            X=X_test_scaled,
            y=y_test,
            feature_names=feature_cols,
            model_name=model_name,
            period_name=name
        )
        op_importances = importances.loc[operation_cols]
        print(op_importances)
        op_importances.to_csv(f"results/figures/feature importance/{model_name}_{name}_permutation_importance_ops.csv") 

        """

        # Save the test predictions to a CSV file for later analysis.
        df_pred = pd.DataFrame({
            "time": df_test_proc["TIME"].values,
            "true": y_test.values,
            "prediction": y_test_pred,
            "model": model_name,
            "periode": name
        })
        pred_dir = "results/predictions/"
        os.makedirs(pred_dir, exist_ok=True)
        pred_filename = os.path.join(pred_dir, f"{model_name}_{name}_test_predictions.csv")
        df_pred.to_csv(pred_filename, index=False)
        logger.info(f"Prediction CSV saved at: {pred_filename}")

        # Compute evaluation metrics and log them for both train and test sets.
        logger.info("Train metrics:")
        print_metrics(evaluate_regression(y_train, y_train_pred))
        logger.info("Test metrics:")
        print_metrics(evaluate_regression(y_test, y_test_pred))

        # Save trained model
        save_model(model, model_name, name)

        # Store metrics and configuration for comparison across periods.
        results_list.append({
            "Bereich": name,
            "Modell": model_name,
            "Train R2": evaluate_regression(y_train, y_train_pred)["R2"],
            "Test R2": evaluate_regression(y_test, y_test_pred)["R2"],
            "Train RMSE": evaluate_regression(y_train, y_train_pred)["RMSE"],
            "Test RMSE": evaluate_regression(y_test, y_test_pred)["RMSE"],
            "Train MAE": evaluate_regression(y_train, y_train_pred)["MAE"],
            "Test MAE": evaluate_regression(y_test, y_test_pred)["MAE"],
            "Best Params": best_params,
            "Skalierung": "StandardScaler",
            "Noise Augment": n_augment,
            "Noise Level": noise_level,
            "Zeitliche Auflösung": resample_rule if resample_rule is not None else "Original",
            "Trainingszeit (s)": round(train_duration, 2),
            "Vorhersagezeit (s)": round(prediction_duration, 4)
        })

    # Compile all results into a DataFrame and save as summary CSV.
    results_df = pd.DataFrame(results_list)
    print("\nModel Comparison:")
    print(results_df)

    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if SAVE_RESULTS:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        append_results_to_csv(results_df, csv_path)

if __name__ == "__main__":
    main()