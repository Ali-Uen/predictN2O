"""Modern main script for N2O emission prediction pipeline.

This script demonstrates the new flexible architecture with:
- YAML-based configuration system
- Plugin-based feature engineering
- CLI argument overrides
- Modular and extensible design

Usage examples:
    python main_modern.py
    python main_modern.py --config config/custom_config.yaml
    python main_modern.py --model RandomForest --split 0.7
    python main_modern.py --config config/experiment.yaml --model XGBoost --augment 4 --noise 0.1
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.core import init_config, get_config, load_plugins, get_plugin_registry
from src.data_loader import load_and_prepare_data
from src.models import train_model, save_model, append_results_to_csv, train_dnn_model
from src.evaluation import evaluate_regression, print_metrics
from src.augmentation import add_noise

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments with support for configuration override."""
    parser = argparse.ArgumentParser(
        description="N2O prediction pipeline with flexible configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --config config/experiment.yaml
  %(prog)s --model RandomForest --split 0.7
  %(prog)s --config config/custom.yaml --model XGBoost --verbose
        """
    )
    
    # Configuration file
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to YAML configuration file"
    )
    
    # Model parameters
    parser.add_argument(
        "--model", 
        type=str,
        help="Model to use (overrides config)"
    )
    
    parser.add_argument(
        "--split", 
        type=float,
        help="Train/test split ratio (overrides config)"
    )
    
    # Data augmentation
    parser.add_argument(
        "--augment", 
        type=int,
        help="Number of augmented samples per original (overrides config)"
    )
    
    parser.add_argument(
        "--noise", 
        type=float,
        help="Noise level for augmentation (overrides config)"
    )
    
    # General settings
    parser.add_argument(
        "--seed", 
        type=int,
        help="Random seed (overrides config)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output (overrides config)"
    )
    
    # Experiment settings
    parser.add_argument(
        "--periods", 
        nargs="+",
        help="Specific periods to run (by name)"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show configuration and exit without training"
    )
    
    return parser.parse_args()


def setup_logging(config):
    """Setup logging based on configuration."""
    log_config = config.get_section('logging')
    
    # Set logging level
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    
    # Create formatters
    formatter = logging.Formatter(log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'))
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if log_config.get('console_logging', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_config.get('file_logging', True):
        log_file = log_config.get('log_file', 'results/logs/pipeline.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def initialize_system(args):
    """Initialize configuration and plugin systems."""
    logger.info("Initializing predictN2O pipeline...")
    
    # Initialize configuration
    config = init_config(args.config, args)
    
    # Setup logging with new configuration
    setup_logging(config)
    
    # Load plugins
    logger.info("Loading plugins...")
    load_plugins(config.to_dict())
    
    # Print system summary
    if config.get('general.verbose', False):
        config.print_summary()
        get_plugin_registry().print_plugin_summary()
    
    return config


def split_train_test_time(df: pd.DataFrame, split_ratio: float = 0.8):
    """Split DataFrame by time (temporal split)."""
    n = len(df)
    split_idx = int(n * split_ratio)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test


def get_period_mask(df: pd.DataFrame, start_mmdd: str, end_mmdd: str, time_col: str = 'TIME'):
    """Create boolean mask for time period."""
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")
        
    timestr = df[time_col].dt.strftime('%m-%d')
    return (timestr >= start_mmdd) & (timestr <= end_mmdd)


def resolve_auto_features(config, df: pd.DataFrame):
    """Resolve AUTO_FEATURES placeholders with actual feature columns."""
    # Get configured feature columns
    feature_columns = config.get('data.feature_columns', [])
    target_column = config.get('data.target_column')
    time_column = config.get('data.time_column')
    exclude_cols = config.get('data.exclude_cols', [])
    
    # Auto-detect features if needed
    if feature_columns == "AUTO_FEATURES" or not feature_columns:
        logger.info("Auto-detecting feature columns...")
        # Use all numeric columns except target and time
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        auto_features = [col for col in numeric_cols 
                        if col != target_column and col != time_column and col not in exclude_cols]
        logger.info(f"Auto-detected {len(auto_features)} feature columns: {auto_features}")
        return auto_features
    elif isinstance(feature_columns, list):
        return feature_columns
    else:
        return []


def apply_feature_engineering(df_train: pd.DataFrame, df_test: pd.DataFrame, config, period_name: str):
    """Apply feature engineering using plugin system."""
    plugin_registry = get_plugin_registry()
    
    # Resolve AUTO_FEATURES in configuration
    feature_columns = resolve_auto_features(config, df_train)
    
    # Update config with resolved feature columns for plugins
    updated_config = config.to_dict()
    updated_config['data']['feature_columns'] = feature_columns
    
    # Get enabled feature engineering plugins
    plugins_config = config.get_section('plugins')
    fe_config = plugins_config.get('feature_engineering', {})
    enabled_plugins = fe_config.get('enabled_plugins', [])
    
    if not enabled_plugins:
        logger.warning("No feature engineering plugins enabled")
        return df_train, df_test
    
    # Create config context for plugins (include current period info and resolved features)
    plugin_config = updated_config.copy()
    plugin_config['current_period_name'] = period_name
    
    # Apply plugins to training data
    df_train_processed = df_train.copy()
    for plugin_name in enabled_plugins:
        try:
            df_train_processed = plugin_registry.execute_plugin(
                plugin_name, df_train_processed, plugin_config
            )
        except Exception as e:
            logger.error(f"Feature engineering plugin '{plugin_name}' failed on training data: {e}")
            raise
    
    # Apply same transformations to test data
    df_test_processed = df_test.copy()
    for plugin_name in enabled_plugins:
        try:
            df_test_processed = plugin_registry.execute_plugin(
                plugin_name, df_test_processed, plugin_config
            )
        except Exception as e:
            logger.error(f"Feature engineering plugin '{plugin_name}' failed on test data: {e}")
            raise
    
    return df_train_processed, df_test_processed


def run_experiment(config, args):
    """Run the main ML pipeline experiment."""
    logger.info("Starting N2O prediction experiment...")
    
    # Extract configuration values
    model_name = config.get('models.default_model')
    split_ratio = config.get('data.default_split_ratio')
    n_augment = config.get('augmentation.n_augment', 0)
    noise_level = config.get('augmentation.noise_level', 0.0)
    random_seed = config.get('general.random_seed')
    
    # Set random seed
    np.random.seed(random_seed)
    
    logger.info(f"""
=== EXPERIMENT CONFIGURATION ===
Model: {model_name}
Split Ratio: {split_ratio}
Augmentation: {n_augment} copies with {noise_level} noise
Random Seed: {random_seed}
=================================""")
    
    results_list = []
    
    # Load data
    data_path = config.get('data.data_path')
    logger.info(f"Loading data from: {data_path}")
    df_full = load_and_prepare_data(data_path)
    
    # Get periods to process
    periods = config.get('periods', [])
    if args.periods:
        # Filter to specific periods
        period_names = [p.lower() for p in args.periods] 
        periods = [p for p in periods if p['name'].lower() in period_names]
        logger.info(f"Running only specified periods: {[p['name'] for p in periods]}")
    
    # Process each period
    for period in periods:
        period_name = period['name']
        start_date = period['start_date']
        end_date = period['end_date']
        
        logger.info(f"\n=== Processing Period: {period_name} ({start_date} to {end_date}) ===")
        
        try:
            # Filter data for period
            time_col = config.get('data.time_column', 'TIME')
            mask = get_period_mask(df_full, start_date, end_date, time_col)
            df_period = df_full[mask].copy()
            
            # Apply resampling if configured
            resample_rule = get_resampling_rule(model_name, period_name, config)
            if resample_rule:
                logger.info(f"Resampling data with rule: {resample_rule}")
                df_period = df_period.resample(resample_rule, on=time_col).mean().reset_index()
            
            # Check minimum data requirements
            min_samples = config.get('data.min_samples_period', 50)
            if len(df_period) < min_samples:
                logger.warning(f"Insufficient data ({len(df_period)} < {min_samples}) - skipping period")
                continue
            
            # Split data temporally
            df_train, df_test = split_train_test_time(df_period, split_ratio)
            
            min_test_samples = config.get('data.min_test_samples', 10)
            if len(df_test) < min_test_samples:
                logger.warning(f"Insufficient test data ({len(df_test)} < {min_test_samples}) - skipping period")
                continue
            
            # Apply feature engineering
            logger.info("Applying feature engineering...")
            df_train_proc, df_test_proc = apply_feature_engineering(df_train, df_test, config, period_name)
            
            # Drop NaN values created by lag/rolling features
            initial_train_len = len(df_train_proc)
            initial_test_len = len(df_test_proc)
            df_train_proc = df_train_proc.dropna().reset_index(drop=True)
            df_test_proc = df_test_proc.dropna().reset_index(drop=True)
            
            logger.info(f"Dropped NaNs: Train {initial_train_len}->{len(df_train_proc)}, Test {initial_test_len}->{len(df_test_proc)}")
            
            # Extract features and target
            target_col = config.get('data.target_column', 'N2O')
            exclude_cols = [time_col, target_col]
            
            feature_cols = [col for col in df_train_proc.columns if col not in exclude_cols]
            logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
            
            X_train = df_train_proc[feature_cols]
            y_train = df_train_proc[target_col]
            X_test = df_test_proc[feature_cols]
            y_test = df_test_proc[target_col]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Apply data augmentation
            if n_augment > 0 and noise_level > 0:
                logger.info(f"Applying data augmentation: {n_augment} copies with {noise_level} noise")
                X_train_aug, y_train_aug = add_noise(
                    X_train_scaled, y_train.values,
                    n_augment=n_augment,
                    noise_level=noise_level,
                    random_state=random_seed
                )
            else:
                X_train_aug, y_train_aug = X_train_scaled, y_train.values
            
            # Train model
            logger.info(f"Training {model_name} model...")
            train_start = time.time()
            
            param_grid = get_param_grid(model_name, period_name, config)
            cv_folds = config.get('models.cv_folds', 5)
            scoring = config.get('models.cv_scoring', 'r2')
            
            if model_name == "DNN":
                # For DNN, convert grid to single parameter dict
                dnn_params = {k: v[0] if isinstance(v, list) else v for k, v in param_grid.items()}
                model, history, best_params = train_dnn_model(X_train_aug, y_train_aug, dnn_params)
            else:
                model, best_params = train_model(
                    X_train_aug, y_train_aug, model_name,
                    param_grid=param_grid,
                    cv=TimeSeriesSplit(n_splits=cv_folds),
                    scoring=scoring
                )
            
            train_duration = time.time() - train_start
            
            # Make predictions
            predict_start = time.time()
            if model_name == "DNN":
                y_test_pred = model.predict(X_test_scaled).flatten()
                y_train_pred = model.predict(X_train_scaled).flatten()
            else:
                y_test_pred = model.predict(X_test_scaled)
                y_train_pred = model.predict(X_train_scaled)
            predict_duration = time.time() - predict_start
            
            # Evaluate results
            train_metrics = evaluate_regression(y_train, y_train_pred)
            test_metrics = evaluate_regression(y_test, y_test_pred)
            
            logger.info("Training Metrics:")
            print_metrics(train_metrics)
            logger.info("Test Metrics:")
            print_metrics(test_metrics)
            
            # Save predictions
            save_predictions(df_test_proc, y_test, y_test_pred, model_name, period_name, config)
            
            # Save model
            save_model(model, model_name, period_name)
            
            # Record results
            results_list.append({
                "Period": period_name,
                "Model": model_name,
                "Train_R2": train_metrics["R2"],
                "Test_R2": test_metrics["R2"],
                "Train_RMSE": train_metrics["RMSE"],
                "Test_RMSE": test_metrics["RMSE"],
                "Train_MAE": train_metrics["MAE"],
                "Test_MAE": test_metrics["MAE"],
                "Best_Params": str(best_params),
                "Features_Count": len(feature_cols),
                "Train_Samples": len(X_train_aug),
                "Test_Samples": len(X_test),
                "Training_Time_s": round(train_duration, 2),
                "Prediction_Time_s": round(predict_duration, 4),
                "Augmentation": f"{n_augment}x{noise_level}",
                "Resampling": resample_rule or "None"
            })
            
        except Exception as e:
            logger.error(f"Error processing period {period_name}: {e}")
            continue
    
    # Save results summary
    if results_list:
        results_df = pd.DataFrame(results_list)
        logger.info("\n=== EXPERIMENT RESULTS ===")
        print(results_df.to_string(index=False))
        
        results_dir = config.get('output.results_dir', 'results')
        results_path = os.path.join(results_dir, 'experiment_results.csv')
        
        os.makedirs(results_dir, exist_ok=True)
        append_results_to_csv(results_df, results_path)
        logger.info(f"Results saved to: {results_path}")
    else:
        logger.warning("No results to save - all periods failed or were skipped")


def get_resampling_rule(model_name: str, period_name: str, config) -> str:
    """Get resampling rule for model and period."""
    # Check for model-specific overrides
    overrides = config.get('resampling.model_overrides', {})
    if model_name in overrides and period_name in overrides[model_name]:
        return overrides[model_name][period_name]
    
    # Use default
    return config.get('resampling.default_rule')


def get_param_grid(model_name: str, period_name: str, config) -> dict:
    """Get hyperparameter grid for model and period."""
    # Get hyperparameters from config
    hyperparams = config.get('hyperparameters', {})
    if model_name in hyperparams:
        return hyperparams[model_name]
    
    logger.warning(f"No hyperparameters found for {model_name}, using defaults")
    return {}


def save_predictions(df_test: pd.DataFrame, y_true, y_pred, 
                    model_name: str, period_name: str, config):
    """Save predictions to CSV file."""
    time_col = config.get('data.time_column', 'TIME')
    predictions_dir = config.get('output.predictions_dir', 'results/predictions')
    
    os.makedirs(predictions_dir, exist_ok=True)
    
    df_pred = pd.DataFrame({
        "time": df_test[time_col].values,
        "true": y_true.values,
        "prediction": y_pred,
        "model": model_name,
        "period": period_name
    })
    
    filename = f"{model_name}_{period_name}_predictions.csv"
    filepath = os.path.join(predictions_dir, filename)
    
    df_pred.to_csv(filepath, index=False)
    logger.info(f"Predictions saved to: {filepath}")


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Initialize system
        config = initialize_system(args)
        
        # Check for dry run
        if args.dry_run:
            logger.info("Dry run completed - configuration loaded successfully")
            return
        
        # Run experiment
        run_experiment(config, args)
        
        logger.info("Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if config.get('general.verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()