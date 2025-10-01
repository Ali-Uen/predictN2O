"""Model Training and Saving Utilities.

This module provides training routines for various regression models,
including tree-based models and deep neural networks (DNNs),
as well as utilities to save models and append results.
"""
import os
import pandas as pd
import logging
import joblib

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Optional imports
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    Sequential = Dense = Adam = EarlyStopping = None
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

# Supported model instances (except DNN, which is built separately)
estimator_dict = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "DNN": None,
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    estimator_dict["XGBoost"] = XGBRegressor(random_state=42, verbosity=0)



def train_model(X, 
                y,
                model_name, 
                param_grid=None, 
                cv=None, 
                scoring='r2', 
                n_jobs=-1, 
                verbose=2):
    """Trains a regression model, optionally with hyperparameter tuning via GridSearchCV.

    Args:
        X: Features, as a numpy array or pd.DataFrame.
        y: Target variable, as a numpy array or pd.Series.
        model_name: Name of the model (must be a key in estimator_dict).
        param_grid: Optional; dictionary of hyperparameters for GridSearchCV.
        cv: Optional; cross-validation splitting strategy.
        scoring: Optional; scoring metric for GridSearchCV.
        n_jobs: Optional; number of parallel jobs for GridSearchCV.
        verbose: Optional; verbosity level for GridSearchCV.

    Returns:
        Tuple containing the best estimator and best_params (or None if not using GridSearch).
    """
    if model_name != "DNN":
        estimator = estimator_dict[model_name]
        if param_grid is not None:
            grid = GridSearchCV(
                estimator,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose
            )
            grid.fit(X, y)
            logger.info("Best Params: %s", grid.best_params_)
            return grid.best_estimator_, grid.best_params_
        else:
            estimator.fit(X, y)
            logger.info("Model '%s' trained without hyperparameter search.", model_name)
            return estimator, None
        
def save_model(model, model_name: str, period_name: str, output_dir: str = "results/model_outputs") -> None:
    """Saves a trained model to disk using joblib.

    Args:
        model: The trained model object.
        model_name: String for the model type (e.g., "XGBoost").
        period_name: String for the data period (e.g., "Januar–März").
        output_dir: Directory to save the model file in.
    """

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_{period_name.replace(' ', '_')}.pkl"
    joblib.dump(model, os.path.join(output_dir, filename))
    logger.info("Model saved: %s", os.path.join(output_dir, filename))

def append_results_to_csv(new_results: pd.DataFrame, csv_path: str) -> None:
    """Appends new results to a CSV file, or creates it if it does not exist.

    Args:
        new_results: DataFrame containing new results to append.
        csv_path: Path to the CSV file.
    """
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
            combined = pd.concat([existing, new_results], ignore_index=True)
            # Optional: Remove duplicates if desired
            # combined = combined.drop_duplicates()
            combined.to_csv(csv_path, index=False)
            logger.info("Results appended to existing file: %s", csv_path)
        except Exception as e:
            logger.error("Error while appending to CSV: %s", e)
    else:
        new_results.to_csv(csv_path, index=False)
        logger.info("Results saved as new file: %s", csv_path)

def build_dnn(input_dim, learning_rate=0.001):
    """Builds a simple deep neural network using Keras Sequential API.

    Args:
        input_dim: Number of input features.
        learning_rate: Learning rate for Adam optimizer.
    Returns:
        Compiled Keras model.
    """
    logger.info("Building DNN with input_dim=%d, learning_rate=%.5f", input_dim, learning_rate)
    model = Sequential([
        Dense(160, activation='relu', input_shape=(input_dim,)),
        Dense(120, activation='relu'),
        Dense(80, activation='relu'),
        Dense(40, activation='relu'),
        Dense(20, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    logger.info("DNN model compiled.")
    return model

def train_dnn_model(X, y, param_dict=None):
    """
    Trains a deep neural network (DNN) model.

    Uses early stopping and validation split for regularization.

    Args:
        X (array-like): Training feature matrix.
        y (array-like): Training target vector.
        param_dict (dict, optional): DNN hyperparameters:
            - 'epochs' (int)
            - 'batch_size' (int)
            - 'learning_rate' (float)
            - 'patience' (int)

    Returns:
        Tuple:
            model (keras.Model): Trained Keras model.
            history (History): Keras training history object.
            dict: Dictionary of parameters used.
    """
    input_dim = X.shape[1]
    if param_dict is not None:
        epochs = param_dict.get("epochs", 200)
        batch_size = param_dict.get("batch_size", 128)
        learning_rate = param_dict.get("learning_rate", 0.001)
        patience = param_dict.get("patience", 20)
    else:
        epochs = 200
        batch_size = 128
        learning_rate = 0.001
        patience = 20

    logger.info(
            "Starting DNN training: epochs=%d, batch_size=%d, learning_rate=%.5f, patience=%d",
            epochs, batch_size, learning_rate, patience
    )

    model = Sequential([
        Dense(160, activation='relu', input_shape=(input_dim,)),
        Dense(120, activation='relu'),
        Dense(80, activation='relu'),
        Dense(40, activation='relu'),
        Dense(20, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # Early stopping callback for better generalization and to avoid overfitting.
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)
    history = model.fit(
        X, y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    logger.info("DNN model trained successfully.")
    return model, history, {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patience": patience
    }