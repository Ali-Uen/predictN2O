"""Data loading and preprocessing module.

Provides utilities to load CSV data and clean it by:
- removing missing values
- filtering negative values
- removing univariate (IQR) and multivariate (Mahalanobis) outliers

Intended for use as the first step in the ML pipeline.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
def remove_outliers_mahalanobis(df, cols, quantile=0.99):
    """Removes multivariate outliers using the Mahalanobis distance.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (list of str): Column names used for distance calculation.
        quantile (float, optional): Quantile threshold for outlier detection. Default is 0.99.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    X = df[cols].values
    mean_vec = np.mean(X, axis=0)
    cov_matrix = np.cov(X, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix)
    diff = X - mean_vec
    m_dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    threshold = np.quantile(m_dist, quantile)
    mask = m_dist <= threshold
    logging.info(f"Removed {np.sum(~mask)} multivariate outliers (Mahalanobis, quantile {quantile})")
    return df[mask].reset_index(drop=True)

def remove_outliers_iqr(df, cols, factor=1.5):
    """Removes univariate outliers based on the Interquartile Range (IQR).

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (list of str): Column names for IQR calculation.
        factor (float, optional): IQR multiplier. Default is 1.5.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    initial_shape = df.shape[0]
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        logging.info(
            f"IQR bounds for '{col}': lower_bound={lower_bound:.4f}, upper_bound={upper_bound:.4f}"
        )
        before = df.shape[0]
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        logging.info(f"Removed outliers in '{col}': {before - df.shape[0]} rows removed")
    logging.info(f"Rows remaining after IQR filter: {df.shape[0]} (from {initial_shape})")
    return df.reset_index(drop=True)

def load_and_prepare_data(train_path, time_col="TIME", exclude_cols=None, include_cols=None):
    """Loads a dataset from CSV, performs cleaning and outlier removal.

    Cleaning includes:
        - parsing datetime
        - removing missing values
        - removing negative values
        - IQR-based and Mahalanobis-based outlier removal

    Args:
        train_path (str): Path to the CSV file.
        time_col (str, optional): Name of the time column. Defaults to "TIME".
        exclude_cols (List[str], optional): Columns to exclude from outlier/negativity checks.
        include_cols (List[str], optional): If provided, only these columns are checked.

    Returns:
        pd.DataFrame: Cleaned and sorted DataFrame.
    """
    # Load a sample to detect column types
    df_sample = pd.read_csv(train_path, nrows=10)
    if exclude_cols is None:
        exclude_cols = [time_col]
    if include_cols is not None:
        cols_check = [col for col in include_cols if col in df_sample.columns]
    else:
        cols_check = [
            col for col in df_sample.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]
    logging.info(f"Automatically selected numeric columns for checks: {cols_check}")

    df = pd.read_csv(train_path)
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(time_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_nan = df.isna().sum().sum()
    logging.info(f"Missing values before removal: {n_nan}")
    df = df.dropna()
    n_neg = (df[cols_check] < 0).sum().sum()
    logging.info(f"Negative values before removal: {n_neg}")
    # Recommended: check all columns at once for negative values
    mask = (df[cols_check] >= 0).all(axis=1)
    df = df[mask]
    df.reset_index(drop=True, inplace=True)
    df = remove_outliers_iqr(df, cols_check)
    df = remove_outliers_mahalanobis(df, cols_check, quantile=0.99)
    logging.info(f"Rows remaining after cleaning: {df.shape[0]}")
    return df