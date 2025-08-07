"""Feature engineering utilities for time series modeling.

This module adds:
- cyclic time features (hour, day of year)
- lag features for past values
- rolling mean and std features
- full feature engineering pipeline

Also includes model-specific lag and rolling configurations.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_time_features(df):
    """Adds cyclic time-based features to the DataFrame.

    Adds sine and cosine transformations of:
    - hour of day
    - day of year

    Args:
        df (pd.DataFrame): Input DataFrame with a 'TIME' column.

    Returns:
        pd.DataFrame: DataFrame with new time features added.
    """
    logger.info("Adding time features: hour_sin/cos, doy_sin/cos.")
    if not np.issubdtype(df['TIME'].dtype, np.datetime64):
        logger.warning("TIME column is not datetime, converting.")
        df['TIME'] = pd.to_datetime(df['TIME'])
    df['hour'] = df['TIME'].dt.hour
    df['dayofyear'] = df['TIME'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df_out = df.drop(columns=['hour', 'dayofyear'])
    logger.info("Finished adding time features.")
    return df_out

def add_lag_features(df, lags_dict):
    """Adds lagged versions of specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        lags_dict (dict): Dictionary where keys are column names and values are lists of lag steps.

    Returns:
        pd.DataFrame: DataFrame with added lag features.
    """
    logger.info(f"Adding lag features: {lags_dict}")
    for col, lags in lags_dict.items():
        for lag in lags:
            new_col = f'{col}_lag{lag}'
            df[new_col] = df[col].shift(lag)
            logger.debug(f"Added lag feature: {new_col}")
    logger.info("Finished adding lag features.")
    return df

def add_rolling_features(df, rolling_dict):
    """Adds rolling mean and std features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        rolling_dict (dict): Dictionary where keys are column names and values are lists of window sizes.

    Returns:
        pd.DataFrame: DataFrame with added rolling features.
    """
    for col, windows in rolling_dict.items():
        for window in windows:
            df[f"{col}_roll{window}"] = df[col].rolling(window, min_periods=1).mean()
            df[f"{col}_std{window}"] = df[col].rolling(window, min_periods=1).std()
    return df

def preprocess_features(df, lags_dict=None, rolling_dict=None):
    """Performs full feature engineering pipeline: time, lag, and rolling features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        lags_dict (dict, optional): Dict of lags to apply per column.
        rolling_dict (dict, optional): Dict of rolling window sizes per column.

    Returns:
        pd.DataFrame: Processed DataFrame with features and dropped NaNs.
    """
    logger.info("Starting feature engineering pipeline.")
    orig_len = len(df)

    df = add_time_features(df)
    if lags_dict is None:
        lags_dict = {}
    df = add_lag_features(df, lags_dict)
    if rolling_dict is None:
        rolling_dict = {}
    df = add_rolling_features(df, rolling_dict)

    after_feat_len = len(df)
    logger.info(f"Rows before dropping NaNs: {after_feat_len}")
    df = df.dropna().reset_index(drop=True)
    final_len = len(df)
    logger.info(f"Rows after dropping NaNs: {final_len} (dropped {after_feat_len - final_len})")
    logger.info("Feature engineering pipeline complete.")
    return df

# Period-specific lag feature configurations
model_period_lags = {
    "KNN": {
        "Januar–März": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "März–Mitte Mai": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "Mai–Oktober": {'DO': [], 'Q_in': [], 'T': []}
    },
    "RandomForest": {
        "Januar–März": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "März–Mitte Mai": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "Mai–Oktober": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]}
    },
    "XGBoost": {
        "Januar–März": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "März–Mitte Mai": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "Mai–Oktober": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]}
    },
    "AdaBoost": {
        "Januar–März": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "März–Mitte Mai": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "Mai–Oktober": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]}
    },
    "DecisionTree": {
        "Januar–März": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "März–Mitte Mai": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "Mai–Oktober": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]}
    },
    "DNN": {
        "Januar–März": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "März–Mitte Mai": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]},
        "Mai–Oktober": {'DO': [1, 2], 'Q_in': [1, 2], 'T': [1, 2]}
    }
}
# Period-specific rolling feature configurations
model_period_rolling = {
    "KNN": {
        "Januar–März": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "März–Mitte Mai": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "Mai–Oktober": {
            "DO": [],
            "Q_in": [],
            "T": []
        }
    },
    "RandomForest": {
        "Januar–März": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "März–Mitte Mai": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "Mai–Oktober": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        }
    },
    "XGBoost": {
        "Januar–März": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "März–Mitte Mai": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "Mai–Oktober": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        }
    },
    "AdaBoost": {
        "Januar–März": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "März–Mitte Mai": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "Mai–Oktober": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        }
    },
    "DecisionTree": {
        "Januar–März": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "März–Mitte Mai": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "Mai–Oktober": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        }
    },
    "DNN": {
        "Januar–März": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "März–Mitte Mai": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        },
        "Mai–Oktober": {
            "DO": [3, 6, 12, 24],
            "Q_in": [3, 6, 12, 24],
            "T": [3, 6, 12, 24]
        }
    }
}