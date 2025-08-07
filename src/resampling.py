"""
resampling.py

Defines resampling frequency per model and period.
Value should be a pandas offset alias string (e.g., '30T' for 30 minutes) or None if no resampling is applied.
"""
resampling = {
    "XGBoost": {
        "Januar–März": None,
        "März–Mitte Mai": None,
        "Mai–Oktober": None
    },
    "RandomForest": {
        "Januar–März": None,
        "März–Mitte Mai": None,
        "Mai–Oktober": None
    },
    "KNN": {
        "Januar–März": None,
        "März–Mitte Mai": "30T", # Resample every 30 minutes
        "Mai–Oktober": None
    },
    "AdaBoost": {
        "Januar–März": None,
        "März–Mitte Mai": None,
        "Mai–Oktober": None
    },
    "DecisionTree": {
        "Januar–März": None,
        "März–Mitte Mai": None,
        "Mai–Oktober": None
    },
    "DNN": {
        "Januar–März": None,
        "März–Mitte Mai": None,
        "Mai–Oktober": None
    }
}