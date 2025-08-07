"""
param_grids.py

Hyperparameter grids for each model and period, used for GridSearchCV.
Each value is a list (even if only one value, for consistency).
Tuned on validation set in preliminary experiments; only optimal settings shown.
"""
param_grids = {
    "XGBoost": {
        "Januar–März": {
            "n_estimators": [1200],
            "learning_rate": [0.003],
            "max_depth": [6],
            "min_child_weight": [3],
            "subsample": [0.8],
            "colsample_bytree": [0.8]
        },
        "März–Mitte Mai": {
            "n_estimators": [1200],
            "learning_rate": [0.003],
            "max_depth": [6],
            "min_child_weight": [3],
            "subsample": [0.8],
            "colsample_bytree": [0.8]
        },
        "Mai–Oktober": {
            "n_estimators": [1200],
            "learning_rate": [0.003],
            "max_depth": [6],
            "min_child_weight": [3],
            "subsample": [0.8],
            "colsample_bytree": [0.8]
        }
    },
    "RandomForest": {
        "Januar–März": {
            "n_estimators": [1000],
            "max_depth": [20],
            "min_samples_split": [10],
            "min_samples_leaf": [2]
        },
        "März–Mitte Mai": {
            "n_estimators": [1000],
            "max_depth": [20],
            "min_samples_split": [10],
            "min_samples_leaf": [2]
        },
        "Mai–Oktober": {
            "n_estimators": [1000],
            "max_depth": [20],
            "min_samples_split": [10],
            "min_samples_leaf": [2]
        }
    },
    "KNN": { 
        "Januar–März": {
            "n_neighbors": [10]
        },
        "März–Mitte Mai": {
            "n_neighbors": [10]
        },
        "Mai–Oktober": {
            "n_neighbors": [10]
        }
    },
    "DecisionTree": {
        "Januar–März": {
            "max_depth": [30],
            "min_samples_split": [7],
            "min_samples_leaf": [2]
        },
        "März–Mitte Mai": {
            "max_depth": [30],
            "min_samples_split": [9],
            "min_samples_leaf": [4]
        },
        "Mai–Oktober": {
            "max_depth": [30],
            "min_samples_split": [3],
            "min_samples_leaf": [3]
        }
    },

    "AdaBoost": {
        "Januar–März": {
            "n_estimators": [800],
            "learning_rate": [1.0],
        },
        "März–Mitte Mai": {
            "n_estimators": [1500],
            "learning_rate": [1.0],

        },
        "Mai–Oktober": {
            "n_estimators": [200],
            "learning_rate": [0.01],
        }
    },
 "DNN": {
        "Januar–März": {
            "epochs": [200],
            "batch_size": [128],
            "learning_rate": [0.001],
            "patience": [20]
        },
        "März–Mitte Mai": {
            "epochs": [200],
            "batch_size": [128],
            "learning_rate": [0.001],
            "patience": [20]
        },
        "Mai–Oktober": {
            "epochs": [200],
            "batch_size": [128],
            "learning_rate": [0.001],
            "patience": [20]
        }
    }
}