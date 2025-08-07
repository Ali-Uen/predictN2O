"""
config.py

Central configuration file for paths, parameters, and settings
used throughout the Altenrhein WWTP ML pipeline.
"""
import os
# === Data paths ===
DATA_PATH = os.path.join("data", "AltenrheinWWTP.csv")
RESULTS_DIR = "results"
MODEL_OUTPUT_DIR = os.path.join(RESULTS_DIR, "model_outputs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# === Default parameters ===
MODELS = ["KNN", 
          "RandomForest", 
          "XGBoost", 
          "AdaBoost", 
          "DecisionTree", 
          "DNN"]
DEFAULT_MODEL = MODELS[0]
DEFAULT_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

# === Plotting & saving settings ===
SAVE_FIGURES = True
SAVE_RESULTS = True

# === Default augmentation parameters ===
AUGMENTATION_N = 0       # Number of augmentation copies (0 = no augmentation)
AUGMENTATION_NOISE = 0.00 # Noise level relative to std dev (e.g., 0.01 = 1% noise)

# === Additional central settings ===
VERBOSE = True           # Enables verbose logging/debug output

# === Time periods for workflow ===
PERIODS = [
    ("Januar–März",    "01-01", "03-31"),
    ("März–Mitte Mai", "03-01", "05-15"),
    ("Mai–Oktober",    "05-15", "10-31")
]