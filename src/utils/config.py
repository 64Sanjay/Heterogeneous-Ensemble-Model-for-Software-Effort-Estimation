"""
Configuration file for the project
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset Configurations
DATASETS = {
    "cocomo81": {
        "filename": "cocomo81.csv",
        "num_samples": 63,
        "description": "COCOMO 81 Dataset"
    },
    "nasa93": {
        "filename": "nasa93.csv",
        "num_samples": 93,
        "description": "NASA 93 Dataset"
    },
    "nasa60": {
        "filename": "nasa60.csv",
        "num_samples": 60,
        "description": "NASA 60 Dataset"
    }
}

# COCOMO II Cost Driver Mappings
COCOMO_MULTIPLIERS = {
    'rely': {'vl': 0.75, 'l': 0.88, 'n': 1.00, 'h': 1.15, 'vh': 1.40, 'xh': None},
    'data': {'vl': None, 'l': 0.94, 'n': 1.00, 'h': 1.08, 'vh': 1.16, 'xh': None},
    'cplx': {'vl': 0.70, 'l': 0.85, 'n': 1.00, 'h': 1.15, 'vh': 1.30, 'xh': 1.65},
    'time': {'vl': None, 'l': None, 'n': 1.00, 'h': 1.11, 'vh': 1.30, 'xh': 1.66},
    'stor': {'vl': None, 'l': None, 'n': 1.00, 'h': 1.06, 'vh': 1.21, 'xh': 1.56},
    'virt': {'vl': None, 'l': 0.87, 'n': 1.00, 'h': 1.15, 'vh': 1.30, 'xh': None},
    'turn': {'vl': None, 'l': 0.87, 'n': 1.00, 'h': 1.07, 'vh': 1.15, 'xh': None},
    'acap': {'vl': 1.46, 'l': 1.19, 'n': 1.00, 'h': 0.86, 'vh': 0.71, 'xh': None},
    'aexp': {'vl': 1.29, 'l': 1.13, 'n': 1.00, 'h': 0.91, 'vh': 0.82, 'xh': None},
    'pcap': {'vl': 1.42, 'l': 1.17, 'n': 1.00, 'h': 0.86, 'vh': 0.70, 'xh': None},
    'vexp': {'vl': 1.21, 'l': 1.10, 'n': 1.00, 'h': 0.90, 'vh': None, 'xh': None},
    'lexp': {'vl': 1.14, 'l': 1.07, 'n': 1.00, 'h': 0.95, 'vh': None, 'xh': None},
    'modp': {'vl': 1.24, 'l': 1.10, 'n': 1.00, 'h': 0.91, 'vh': 0.82, 'xh': None},
    'tool': {'vl': 1.24, 'l': 1.10, 'n': 1.00, 'h': 0.91, 'vh': 0.83, 'xh': None},
    'sced': {'vl': 1.23, 'l': 1.08, 'n': 1.00, 'h': 1.04, 'vh': 1.10, 'xh': None}
}

# Model Hyperparameters
MODEL_PARAMS = {
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "random_state": 42
    },
    "ann": {
        "hidden_layer_sizes": (100, 50),
        "max_iter": 1000,
        "learning_rate_init": 0.01,
        "activation": "relu",
        "solver": "adam",
        "random_state": 42
    },
    "knn": {
        "n_neighbors": 5
    },
    "svr": {
        "kernel": "rbf",
        "C": 100,
        "gamma": 0.1,
        "epsilon": 0.1
    },
    "cbr": {
        "k_range": (1, 10)
    },
    "cocomo": {
        "a": 2.94,
        "b": 1.12,
        "epochs": 500,
        "batch_size": 16
    }
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    "combination_rules": ["median", "linear"],
    "linear_weights": [0.4, 0.3, 0.3],  # CBR, COCOMO, ML
    "base_models": ["CBR", "COCOMO"],
    "ml_models": ["ANN", "KNN", "XGBoost", "SVR"]
}

# Cross-Validation Settings
CV_CONFIG = {
    "loocv": True,
    "kfold_splits": [3, 5, 10],
    "random_state": 42
}

# Random Seed for Reproducibility
RANDOM_SEED = 42