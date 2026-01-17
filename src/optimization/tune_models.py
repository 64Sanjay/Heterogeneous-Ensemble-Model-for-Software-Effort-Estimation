"""
Hyperparameter tuning for all models
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.evaluation.metrics import calculate_mae


def tune_xgboost(X, y, n_trials=50):
    """Tune XGBoost hyperparameters"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'verbosity': 0
        }
        
        model = XGBRegressor(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = []
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            scores.append(calculate_mae(y_val, pred))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest XGBoost params: {study.best_params}")
    print(f"Best MAE: {study.best_value:.2f}")
    
    return study.best_params


def tune_knn(X, y, n_trials=30):
    """Tune KNN hyperparameters"""
    
    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 3)
        }
        
        model = KNeighborsRegressor(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = []
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            scores.append(calculate_mae(y_val, pred))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest KNN params: {study.best_params}")
    print(f"Best MAE: {study.best_value:.2f}")
    
    return study.best_params


def tune_svr(X, y, n_trials=50):
    """Tune SVR hyperparameters"""
    
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 1000, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
        
        model = SVR(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = []
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            scores.append(calculate_mae(y_val, pred))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest SVR params: {study.best_params}")
    print(f"Best MAE: {study.best_value:.2f}")
    
    return study.best_params


def main():
    print("="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)
    
    # Load data
    loader = DataLoader("cocomo81")
    df = loader.load_raw_data()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
    print(f"\nDataset: COCOMO81")
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")
    
    all_params = {}
    
    print("\n--- Tuning XGBoost ---")
    all_params['XGBoost'] = tune_xgboost(X, y, n_trials=30)
    
    print("\n--- Tuning KNN ---")
    all_params['KNN'] = tune_knn(X, y, n_trials=20)
    
    print("\n--- Tuning SVR ---")
    all_params['SVR'] = tune_svr(X, y, n_trials=30)
    
    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)
    print("\nBest Parameters:")
    for model, params in all_params.items():
        print(f"\n{model}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
    
    return all_params


if __name__ == "__main__":
    params = main()
