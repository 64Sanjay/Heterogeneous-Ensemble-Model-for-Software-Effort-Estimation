"""
Hyperparameter tuning for software effort estimation models
"""

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from typing import Dict, Any, Callable, Optional
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

from src.models.cbr_model import CBRModel
from src.models.ml_models import XGBoostModel, ANNModel, KNNModel, SVRModel
from src.evaluation.metrics import calculate_mmre, calculate_mae
from src.utils.config import RANDOM_SEED


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna"""
    
    def __init__(self, n_trials: int = 100, 
                 cv_folds: int = 5,
                 metric: str = "mae",
                 random_state: int = RANDOM_SEED):
        """
        Initialize tuner
        
        Args:
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            metric: Optimization metric ('mae', 'mmre')
            random_state: Random seed
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.metric = metric
        self.random_state = random_state
        self.best_params = None
        self.study = None
        
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model using cross-validation"""
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            
            if self.metric == "mae":
                score = calculate_mae(y_val, predictions)
            else:
                score = calculate_mmre(y_val, predictions)
            
            scores.append(score)
        
        return np.mean(scores)
    
    def tune_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune XGBoost hyperparameters"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': self.random_state
            }
            
            from xgboost import XGBRegressor
            model = XGBRegressor(**params)
            
            return self._evaluate_model(model, X, y)
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        print(f"\nBest XGBoost params: {self.best_params}")
        print(f"Best score: {self.study.best_value:.4f}")
        
        return self.best_params
    
    def tune_ann(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune ANN (MLPRegressor) hyperparameters"""
        
        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 1, 3)
            layers = []
            for i in range(n_layers):
                layers.append(trial.suggest_int(f'n_units_l{i}', 16, 128))
            
            params = {
                'hidden_layer_sizes': tuple(layers),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.1),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.01),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'solver': 'adam',
                'max_iter': 1000,
                'random_state': self.random_state
            }
            
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(**params)
            
            return self._evaluate_model(model, X, y)
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        print(f"\nBest ANN params: {self.best_params}")
        print(f"Best score: {self.study.best_value:.4f}")
        
        return self.best_params
    
    def tune_knn(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune KNN hyperparameters"""
        
        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
                'p': trial.suggest_int('p', 1, 3)
            }
            
            from sklearn.neighbors import KNeighborsRegressor
            model = KNeighborsRegressor(**params)
            
            return self._evaluate_model(model, X, y)
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        self.study.optimize(objective, n_trials=min(self.n_trials, 50), show_progress_bar=True)
        
        self.best_params = self.study.best_params
        print(f"\nBest KNN params: {self.best_params}")
        print(f"Best score: {self.study.best_value:.4f}")
        
        return self.best_params
    
    def tune_svr(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune SVR hyperparameters"""
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.1, 1000, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }
            
            from sklearn.svm import SVR
            model = SVR(**params)
            
            return self._evaluate_model(model, X, y)
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        print(f"\nBest SVR params: {self.best_params}")
        print(f"Best score: {self.study.best_value:.4f}")
        
        return self.best_params
    
    def tune_cbr(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune CBR hyperparameters"""
        from scipy.spatial.distance import cdist
        
        def objective(trial):
            k = trial.suggest_int('k', 1, min(20, len(X) - 1))
            weighting = trial.suggest_categorical('weighting', ['uniform', 'distance', 'rank'])
            metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine'])
            
            # Feature weights
            n_features = X.shape[1]
            feature_weights = np.array([
                trial.suggest_float(f'fw_{i}', 0.1, 2.0) for i in range(n_features)
            ])
            feature_weights /= feature_weights.sum()
            
            # Cross-validation
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in kfold.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Apply feature weights
                X_train_w = X_train * feature_weights
                X_val_w = X_val * feature_weights
                
                predictions = []
                for x in X_val_w:
                    distances = cdist([x], X_train_w, metric=metric)[0]
                    neighbors = np.argsort(distances)[:k]
                    
                    if weighting == 'uniform':
                        pred = np.mean(y_train[neighbors])
                    elif weighting == 'distance':
                        weights = 1 / (distances[neighbors] + 1e-9)
                        pred = np.average(y_train[neighbors], weights=weights)
                    else:  # rank
                        weights = [(2 ** (k - i - 1)) / (2 ** k - 1) for i in range(k)]
                        pred = np.average(y_train[neighbors], weights=weights)
                    
                    predictions.append(pred)
                
                if self.metric == "mae":
                    score = calculate_mae(y_val, np.array(predictions))
                else:
                    score = calculate_mmre(y_val, np.array(predictions))
                
                scores.append(score)
            
            return np.mean(scores)
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        print(f"\nBest CBR params: {self.best_params}")
        print(f"Best score: {self.study.best_value:.4f}")
        
        return self.best_params
    
    def tune_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Tune all models and return best parameters"""
        results = {}
        
        print("Tuning XGBoost...")
        results['XGBoost'] = self.tune_xgboost(X, y)
        
        print("\nTuning ANN...")
        results['ANN'] = self.tune_ann(X, y)
        
        print("\nTuning KNN...")
        results['KNN'] = self.tune_knn(X, y)
        
        print("\nTuning SVR...")
        results['SVR'] = self.tune_svr(X, y)
        
        print("\nTuning CBR...")
        results['CBR'] = self.tune_cbr(X, y)
        
        return results
    
    def get_optimization_history(self) -> Dict:
        """Get optimization history from last study"""
        if self.study is None:
            return {}
        
        return {
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials),
            'values': [t.value for t in self.study.trials if t.value is not None]
        }


def quick_tune(model_name: str, X: np.ndarray, y: np.ndarray,
               n_trials: int = 50) -> Dict[str, Any]:
    """
    Quick hyperparameter tuning for a single model
    
    Args:
        model_name: Name of model ('xgboost', 'ann', 'knn', 'svr', 'cbr')
        X: Features
        y: Target values
        n_trials: Number of trials
        
    Returns:
        Best parameters
    """
    tuner = HyperparameterTuner(n_trials=n_trials)
    
    tune_funcs = {
        'xgboost': tuner.tune_xgboost,
        'ann': tuner.tune_ann,
        'knn': tuner.tune_knn,
        'svr': tuner.tune_svr,
        'cbr': tuner.tune_cbr
    }
    
    if model_name.lower() not in tune_funcs:
        raise ValueError(f"Unknown model: {model_name}")
    
    return tune_funcs[model_name.lower()](X, y)


if __name__ == "__main__":
    # Test hyperparameter tuning
    from src.data.data_loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    
    print("Loading data...")
    loader = DataLoader("cocomo81")
    df = loader.load_raw_data()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Quick tune XGBoost
    print("\nQuick tuning XGBoost (20 trials)...")
    best_params = quick_tune("xgboost", X, y, n_trials=20)
    print(f"\nBest parameters: {best_params}")