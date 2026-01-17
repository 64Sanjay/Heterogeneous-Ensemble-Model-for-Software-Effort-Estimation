"""
Weighted Ensemble Model with Learned Weights
"""

import numpy as np
from typing import List, Dict, Optional
import time
from scipy.optimize import minimize

from src.models.base_model import BaseEstimator
from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel
from src.models.ml_models import XGBoostModel, ANNModel, KNNModel, SVRModel
from src.evaluation.metrics import calculate_mae


class WeightedEnsemble(BaseEstimator):
    """
    Weighted Ensemble that learns optimal weights for each model
    """
    
    def __init__(self, ml_model_name: str = "XGBoost", optimize_weights: bool = True):
        super().__init__(f"WeightedEnsemble_{ml_model_name}")
        
        self.ml_model_name = ml_model_name
        self.optimize_weights = optimize_weights
        
        # Initialize models
        self.cbr_model = CBRModel()
        self.cocomo_model = COCOMOModel()
        self.ml_model = self._get_ml_model(ml_model_name)
        
        # Default weights (equal)
        self.weights = np.array([1/3, 1/3, 1/3])
        
    def _get_ml_model(self, name: str) -> BaseEstimator:
        models = {
            "XGBoost": XGBoostModel,
            "ANN": ANNModel,
            "KNN": KNNModel,
            "SVR": SVRModel
        }
        return models[name]()
    
    def _optimize_weights(self, X: np.ndarray, y: np.ndarray):
        """Learn optimal weights using training data"""
        
        # Get predictions from each model
        pred_cbr = self.cbr_model.predict(X)
        pred_cocomo = self.cocomo_model.predict(X)
        pred_ml = self.ml_model.predict(X)
        
        def objective(weights):
            # Normalize weights
            w = weights / weights.sum()
            combined = w[0] * pred_cbr + w[1] * pred_cocomo + w[2] * pred_ml
            return calculate_mae(y, combined)
        
        # Constraints: weights sum to 1, all positive
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1), (0, 1), (0, 1)]
        
        # Optimize
        result = minimize(
            objective,
            x0=[1/3, 1/3, 1/3],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.weights = result.x / result.x.sum()
        print(f"    Optimized weights: CBR={self.weights[0]:.3f}, COCOMO={self.weights[1]:.3f}, {self.ml_model_name}={self.weights[2]:.3f}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightedEnsemble':
        start_time = time.time()
        
        # Fit each model
        self.cbr_model.fit(X, y)
        self.cocomo_model.fit(X, y)
        self.ml_model.fit(X, y)
        
        # Optimize weights
        if self.optimize_weights:
            self._optimize_weights(X, y)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        pred_cbr = self.cbr_model.predict(X)
        pred_cocomo = self.cocomo_model.predict(X)
        pred_ml = self.ml_model.predict(X)
        
        # Weighted combination
        combined = (self.weights[0] * pred_cbr + 
                   self.weights[1] * pred_cocomo + 
                   self.weights[2] * pred_ml)
        
        return np.maximum(combined, 0)
    
    def get_weights(self) -> Dict[str, float]:
        return {
            "CBR": self.weights[0],
            "COCOMO": self.weights[1],
            self.ml_model_name: self.weights[2]
        }


class SelectivEnsemble(BaseEstimator):
    """
    Ensemble that selects best K models dynamically
    """
    
    def __init__(self, top_k: int = 2):
        super().__init__(f"SelectiveEnsemble_Top{top_k}")
        self.top_k = top_k
        
        self.models = {
            "CBR": CBRModel(),
            "COCOMO": COCOMOModel(),
            "XGBoost": XGBoostModel(),
            "KNN": KNNModel(),
            "SVR": SVRModel()
        }
        
        self.selected_models = []
        self.model_scores = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SelectivEnsemble':
        start_time = time.time()
        
        # Fit all models and get their training errors
        for name, model in self.models.items():
            model.fit(X, y)
            predictions = model.predict(X)
            mae = calculate_mae(y, predictions)
            self.model_scores[name] = mae
        
        # Select top K models with lowest MAE
        sorted_models = sorted(self.model_scores.items(), key=lambda x: x[1])
        self.selected_models = [name for name, _ in sorted_models[:self.top_k]]
        
        print(f"    Selected models: {self.selected_models}")
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for name in self.selected_models:
            pred = self.models[name].predict(X)
            predictions.append(pred)
        
        # Average of selected models
        return np.mean(predictions, axis=0)
