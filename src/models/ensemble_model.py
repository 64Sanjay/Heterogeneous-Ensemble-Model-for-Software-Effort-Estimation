"""
Heterogeneous Ensemble Model for Software Effort Estimation
"""

import numpy as np
from typing import List, Dict, Optional, Callable
import time

from src.models.base_model import BaseEstimator
from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel
from src.models.ml_models import (
    ANNModel, KNNModel, XGBoostModel, SVRModel, LinearRegressionModel
)
from src.utils.config import ENSEMBLE_CONFIG


class EnsembleModel(BaseEstimator):
    """
    Heterogeneous Ensemble model combining CBR, COCOMO, and ML models
    """
    
    def __init__(self, 
                 ml_model_name: str = "XGBoost",
                 combination_rule: str = "median",
                 weights: Optional[List[float]] = None):
        """
        Initialize Ensemble Model
        
        Args:
            ml_model_name: Name of ML model to use ('ANN', 'KNN', 'XGBoost', 'SVR')
            combination_rule: How to combine predictions ('median', 'linear', 'mean')
            weights: Weights for linear combination [w_cbr, w_cocomo, w_ml]
        """
        super().__init__(f"Ensemble_{ml_model_name}")
        
        self.ml_model_name = ml_model_name
        self.combination_rule = combination_rule
        self.weights = weights or ENSEMBLE_CONFIG["linear_weights"]
        
        # Initialize component models
        self.cbr_model = CBRModel()
        self.cocomo_model = COCOMOModel()
        self.ml_model = self._get_ml_model(ml_model_name)
        
    def _get_ml_model(self, name: str) -> BaseEstimator:
        """Get ML model by name"""
        models = {
            "ANN": ANNModel,
            "KNN": KNNModel,
            "XGBoost": XGBoostModel,
            "SVR": SVRModel,
            "LinearRegression": LinearRegressionModel
        }
        
        if name not in models:
            raise ValueError(f"Unknown ML model: {name}. Available: {list(models.keys())}")
        
        return models[name]()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """
        Fit all component models
        
        Args:
            X: Training features
            y: Training efforts
            
        Returns:
            self
        """
        start_time = time.time()
        
        # Fit each component model
        self.cbr_model.fit(X, y)
        self.cocomo_model.fit(X, y)
        self.ml_model.fit(X, y)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        return self
    
    def _combine_predictions(self, 
                             pred_cbr: np.ndarray,
                             pred_cocomo: np.ndarray,
                             pred_ml: np.ndarray) -> np.ndarray:
        """Combine predictions from all models"""
        
        if self.combination_rule == "median":
            stacked = np.stack([pred_cbr, pred_cocomo, pred_ml])
            return np.median(stacked, axis=0)
            
        elif self.combination_rule == "mean":
            return (pred_cbr + pred_cocomo + pred_ml) / 3
            
        elif self.combination_rule == "linear":
            w = self.weights
            return w[0] * pred_cbr + w[1] * pred_cocomo + w[2] * pred_ml
            
        else:
            raise ValueError(f"Unknown combination rule: {self.combination_rule}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict effort using ensemble
        
        Args:
            X: Features to predict
            
        Returns:
            Combined predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get predictions from each model
        pred_cbr = self.cbr_model.predict(X)
        pred_cocomo = self.cocomo_model.predict(X)
        pred_ml = self.ml_model.predict(X)
        
        # Combine predictions
        return self._combine_predictions(pred_cbr, pred_cocomo, pred_ml)
    
    def predict_individual(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual predictions from each model
        
        Returns:
            Dictionary of predictions from each model
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        return {
            "CBR": self.cbr_model.predict(X),
            "COCOMO": self.cocomo_model.predict(X),
            self.ml_model_name: self.ml_model.predict(X)
        }
    
    def get_component_training_times(self) -> Dict[str, float]:
        """Get training times for each component model"""
        return {
            "CBR": self.cbr_model.training_time,
            "COCOMO": self.cocomo_model.training_time,
            self.ml_model_name: self.ml_model.training_time,
            "Total": self.training_time
        }


def create_all_ensembles(combination_rule: str = "median") -> Dict[str, EnsembleModel]:
    """Create ensemble models with all ML variants"""
    ensembles = {}
    
    for ml_name in ENSEMBLE_CONFIG["ml_models"]:
        ensemble = EnsembleModel(
            ml_model_name=ml_name,
            combination_rule=combination_rule
        )
        ensembles[f"CBR_COCOMO_{ml_name}"] = ensemble
    
    return ensembles


if __name__ == "__main__":
    # Test ensemble model
    from src.data.data_loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.evaluation.metrics import calculate_all_metrics
    from sklearn.model_selection import train_test_split
    
    loader = DataLoader("cocomo81")
    df = loader.load_raw_data()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test single ensemble
    ensemble = EnsembleModel(ml_model_name="XGBoost", combination_rule="median")
    ensemble.fit(X_train, y_train)
    predictions = ensemble.predict(X_test)
    
    print("Ensemble Model Test:")
    print(f"Training times: {ensemble.get_component_training_times()}")
    
    metrics = calculate_all_metrics(y_test, predictions)
    print(f"\nEnsemble Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test individual predictions
    individual = ensemble.predict_individual(X_test)
    print(f"\nIndividual Model Predictions (first 3):")
    for model_name, preds in individual.items():
        print(f"  {model_name}: {preds[:3]}")