"""
Base model class for all estimation models
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
import time


class BaseEstimator(ABC):
    """Abstract base class for all effort estimation models"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.training_time = 0
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEstimator':
        """
        Train the model
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted effort values
        """
        pass
    
    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> np.ndarray:
        """Fit model and make predictions"""
        self.fit(X_train, y_train)
        return self.predict(X_test)
    
    def get_training_time(self) -> float:
        """Get training time in seconds"""
        return self.training_time
    
    def __repr__(self):
        return f"{self.name}(fitted={self.is_fitted})"