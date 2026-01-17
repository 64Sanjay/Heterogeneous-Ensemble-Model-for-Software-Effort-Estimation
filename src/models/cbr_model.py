"""
Case-Based Reasoning (CBR) Model for Software Effort Estimation
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, Tuple
import random
import time

from src.models.base_model import BaseEstimator
from src.utils.config import MODEL_PARAMS, RANDOM_SEED


class CBRModel(BaseEstimator):
    """
    Case-Based Reasoning model for effort estimation
    
    Uses similarity-based retrieval to find similar past projects
    and estimates effort based on their actual efforts.
    """
    
    def __init__(self, k: Optional[int] = None, 
                 similarity_metric: str = "euclidean",
                 weighting_scheme: str = "uniform"):
        """
        Initialize CBR Model
        
        Args:
            k: Number of nearest neighbors (None for auto-selection)
            similarity_metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            weighting_scheme: How to weight neighbors ('uniform', 'distance', 'rank')
        """
        super().__init__("CBR")
        self.k = k
        self.similarity_metric = similarity_metric
        self.weighting_scheme = weighting_scheme
        self.X_train = None
        self.y_train = None
        self.feature_weights = None
        
        # Set random seed
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CBRModel':
        """
        Fit CBR model (store training cases)
        
        Args:
            X: Training features
            y: Training efforts
            
        Returns:
            self
        """
        start_time = time.time()
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Initialize feature weights
        self.feature_weights = np.ones(X.shape[1]) / X.shape[1]
        
        # Auto-select k if not specified
        if self.k is None:
            k_min, k_max = MODEL_PARAMS["cbr"]["k_range"]
            self.k = min(random.randint(k_min, k_max), len(X) - 1)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        return self
    
    def _calculate_distances(self, X_test: np.ndarray) -> np.ndarray:
        """Calculate distances between test cases and training cases"""
        weighted_X_train = self.X_train * self.feature_weights
        weighted_X_test = X_test * self.feature_weights
        
        return cdist(weighted_X_test, weighted_X_train, metric=self.similarity_metric)
    
    def _get_neighbor_weights(self, distances: np.ndarray) -> np.ndarray:
        """Calculate weights for k nearest neighbors"""
        k = self.k
        
        if self.weighting_scheme == "uniform":
            return np.ones(k) / k
            
        elif self.weighting_scheme == "distance":
            # Inverse distance weighting
            weights = 1 / (distances + 1e-9)
            return weights / weights.sum()
            
        elif self.weighting_scheme == "rank":
            # Rank-based weighting
            weights = np.array([(2 ** (k - i - 1)) / (2 ** k - 1) for i in range(k)])
            return weights
            
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting_scheme}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict effort for new projects
        
        Args:
            X: Features of new projects
            
        Returns:
            Predicted efforts
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        distances = self._calculate_distances(X)
        predictions = []
        
        for i, dist in enumerate(distances):
            # Get k nearest neighbors
            neighbor_indices = np.argsort(dist)[:self.k]
            neighbor_efforts = self.y_train[neighbor_indices]
            neighbor_distances = dist[neighbor_indices]
            
            # Calculate weighted prediction
            weights = self._get_neighbor_weights(neighbor_distances)
            predicted_effort = np.sum(weights * neighbor_efforts)
            predictions.append(predicted_effort)
        
        return np.array(predictions)
    
    def set_feature_weights(self, weights: np.ndarray):
        """Set custom feature weights"""
        if len(weights) != self.X_train.shape[1]:
            raise ValueError("Weights length must match number of features")
        self.feature_weights = weights / weights.sum()
        
    def optimize_k(self, X_val: np.ndarray, y_val: np.ndarray, 
                   k_range: Tuple[int, int] = (1, 20)) -> int:
        """
        Find optimal k using validation data
        
        Args:
            X_val: Validation features
            y_val: Validation efforts
            k_range: Range of k values to try
            
        Returns:
            Optimal k value
        """
        from src.evaluation.metrics import calculate_mae
        
        best_k = k_range[0]
        best_mae = float('inf')
        
        for k in range(k_range[0], min(k_range[1], len(self.X_train))):
            self.k = k
            predictions = self.predict(X_val)
            mae = calculate_mae(y_val, predictions)
            
            if mae < best_mae:
                best_mae = mae
                best_k = k
        
        self.k = best_k
        return best_k


if __name__ == "__main__":
    # Test CBR model
    from src.data.data_loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from sklearn.model_selection import train_test_split
    
    loader = DataLoader("cocomo81")
    df = loader.load_raw_data()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CBRModel(k=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(f"CBR Model Test")
    print(f"Training time: {model.training_time:.4f}s")
    print(f"Predictions: {predictions[:5]}")
    print(f"Actuals: {y_test[:5]}")