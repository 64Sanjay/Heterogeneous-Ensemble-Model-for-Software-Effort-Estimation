"""
COCOMO II Model for Software Effort Estimation (Fixed for Scaled Data)
"""

import numpy as np
import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from typing import Optional
import time

from src.models.base_model import BaseEstimator
from src.utils.config import MODEL_PARAMS, RANDOM_SEED


class COCOMOModel(BaseEstimator):
    """
    COCOMO II model - Uses neural network regression on scaled data
    Since data is scaled, we use a learned model instead of the formula
    """
    
    def __init__(self, a: float = 2.94, b: float = 1.12,
                 use_nn_correction: bool = True):
        super().__init__("COCOMO")
        self.a = a
        self.b = b
        self.use_nn_correction = use_nn_correction
        self.nn_model = None
        
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
    def _build_nn(self, input_dim: int):
        """Build neural network"""
        self.nn_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        self.nn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mae'
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'COCOMOModel':
        """Fit COCOMO model using neural network"""
        start_time = time.time()
        
        self._build_nn(X.shape[1])
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=30,
            restore_best_weights=True
        )
        
        # Reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001
        )
        
        self.nn_model.fit(
            X, y,
            epochs=MODEL_PARAMS["cocomo"]["epochs"],
            batch_size=MODEL_PARAMS["cocomo"]["batch_size"],
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict effort"""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = self.nn_model.predict(X, verbose=0).flatten()
        return np.maximum(predictions, 0)


class PureCOCOMO(BaseEstimator):
    """Pure COCOMO II model for raw (unscaled) data"""
    
    def __init__(self, a: float = 2.94, b: float = 1.12):
        super().__init__("PureCOCOMO")
        self.a = a
        self.b = b
        self.loc_column_idx = -1
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PureCOCOMO':
        start_time = time.time()
        self.loc_column_idx = X.shape[1] - 1
        self.is_fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        kloc = np.abs(X[:, self.loc_column_idx])
        kloc = np.where(kloc <= 0, 0.1, kloc)
        
        cost_drivers = np.abs(X[:, :-1])
        cost_drivers = np.where(cost_drivers == 0, 1, cost_drivers)
        eaf = np.prod(cost_drivers, axis=1)
        
        with np.errstate(invalid='ignore'):
            effort = self.a * np.power(kloc, self.b) * eaf
        
        effort = np.nan_to_num(effort, nan=0.0, posinf=1e6, neginf=0.0)
        return np.maximum(effort, 0)
