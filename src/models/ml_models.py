# """
# Machine Learning Models for Software Effort Estimation
# """

# import numpy as np
# import time
# from typing import Dict, Any

# from sklearn.neural_network import MLPRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor

# from src.models.base_model import BaseEstimator
# from src.utils.config import MODEL_PARAMS, RANDOM_SEED


# class ANNModel(BaseEstimator):
#     """Artificial Neural Network model using MLPRegressor"""
    
#     def __init__(self, **kwargs):
#         super().__init__("ANN")
#         params = MODEL_PARAMS["ann"].copy()
#         params.update(kwargs)
#         self.model = MLPRegressor(**params)
        
#     def fit(self, X: np.ndarray, y: np.ndarray) -> 'ANNModel':
#         start_time = time.time()
#         self.model.fit(X, y)
#         self.is_fitted = True
#         self.training_time = time.time() - start_time
#         return self
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
#         return self.model.predict(X)


# class KNNModel(BaseEstimator):
#     """K-Nearest Neighbors model"""
    
#     def __init__(self, **kwargs):
#         super().__init__("KNN")
#         params = MODEL_PARAMS["knn"].copy()
#         params.update(kwargs)
#         self.model = KNeighborsRegressor(**params)
        
#     def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNModel':
#         start_time = time.time()
#         self.model.fit(X, y)
#         self.is_fitted = True
#         self.training_time = time.time() - start_time
#         return self
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
#         return self.model.predict(X)


# class XGBoostModel(BaseEstimator):
#     """XGBoost model"""
    
#     def __init__(self, **kwargs):
#         super().__init__("XGBoost")
#         params = MODEL_PARAMS["xgboost"].copy()
#         params.update(kwargs)
#         self.model = XGBRegressor(**params)
        
#     def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostModel':
#         start_time = time.time()
#         self.model.fit(X, y)
#         self.is_fitted = True
#         self.training_time = time.time() - start_time
#         return self
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
#         return self.model.predict(X)


# class SVRModel(BaseEstimator):
#     """Support Vector Regression model"""
    
#     def __init__(self, **kwargs):
#         super().__init__("SVR")
#         params = MODEL_PARAMS["svr"].copy()
#         params.update(kwargs)
#         self.model = SVR(**params)
        
#     def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVRModel':
#         start_time = time.time()
#         self.model.fit(X, y)
#         self.is_fitted = True
#         self.training_time = time.time() - start_time
#         return self
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
#         return self.model.predict(X)


# class LinearRegressionModel(BaseEstimator):
#     """Linear Regression model"""
    
#     def __init__(self):
#         super().__init__("LinearRegression")
#         self.model = LinearRegression()
        
#     def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionModel':
#         start_time = time.time()
#         self.model.fit(X, y)
#         self.is_fitted = True
#         self.training_time = time.time() - start_time
#         return self
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
#         return self.model.predict(X)


# def get_all_ml_models() -> Dict[str, BaseEstimator]:
#     """Get dictionary of all ML models"""
#     return {
#         "ANN": ANNModel(),
#         "KNN": KNNModel(),
#         "XGBoost": XGBoostModel(),
#         "SVR": SVRModel(),
#         "LinearRegression": LinearRegressionModel()
#     }


# if __name__ == "__main__":
#     # Test all ML models
#     from src.data.data_loader import DataLoader
#     from src.data.preprocessor import DataPreprocessor
#     from src.evaluation.metrics import calculate_all_metrics
#     from sklearn.model_selection import train_test_split
    
#     loader = DataLoader("cocomo81")
#     df = loader.load_raw_data()
    
#     preprocessor = DataPreprocessor()
#     X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     models = get_all_ml_models()
    
#     print("ML Models Test Results:")
#     print("-" * 60)
    
#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         metrics = calculate_all_metrics(y_test, predictions)
        
#         print(f"\n{name}:")
#         print(f"  Training time: {model.training_time:.4f}s")
#         print(f"  MAE: {metrics['MAE']:.2f}")
#         print(f"  MMRE: {metrics['MMRE']:.4f}")
# --------v2--------
# """
# Machine Learning Models for Software Effort Estimation
# """

# import numpy as np
# import time
# import warnings
# from typing import Dict, Any

# # Suppress warnings
# warnings.filterwarnings('ignore')

# from sklearn.neural_network import MLPRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor

# from src.models.base_model import BaseEstimator
# from src.utils.config import MODEL_PARAMS, RANDOM_SEED


# class ANNModel(BaseEstimator):
#     """Artificial Neural Network model using MLPRegressor"""
    
#     def __init__(self, **kwargs):
#         super().__init__("ANN")
#         params = MODEL_PARAMS["ann"].copy()
#         params.update(kwargs)
#         # Increase max_iter to avoid convergence warning
#         params['max_iter'] = 2000
#         params['early_stopping'] = True
#         params['validation_fraction'] = 0.1
#         self.model = MLPRegressor(**params)
        
#     def fit(self, X: np.ndarray, y: np.ndarray) -> 'ANNModel':
#         start_time = time.time()
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             self.model.fit(X, y)
#         self.is_fitted = True
#         self.training_time = time.time() - start_time
#         return self
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
#         return self.model.predict(X)


# class KNNModel(BaseEstimator):
#     """K-Nearest Neighbors model"""
    
#     def __init__(self, **kwargs):
#         super().__init__("KNN")
#         params = MODEL_PARAMS["knn"].copy()
#         params.update(kwargs)
#         self.model = KNeighborsRegressor(**params)
        
#     def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNModel':
#         start_time = time.time()
#         self.model.fit(X, y)
#         self.is_fitted = True
#         self.training_time = time.time() - start_time
#         return self
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
#         return self.model.predict(X)


# class XGBoostModel(BaseEstimator):
#     """XGBoost model"""
    
#     def __init__(self, **kwargs):
#         super().__init__("XGBoost")
#         params = MODEL_PARAMS["xgboost"].copy()
#         params.update(kwargs)
#         # Suppress XGBoost warnings
#         params['verbosity'] = 0
#         self.model = XGBRegressor(**params)
        
#     def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostModel':
#         start_time = time.time()
#         self.model.fit(X, y)
#         self.is_fitted = True
#         self.training_time = time.time() - start_time
#         return self
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
#         return self.model.predict(X)


# class SVRModel(BaseEstimator):
#     """Support Vector Regression model"""
    
#     def __init__(self, **kwargs):
#         super().__init__("SVR")
#         params = MODEL_PARAMS["svr"].copy()
#         params.update(kwargs)
#         self.model = SVR(**params)
        
#     def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVRModel':
#         start_time = time.time()
#         self.model.fit(X, y)
#         self.is_fitted = True
#         self.training_time = time.time() - start_time
#         return self
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
#         return self.model.predict(X)


# class LinearRegressionModel(BaseEstimator):
#     """Linear Regression model"""
    
#     def __init__(self):
#         super().__init__("LinearRegression")
#         self.model = LinearRegression()
        
#     def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionModel':
#         start_time = time.time()
#         self.model.fit(X, y)
#         self.is_fitted = True
#         self.training_time = time.time() - start_time
#         return self
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if X.ndim == 1:
#             X = X.reshape(1, -1)
#         return self.model.predict(X)


# def get_all_ml_models() -> Dict[str, BaseEstimator]:
#     """Get dictionary of all ML models"""
#     return {
#         "ANN": ANNModel(),
#         "KNN": KNNModel(),
#         "XGBoost": XGBoostModel(),
#         "SVR": SVRModel(),
#         "LinearRegression": LinearRegressionModel()
#     }


# if __name__ == "__main__":
#     # Test all ML models
#     from src.data.data_loader import DataLoader
#     from src.data.preprocessor import DataPreprocessor
#     from src.evaluation.metrics import calculate_all_metrics
#     from sklearn.model_selection import train_test_split
    
#     loader = DataLoader("cocomo81")
#     df = loader.load_raw_data()
    
#     preprocessor = DataPreprocessor()
#     X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     models = get_all_ml_models()
    
#     print("ML Models Test Results:")
#     print("-" * 60)
    
#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         metrics = calculate_all_metrics(y_test, predictions)
        
#         print(f"\n{name}:")
#         print(f"  Training time: {model.training_time:.4f}s")
#         print(f"  MAE: {metrics['MAE']:.2f}")
#         print(f"  MMRE: {metrics['MMRE']:.4f}")
# ---v3--
"""
Machine Learning Models for Software Effort Estimation (GPU Optimized)
"""

import numpy as np
import time
import warnings
import os
from typing import Dict, Any

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.models.base_model import BaseEstimator
from src.utils.config import MODEL_PARAMS, RANDOM_SEED


class ANNModel(BaseEstimator):
    """GPU-Optimized Artificial Neural Network model using TensorFlow"""
    
    def __init__(self, **kwargs):
        super().__init__("ANN")
        self.model = None
        self.input_dim = None
        
    def _build_model(self, input_dim: int):
        """Build TensorFlow model for GPU"""
        self.input_dim = input_dim
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mae'
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ANNModel':
        start_time = time.time()
        
        self._build_model(X.shape[1])
        
        # Convert to TensorFlow tensors
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
        
        # Create optimized dataset
        dataset = tf.data.Dataset.from_tensor_slices((X_tf, y_tf))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=50,
            restore_best_weights=True
        )
        
        # Train
        self.model.fit(
            dataset,
            epochs=500,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        return self.model.predict(X_tf, verbose=0).flatten()


class KNNModel(BaseEstimator):
    """K-Nearest Neighbors model"""
    
    def __init__(self, **kwargs):
        super().__init__("KNN")
        params = MODEL_PARAMS["knn"].copy()
        params.update(kwargs)
        self.model = KNeighborsRegressor(**params)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNModel':
        start_time = time.time()
        self.model.fit(X, y)
        self.is_fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)


class XGBoostModel(BaseEstimator):
    """XGBoost model with GPU support"""
    
    def __init__(self, **kwargs):
        super().__init__("XGBoost")
        params = MODEL_PARAMS["xgboost"].copy()
        params.update(kwargs)
        params['verbosity'] = 0
        # Enable GPU if available
        params['tree_method'] = 'hist'  # Use 'gpu_hist' if GPU XGBoost is installed
        params['device'] = 'cuda'  # Use GPU
        self.model = XGBRegressor(**params)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostModel':
        start_time = time.time()
        try:
            self.model.fit(X, y)
        except Exception:
            # Fallback to CPU if GPU fails
            self.model.set_params(tree_method='hist', device='cpu')
            self.model.fit(X, y)
        self.is_fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)


class SVRModel(BaseEstimator):
    """Support Vector Regression model"""
    
    def __init__(self, **kwargs):
        super().__init__("SVR")
        params = MODEL_PARAMS["svr"].copy()
        params.update(kwargs)
        self.model = SVR(**params)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVRModel':
        start_time = time.time()
        self.model.fit(X, y)
        self.is_fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)


class LinearRegressionModel(BaseEstimator):
    """Linear Regression model"""
    
    def __init__(self):
        super().__init__("LinearRegression")
        self.model = LinearRegression()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionModel':
        start_time = time.time()
        self.model.fit(X, y)
        self.is_fitted = True
        self.training_time = time.time() - start_time
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)


def get_all_ml_models() -> Dict[str, BaseEstimator]:
    """Get dictionary of all ML models"""
    return {
        "ANN": ANNModel(),
        "KNN": KNNModel(),
        "XGBoost": XGBoostModel(),
        "SVR": SVRModel(),
        "LinearRegression": LinearRegressionModel()
    }