"""
Data preprocessing utilities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import COCOMO_MULTIPLIERS, PROCESSED_DATA_DIR


class DataPreprocessor:
    """Class for preprocessing software effort estimation data"""
    
    def __init__(self):
        self.scaler = None
        self.feature_names = None
        
    def convert_categorical_to_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert categorical COCOMO ratings to numerical multipliers
        
        Args:
            df: DataFrame with categorical values (vl, l, n, h, vh, xh)
            
        Returns:
            DataFrame with numerical values
        """
        df_converted = df.copy()
        
        for column, mapping in COCOMO_MULTIPLIERS.items():
            if column in df_converted.columns:
                df_converted[column] = df_converted[column].map(mapping)
                
        return df_converted
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: 'mean', 'median', or 'drop'
            
        Returns:
            DataFrame with handled missing values
        """
        df_clean = df.copy()
        
        if strategy == "mean":
            df_clean = df_clean.fillna(df_clean.mean())
        elif strategy == "median":
            df_clean = df_clean.fillna(df_clean.median())
        elif strategy == "drop":
            df_clean = df_clean.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        return df_clean
    
    def scale_features(self, X: np.ndarray, method: str = "standard", 
                       fit: bool = True) -> np.ndarray:
        """
        Scale features using specified method
        
        Args:
            X: Feature matrix
            method: 'standard' or 'minmax'
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Scaled feature matrix
        """
        if method == "standard":
            if fit or self.scaler is None:
                self.scaler = StandardScaler()
                return self.scaler.fit_transform(X)
            return self.scaler.transform(X)
            
        elif method == "minmax":
            if fit or self.scaler is None:
                self.scaler = MinMaxScaler()
                return self.scaler.fit_transform(X)
            return self.scaler.transform(X)
        
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                           scale: bool = True,
                           scale_method: str = "standard") -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Raw DataFrame
            scale: Whether to scale features
            scale_method: Scaling method to use
            
        Returns:
            Tuple of (X, y) preprocessed arrays
        """
        # Store feature names
        self.feature_names = list(df.columns[:-1])
        
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Split features and target
        X = df_clean.iloc[:, :-1].values
        y = df_clean.iloc[:, -1].values
        
        # Scale features if requested
        if scale:
            X = self.scale_features(X, method=scale_method)
            
        return X, y
    
    def save_processed_data(self, df: pd.DataFrame, dataset_name: str):
        """Save processed data to file"""
        output_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        
    def inverse_scale(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted yet")
        return self.scaler.inverse_transform(X_scaled)


if __name__ == "__main__":
    # Test preprocessing
    from src.data.data_loader import DataLoader
    
    loader = DataLoader("cocomo81")
    df = loader.load_raw_data()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
    print(f"Preprocessed X shape: {X.shape}")
    print(f"Preprocessed y shape: {y.shape}")
    print(f"X mean (should be ~0): {X.mean(axis=0)[:3]}")
    print(f"X std (should be ~1): {X.std(axis=0)[:3]}")