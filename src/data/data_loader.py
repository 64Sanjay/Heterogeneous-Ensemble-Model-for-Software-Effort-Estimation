"""
Data loading utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASETS


class DataLoader:
    """Class to load and manage datasets"""
    
    def __init__(self, dataset_name: str):
        """
        Initialize DataLoader
        
        Args:
            dataset_name: Name of dataset ('cocomo81', 'nasa93', 'nasa60')
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Dataset {dataset_name} not found. Available: {list(DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.config = DATASETS[dataset_name]
        self.data = None
        self.X = None
        self.y = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV file"""
        file_path = RAW_DATA_DIR / self.config["filename"]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        
        self.data = pd.read_csv(file_path)
        print(f"Loaded {self.dataset_name}: {len(self.data)} samples, {len(self.data.columns)} features")
        return self.data
    
    def load_processed_data(self) -> pd.DataFrame:
        """Load processed data"""
        file_path = PROCESSED_DATA_DIR / f"{self.dataset_name}_processed.csv"
        
        if not file_path.exists():
            print(f"Processed data not found. Loading raw data instead.")
            return self.load_raw_data()
        
        self.data = pd.read_csv(file_path)
        return self.data
    
    def get_features_and_target(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into features (X) and target (y)
        
        Returns:
            Tuple of (X, y) as numpy arrays
        """
        if self.data is None:
            self.load_raw_data()
        
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values
        
        return self.X, self.y
    
    def get_feature_names(self) -> list:
        """Get list of feature names"""
        if self.data is None:
            self.load_raw_data()
        return list(self.data.columns[:-1])
    
    def get_target_name(self) -> str:
        """Get target variable name"""
        if self.data is None:
            self.load_raw_data()
        return self.data.columns[-1]
    
    def get_data_summary(self) -> dict:
        """Get summary statistics of the dataset"""
        if self.data is None:
            self.load_raw_data()
            
        return {
            "num_samples": len(self.data),
            "num_features": len(self.data.columns) - 1,
            "feature_names": self.get_feature_names(),
            "target_name": self.get_target_name(),
            "target_stats": {
                "min": self.data.iloc[:, -1].min(),
                "max": self.data.iloc[:, -1].max(),
                "mean": self.data.iloc[:, -1].mean(),
                "median": self.data.iloc[:, -1].median(),
                "std": self.data.iloc[:, -1].std()
            }
        }


def load_all_datasets() -> dict:
    """Load all available datasets"""
    datasets = {}
    for name in DATASETS.keys():
        try:
            loader = DataLoader(name)
            X, y = loader.get_features_and_target()
            datasets[name] = {"X": X, "y": y, "loader": loader}
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    return datasets


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader("cocomo81")
    X, y = loader.get_features_and_target()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Summary: {loader.get_data_summary()}")