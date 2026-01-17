"""
Unit tests for preprocessing
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer, engineer_features


# Fixtures
@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame"""
    np.random.seed(42)
    data = {
        'rely': np.random.choice([0.75, 0.88, 1.0, 1.15, 1.4], 20),
        'data': np.random.choice([0.94, 1.0, 1.08, 1.16], 20),
        'cplx': np.random.choice([0.7, 0.85, 1.0, 1.15, 1.3], 20),
        'time': np.random.choice([1.0, 1.11, 1.3, 1.66], 20),
        'stor': np.random.choice([1.0, 1.06, 1.21, 1.56], 20),
        'loc': np.random.randint(10, 1000, 20),
        'actual': np.random.randint(100, 5000, 20)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_array():
    """Create sample numpy arrays"""
    np.random.seed(42)
    X = np.random.rand(50, 16)
    y = np.random.rand(50) * 1000 + 100
    return X, y


# Preprocessor Tests
class TestDataPreprocessor:
    def test_initialization(self):
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler is None
    
    def test_handle_missing_values_mean(self, sample_dataframe):
        df = sample_dataframe.copy()
        df.iloc[0, 0] = np.nan
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(df, strategy="mean")
        
        assert not df_clean.isnull().any().any()
    
    def test_handle_missing_values_median(self, sample_dataframe):
        df = sample_dataframe.copy()
        df.iloc[0, 0] = np.nan
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(df, strategy="median")
        
        assert not df_clean.isnull().any().any()
    
    def test_scale_features_standard(self, sample_array):
        X, y = sample_array
        
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.scale_features(X, method="standard")
        
        assert X_scaled.shape == X.shape
        assert np.abs(X_scaled.mean(axis=0)).max() < 0.1  # Close to 0
        assert np.abs(X_scaled.std(axis=0) - 1).max() < 0.1  # Close to 1
    
    def test_scale_features_minmax(self, sample_array):
        X, y = sample_array
        
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.scale_features(X, method="minmax")
        
        assert X_scaled.min() >= 0
        assert X_scaled.max() <= 1
    
    def test_preprocess_pipeline(self, sample_dataframe):
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_pipeline(sample_dataframe, scale=True)
        
        assert X.shape[0] == len(sample_dataframe)
        assert X.shape[1] == len(sample_dataframe.columns) - 1
        assert len(y) == len(sample_dataframe)
    
    def test_inverse_scale(self, sample_array):
        X, y = sample_array
        
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.scale_features(X, method="standard")
        X_inversed = preprocessor.inverse_scale(X_scaled)
        
        np.testing.assert_array_almost_equal(X, X_inversed, decimal=10)


# Feature Engineering Tests
class TestFeatureEngineer:
    def test_create_eaf_feature(self, sample_array):
        X, y = sample_array
        
        engineer = FeatureEngineer()
        X_eaf = engineer.create_eaf_feature(X)
        
        assert X_eaf.shape[1] == X.shape[1] + 1
    
    def test_create_size_derived_features(self, sample_array):
        X, y = sample_array
        
        engineer = FeatureEngineer()
        X_size = engineer.create_size_derived_features(X)
        
        # Should add 3 features: log_loc, sqrt_loc, loc_squared
        assert X_size.shape[1] == X.shape[1] + 3
    
    def test_select_features_kbest(self, sample_array):
        X, y = sample_array
        
        engineer = FeatureEngineer()
        X_selected, indices = engineer.select_features_kbest(X, y, k=5)
        
        assert X_selected.shape[1] == 5
        assert len(indices) == 5
    
    def test_get_feature_importance(self, sample_array):
        X, y = sample_array
        
        engineer = FeatureEngineer()
        importance = engineer.get_feature_importance(X, y)
        
        assert 'Feature' in importance.columns
        assert 'F_Score' in importance.columns
        assert 'MI_Score' in importance.columns
        assert len(importance) == X.shape[1]
    
    def test_remove_outliers_iqr(self, sample_array):
        X, y = sample_array
        
        # Add some outliers
        y_with_outliers = y.copy()
        y_with_outliers[0] = 100000
        
        engineer = FeatureEngineer()
        X_clean, y_clean = engineer.remove_outliers(X, y_with_outliers, method="iqr")
        
        assert len(y_clean) <= len(y)


# Integration Tests
class TestFeatureEngineeringPipeline:
    def test_engineer_features(self, sample_array):
        X, y = sample_array
        
        X_engineered, engineer = engineer_features(
            X, y,
            add_eaf=True,
            add_size_features=True,
            add_interactions=False
        )
        
        assert X_engineered.shape[0] == X.shape[0]
        assert X_engineered.shape[1] > X.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])