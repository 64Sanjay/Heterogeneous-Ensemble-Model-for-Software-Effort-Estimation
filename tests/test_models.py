"""
Unit tests for models
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel, PureCOCOMO
from src.models.ml_models import XGBoostModel, ANNModel, KNNModel, SVRModel
from src.models.ensemble_model import EnsembleModel


# Fixtures
@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    X = np.random.rand(50, 16)
    y = np.random.rand(50) * 1000 + 100
    return X, y


@pytest.fixture
def train_test_split(sample_data):
    """Split sample data into train and test"""
    X, y = sample_data
    split_idx = 40
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


# CBR Model Tests
class TestCBRModel:
    def test_initialization(self):
        model = CBRModel(k=5)
        assert model.k == 5
        assert model.name == "CBR"
        assert not model.is_fitted
    
    def test_fit(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        model = CBRModel(k=3)
        model.fit(X_train, y_train)
        
        assert model.is_fitted
        assert model.X_train is not None
        assert model.y_train is not None
    
    def test_predict(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        model = CBRModel(k=3)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert all(p > 0 for p in predictions)
    
    def test_predict_single_sample(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        model = CBRModel(k=3)
        model.fit(X_train, y_train)
        
        prediction = model.predict(X_test[0])
        
        assert len(prediction) == 1


# COCOMO Model Tests
class TestCOCOMOModel:
    def test_initialization(self):
        model = COCOMOModel()
        assert model.name == "COCOMO"
        assert model.a == 2.94
        assert model.b == 1.12
    
    def test_fit_predict(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        model = COCOMOModel(use_nn_correction=False)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(y_test)
    
    def test_pure_cocomo(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        model = PureCOCOMO()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(y_test)


# ML Model Tests
class TestMLModels:
    def test_xgboost(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        model = XGBoostModel()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert model.is_fitted
    
    def test_ann(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        model = ANNModel()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(y_test)
    
    def test_knn(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        model = KNNModel()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(y_test)
    
    def test_svr(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        model = SVRModel()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(y_test)


# Ensemble Model Tests
class TestEnsembleModel:
    def test_initialization(self):
        model = EnsembleModel(ml_model_name="XGBoost")
        assert "Ensemble" in model.name
    
    def test_fit_predict(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        model = EnsembleModel(ml_model_name="XGBoost", combination_rule="median")
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert model.is_fitted
    
    def test_individual_predictions(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        model = EnsembleModel(ml_model_name="XGBoost")
        model.fit(X_train, y_train)
        
        individual = model.predict_individual(X_test)
        
        assert "CBR" in individual
        assert "COCOMO" in individual
        assert "XGBoost" in individual
    
    def test_combination_rules(self, train_test_split):
        X_train, X_test, y_train, y_test = train_test_split
        
        for rule in ["median", "mean", "linear"]:
            model = EnsembleModel(ml_model_name="KNN", combination_rule=rule)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            assert len(predictions) == len(y_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])