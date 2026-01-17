"""
Unit tests for evaluation metrics
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    calculate_mae,
    calculate_mmre,
    calculate_mdmre,
    calculate_pred,
    calculate_rmse,
    calculate_bmmre,
    calculate_all_metrics
)


# Fixtures
@pytest.fixture
def perfect_predictions():
    """Perfect predictions (actual == predicted)"""
    actuals = np.array([100, 200, 300, 400, 500])
    predictions = np.array([100, 200, 300, 400, 500])
    return actuals, predictions


@pytest.fixture
def sample_predictions():
    """Sample predictions with some error"""
    actuals = np.array([100, 200, 300, 400, 500])
    predictions = np.array([110, 190, 320, 380, 520])
    return actuals, predictions


@pytest.fixture
def high_error_predictions():
    """Predictions with high error"""
    actuals = np.array([100, 200, 300, 400, 500])
    predictions = np.array([200, 400, 100, 800, 250])
    return actuals, predictions


# MAE Tests
class TestMAE:
    def test_perfect_predictions(self, perfect_predictions):
        actuals, predictions = perfect_predictions
        mae = calculate_mae(actuals, predictions)
        assert mae == 0.0
    
    def test_sample_predictions(self, sample_predictions):
        actuals, predictions = sample_predictions
        mae = calculate_mae(actuals, predictions)
        # Expected: (10 + 10 + 20 + 20 + 20) / 5 = 16
        assert mae == 16.0
    
    def test_non_negative(self, high_error_predictions):
        actuals, predictions = high_error_predictions
        mae = calculate_mae(actuals, predictions)
        assert mae >= 0


# MMRE Tests
class TestMMRE:
    def test_perfect_predictions(self, perfect_predictions):
        actuals, predictions = perfect_predictions
        mmre = calculate_mmre(actuals, predictions)
        assert mmre == 0.0
    
    def test_sample_predictions(self, sample_predictions):
        actuals, predictions = sample_predictions
        mmre = calculate_mmre(actuals, predictions)
        # Expected MREs: 0.1, 0.05, 0.067, 0.05, 0.04
        assert 0 < mmre < 1
    
    def test_non_negative(self, high_error_predictions):
        actuals, predictions = high_error_predictions
        mmre = calculate_mmre(actuals, predictions)
        assert mmre >= 0


# MdMRE Tests
class TestMdMRE:
    def test_perfect_predictions(self, perfect_predictions):
        actuals, predictions = perfect_predictions
        mdmre = calculate_mdmre(actuals, predictions)
        assert mdmre == 0.0
    
    def test_median_calculation(self, sample_predictions):
        actuals, predictions = sample_predictions
        mdmre = calculate_mdmre(actuals, predictions)
        assert 0 < mdmre < 1


# PRED Tests
class TestPRED:
    def test_perfect_predictions(self, perfect_predictions):
        actuals, predictions = perfect_predictions
        pred25 = calculate_pred(actuals, predictions, threshold=0.25)
        assert pred25 == 1.0
    
    def test_sample_predictions(self, sample_predictions):
        actuals, predictions = sample_predictions
        pred25 = calculate_pred(actuals, predictions, threshold=0.25)
        # All predictions are within 25%
        assert pred25 == 1.0
    
    def test_high_error(self, high_error_predictions):
        actuals, predictions = high_error_predictions
        pred25 = calculate_pred(actuals, predictions, threshold=0.25)
        assert 0 <= pred25 <= 1
    
    def test_different_thresholds(self, sample_predictions):
        actuals, predictions = sample_predictions
        pred10 = calculate_pred(actuals, predictions, threshold=0.10)
        pred25 = calculate_pred(actuals, predictions, threshold=0.25)
        pred50 = calculate_pred(actuals, predictions, threshold=0.50)
        
        assert pred10 <= pred25 <= pred50


# RMSE Tests
class TestRMSE:
    def test_perfect_predictions(self, perfect_predictions):
        actuals, predictions = perfect_predictions
        rmse = calculate_rmse(actuals, predictions)
        assert rmse == 0.0
    
    def test_non_negative(self, sample_predictions):
        actuals, predictions = sample_predictions
        rmse = calculate_rmse(actuals, predictions)
        assert rmse >= 0


# BMMRE Tests
class TestBMMRE:
    def test_single_model(self, sample_predictions):
        actuals, predictions = sample_predictions
        bmmre = calculate_bmmre(actuals, [predictions])
        mmre = calculate_mmre(actuals, predictions)
        assert bmmre == mmre
    
    def test_multiple_models(self, sample_predictions):
        actuals, predictions = sample_predictions
        pred2 = predictions + 10
        pred3 = predictions - 5
        
        bmmre = calculate_bmmre(actuals, [predictions, pred2, pred3])
        mmre = calculate_mmre(actuals, predictions)
        
        # BMMRE should be less than or equal to MMRE
        assert bmmre <= mmre


# All Metrics Tests
class TestAllMetrics:
    def test_returns_all_metrics(self, sample_predictions):
        actuals, predictions = sample_predictions
        metrics = calculate_all_metrics(actuals, predictions)
        
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "MMRE" in metrics
        assert "MdMRE" in metrics
        assert "PRED(0.25)" in metrics
    
    def test_metrics_types(self, sample_predictions):
        actuals, predictions = sample_predictions
        metrics = calculate_all_metrics(actuals, predictions)
        
        for value in metrics.values():
            assert isinstance(value, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])