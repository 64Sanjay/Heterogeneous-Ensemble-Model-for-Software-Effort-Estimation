"""
Cross-validation utilities for model evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import LeaveOneOut, KFold
import time

from src.models.base_model import BaseEstimator
from src.evaluation.metrics import calculate_all_metrics, calculate_bmmre


class CrossValidator:
    """Cross-validation handler for effort estimation models"""
    
    def __init__(self, cv_type: str = "loocv", n_splits: int = 5, 
                 random_state: int = 42):
        """
        Initialize CrossValidator
        
        Args:
            cv_type: 'loocv' or 'kfold'
            n_splits: Number of folds for k-fold CV
            random_state: Random seed for reproducibility
        """
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.random_state = random_state
        
        if cv_type == "loocv":
            self.cv = LeaveOneOut()
        elif cv_type == "kfold":
            self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            raise ValueError(f"Unknown CV type: {cv_type}")
    
    def evaluate_model(self, model: BaseEstimator, 
                       X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate a single model using cross-validation
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target values
            
        Returns:
            Dictionary with metrics and predictions
        """
        predictions = []
        actuals = []
        training_times = []
        
        for train_idx, test_idx in self.cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_times.append(time.time() - start_time)
            
            pred = model.predict(X_test)
            predictions.extend(pred.flatten())
            actuals.extend(y_test.flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        metrics = calculate_all_metrics(actuals, predictions)
        metrics["Training_Time"] = np.mean(training_times)
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "actuals": actuals
        }
    
    def evaluate_multiple_models(self, models: Dict[str, BaseEstimator],
                                  X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Evaluate multiple models and compare results
        
        Args:
            models: Dictionary of {model_name: model}
            X: Features
            y: Target values
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        all_predictions = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            result = self.evaluate_model(model, X, y)
            
            row = {"Model": name}
            row.update(result["metrics"])
            results.append(row)
            
            all_predictions[name] = result["predictions"]
        
        df = pd.DataFrame(results)
        return df, all_predictions
    
    def evaluate_ensemble_vs_individual(self, 
                                        ensemble_models: Dict[str, BaseEstimator],
                                        individual_models: Dict[str, BaseEstimator],
                                        X: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compare ensemble models against individual models
        
        Returns:
            Tuple of (ensemble_results_df, individual_results_df)
        """
        print("Evaluating ensemble models...")
        ensemble_df, ensemble_preds = self.evaluate_multiple_models(ensemble_models, X, y)
        
        print("\nEvaluating individual models...")
        individual_df, individual_preds = self.evaluate_multiple_models(individual_models, X, y)
        
        return ensemble_df, individual_df, ensemble_preds, individual_preds


def run_complete_evaluation(X: np.ndarray, y: np.ndarray,
                            dataset_name: str,
                            cv_types: List[str] = ["loocv", "kfold"]) -> Dict:
    """
    Run complete evaluation with multiple CV strategies
    
    Returns:
        Dictionary with all results
    """
    from src.models.ensemble_model import create_all_ensembles
    from src.models.cbr_model import CBRModel
    from src.models.cocomo_model import COCOMOModel
    from src.models.ml_models import get_all_ml_models
    
    results = {}
    
    for cv_type in cv_types:
        print(f"\n{'='*60}")
        print(f"Running {cv_type.upper()} evaluation for {dataset_name}")
        print('='*60)
        
        cv = CrossValidator(cv_type=cv_type)
        
        # Create models
        ensemble_models = create_all_ensembles(combination_rule="median")
        individual_models = {
            "CBR": CBRModel(),
            "COCOMO": COCOMOModel(),
            **get_all_ml_models()
        }
        
        # Evaluate
        ensemble_df, individual_df, _, _ = cv.evaluate_ensemble_vs_individual(
            ensemble_models, individual_models, X, y
        )
        
        results[cv_type] = {
            "ensemble": ensemble_df,
            "individual": individual_df
        }
        
        print(f"\nEnsemble Results ({cv_type}):")
        print(ensemble_df.to_string(index=False))
        
        print(f"\nIndividual Results ({cv_type}):")
        print(individual_df.to_string(index=False))
    
    return results


if __name__ == "__main__":
    # Test cross-validation
    from src.data.data_loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    
    loader = DataLoader("cocomo81")
    df = loader.load_raw_data()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
    results = run_complete_evaluation(X, y, "cocomo81", cv_types=["kfold"])