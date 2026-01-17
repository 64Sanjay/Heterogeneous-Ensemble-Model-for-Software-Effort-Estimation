# """
# Evaluation metrics for software effort estimation
# """

# import numpy as np
# from typing import List, Tuple
# from sklearn.metrics import mean_absolute_error, mean_squared_error


# def calculate_mre(actual: float, predicted: float) -> float:
#     """
#     Calculate Magnitude of Relative Error for single prediction
    
#     MRE = |actual - predicted| / actual
#     """
#     if actual == 0:
#         return 0
#     return abs(actual - predicted) / actual


# def calculate_mmre(actuals: np.ndarray, predictions: np.ndarray) -> float:
#     """
#     Calculate Mean Magnitude of Relative Error
    
#     MMRE = (1/n) * Σ|actual - predicted| / actual
#     """
#     actuals = np.array(actuals)
#     predictions = np.array(predictions)
    
#     # Avoid division by zero
#     safe_actuals = np.where(actuals == 0, 1e-10, actuals)
#     relative_errors = np.abs((actuals - predictions) / safe_actuals)
    
#     return np.mean(relative_errors)


# def calculate_mdmre(actuals: np.ndarray, predictions: np.ndarray) -> float:
#     """
#     Calculate Median Magnitude of Relative Error
    
#     MdMRE = median(|actual - predicted| / actual)
#     """
#     actuals = np.array(actuals)
#     predictions = np.array(predictions)
    
#     safe_actuals = np.where(actuals == 0, 1e-10, actuals)
#     relative_errors = np.abs((actuals - predictions) / safe_actuals)
    
#     return np.median(relative_errors)


# def calculate_pred(actuals: np.ndarray, predictions: np.ndarray, 
#                    threshold: float = 0.25) -> float:
#     """
#     Calculate PRED(k) - percentage of predictions within k% of actual
    
#     PRED(0.25) = percentage of predictions where MRE <= 0.25
#     """
#     actuals = np.array(actuals)
#     predictions = np.array(predictions)
    
#     safe_actuals = np.where(actuals == 0, 1e-10, actuals)
#     relative_errors = np.abs((actuals - predictions) / safe_actuals)
    
#     within_threshold = np.sum(relative_errors <= threshold)
#     return within_threshold / len(actuals)


# def calculate_mae(actuals: np.ndarray, predictions: np.ndarray) -> float:
#     """Calculate Mean Absolute Error"""
#     return mean_absolute_error(actuals, predictions)


# def calculate_rmse(actuals: np.ndarray, predictions: np.ndarray) -> float:
#     """Calculate Root Mean Squared Error"""
#     return np.sqrt(mean_squared_error(actuals, predictions))


# def calculate_bmmre(actuals: np.ndarray, 
#                     all_predictions: List[np.ndarray]) -> float:
#     """
#     Calculate Best Mean Magnitude of Relative Error
    
#     For each sample, takes the minimum MRE across all models
#     """
#     actuals = np.array(actuals)
#     all_predictions = np.array(all_predictions)
    
#     safe_actuals = np.where(actuals == 0, 1e-10, actuals)
    
#     # Calculate relative errors for each model
#     relative_errors = []
#     for pred in all_predictions:
#         re = np.abs((actuals - pred) / safe_actuals)
#         relative_errors.append(re)
    
#     # Take minimum error for each sample
#     min_relative_errors = np.min(relative_errors, axis=0)
    
#     return np.mean(min_relative_errors)


# def calculate_all_metrics(actuals: np.ndarray, 
#                           predictions: np.ndarray) -> dict:
#     """
#     Calculate all evaluation metrics
    
#     Returns:
#         Dictionary containing all metrics
#     """
#     return {
#         "MAE": calculate_mae(actuals, predictions),
#         "RMSE": calculate_rmse(actuals, predictions),
#         "MMRE": calculate_mmre(actuals, predictions),
#         "MdMRE": calculate_mdmre(actuals, predictions),
#         "PRED(0.25)": calculate_pred(actuals, predictions, 0.25),
#         "PRED(0.30)": calculate_pred(actuals, predictions, 0.30)
#     }


# def compare_models(actuals: np.ndarray, 
#                    predictions_dict: dict) -> pd.DataFrame:
#     """
#     Compare multiple models using all metrics
    
#     Args:
#         actuals: Actual effort values
#         predictions_dict: Dictionary of {model_name: predictions}
        
#     Returns:
#         DataFrame with metrics for each model
#     """
#     import pandas as pd
    
#     results = []
#     for model_name, predictions in predictions_dict.items():
#         metrics = calculate_all_metrics(actuals, predictions)
#         metrics["Model"] = model_name
#         results.append(metrics)
    
#     df = pd.DataFrame(results)
#     df = df[["Model", "MAE", "RMSE", "MMRE", "MdMRE", "PRED(0.25)", "PRED(0.30)"]]
    
#     return df


# if __name__ == "__main__":
#     # Test metrics
#     actuals = np.array([100, 200, 300, 400, 500])
#     predictions = np.array([110, 190, 320, 380, 520])
    
#     print("Test Metrics:")
#     print(f"MAE: {calculate_mae(actuals, predictions):.4f}")
#     print(f"MMRE: {calculate_mmre(actuals, predictions):.4f}")
#     print(f"MdMRE: {calculate_mdmre(actuals, predictions):.4f}")
#     print(f"PRED(0.25): {calculate_pred(actuals, predictions):.4f}")
# ----v2----
"""
Evaluation metrics for software effort estimation
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mre(actual: float, predicted: float) -> float:
    """
    Calculate Magnitude of Relative Error for single prediction
    
    MRE = |actual - predicted| / actual
    """
    if actual == 0:
        return 0
    return abs(actual - predicted) / actual


def calculate_mmre(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate Mean Magnitude of Relative Error
    
    MMRE = (1/n) * Σ|actual - predicted| / actual
    """
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    
    # Avoid division by zero
    safe_actuals = np.where(actuals == 0, 1e-10, actuals)
    relative_errors = np.abs((actuals - predictions) / safe_actuals)
    
    return np.mean(relative_errors)


def calculate_mdmre(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate Median Magnitude of Relative Error
    
    MdMRE = median(|actual - predicted| / actual)
    """
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    
    safe_actuals = np.where(actuals == 0, 1e-10, actuals)
    relative_errors = np.abs((actuals - predictions) / safe_actuals)
    
    return np.median(relative_errors)


def calculate_pred(actuals: np.ndarray, predictions: np.ndarray, 
                   threshold: float = 0.25) -> float:
    """
    Calculate PRED(k) - percentage of predictions within k% of actual
    
    PRED(0.25) = percentage of predictions where MRE <= 0.25
    """
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    
    safe_actuals = np.where(actuals == 0, 1e-10, actuals)
    relative_errors = np.abs((actuals - predictions) / safe_actuals)
    
    within_threshold = np.sum(relative_errors <= threshold)
    return within_threshold / len(actuals)


def calculate_mae(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    return mean_absolute_error(actuals, predictions)


def calculate_rmse(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(actuals, predictions))


def calculate_bmmre(actuals: np.ndarray, 
                    all_predictions: List[np.ndarray]) -> float:
    """
    Calculate Best Mean Magnitude of Relative Error
    
    For each sample, takes the minimum MRE across all models
    """
    actuals = np.array(actuals)
    all_predictions = np.array(all_predictions)
    
    safe_actuals = np.where(actuals == 0, 1e-10, actuals)
    
    # Calculate relative errors for each model
    relative_errors = []
    for pred in all_predictions:
        re = np.abs((actuals - pred) / safe_actuals)
        relative_errors.append(re)
    
    # Take minimum error for each sample
    min_relative_errors = np.min(relative_errors, axis=0)
    
    return np.mean(min_relative_errors)


def calculate_all_metrics(actuals: np.ndarray, 
                          predictions: np.ndarray) -> Dict:
    """
    Calculate all evaluation metrics
    
    Returns:
        Dictionary containing all metrics
    """
    return {
        "MAE": calculate_mae(actuals, predictions),
        "RMSE": calculate_rmse(actuals, predictions),
        "MMRE": calculate_mmre(actuals, predictions),
        "MdMRE": calculate_mdmre(actuals, predictions),
        "PRED(0.25)": calculate_pred(actuals, predictions, 0.25),
        "PRED(0.30)": calculate_pred(actuals, predictions, 0.30)
    }


def compare_models(actuals: np.ndarray, 
                   predictions_dict: Dict) -> pd.DataFrame:
    """
    Compare multiple models using all metrics
    
    Args:
        actuals: Actual effort values
        predictions_dict: Dictionary of {model_name: predictions}
        
    Returns:
        DataFrame with metrics for each model
    """
    results = []
    for model_name, predictions in predictions_dict.items():
        metrics = calculate_all_metrics(actuals, predictions)
        metrics["Model"] = model_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df[["Model", "MAE", "RMSE", "MMRE", "MdMRE", "PRED(0.25)", "PRED(0.30)"]]
    
    return df


if __name__ == "__main__":
    # Test metrics
    actuals = np.array([100, 200, 300, 400, 500])
    predictions = np.array([110, 190, 320, 380, 520])
    
    print("Test Metrics:")
    print(f"MAE: {calculate_mae(actuals, predictions):.4f}")
    print(f"MMRE: {calculate_mmre(actuals, predictions):.4f}")
    print(f"MdMRE: {calculate_mdmre(actuals, predictions):.4f}")
    print(f"PRED(0.25): {calculate_pred(actuals, predictions):.4f}")
    
    print("\nAll Metrics:")
    all_metrics = calculate_all_metrics(actuals, predictions)
    for name, value in all_metrics.items():
        print(f"  {name}: {value:.4f}")