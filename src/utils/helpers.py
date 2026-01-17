"""
Helper functions for the project
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pickle
from datetime import datetime


def save_results_to_excel(filepath: Path, 
                          individual_df: pd.DataFrame,
                          ensemble_df: pd.DataFrame,
                          sheet_prefix: str = ""):
    """
    Save results to Excel file with multiple sheets
    
    Args:
        filepath: Output file path
        individual_df: Individual model results
        ensemble_df: Ensemble model results
        sheet_prefix: Prefix for sheet names
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        individual_df.to_excel(
            writer, 
            sheet_name=f'{sheet_prefix}Individual_Models'.strip('_'), 
            index=False
        )
        ensemble_df.to_excel(
            writer, 
            sheet_name=f'{sheet_prefix}Ensemble_Models'.strip('_'), 
            index=False
        )
    
    print(f"Results saved to: {filepath}")


def print_results_table(df: pd.DataFrame, title: str = ""):
    """
    Print a formatted results table
    
    Args:
        df: DataFrame with results
        title: Optional title
    """
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    
    # Format numeric columns
    df_display = df.copy()
    for col in df_display.select_dtypes(include=[np.number]).columns:
        if col in ['MAE', 'RMSE']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")
        elif col in ['MMRE', 'MdMRE', 'Training_Time']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
        elif 'PRED' in col:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
    
    print(df_display.to_string(index=False))


def save_model(model: Any, filepath: Path):
    """Save model to pickle file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {filepath}")


def load_model(filepath: Path) -> Any:
    """Load model from pickle file"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def save_json(data: Dict, filepath: Path):
    """Save dictionary to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> Dict:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_experiment_log(experiment_name: str,
                          dataset: str,
                          model_name: str,
                          metrics: Dict,
                          params: Dict = None) -> Dict:
    """
    Create an experiment log entry
    
    Args:
        experiment_name: Name of the experiment
        dataset: Dataset used
        model_name: Model name
        metrics: Evaluation metrics
        params: Model parameters
        
    Returns:
        Log entry dictionary
    """
    return {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "model_name": model_name,
        "metrics": metrics,
        "parameters": params or {}
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def calculate_improvement(baseline: float, improved: float) -> float:
    """Calculate percentage improvement"""
    if baseline == 0:
        return 0.0
    return ((baseline - improved) / baseline) * 100


def get_best_model(results_df: pd.DataFrame, 
                   metric: str = "MAE",
                   minimize: bool = True) -> pd.Series:
    """
    Get best model from results DataFrame
    
    Args:
        results_df: DataFrame with model results
        metric: Metric to optimize
        minimize: Whether to minimize (True) or maximize (False)
        
    Returns:
        Series with best model info
    """
    if minimize:
        idx = results_df[metric].idxmin()
    else:
        idx = results_df[metric].idxmax()
    
    return results_df.loc[idx]


def summarize_results(individual_df: pd.DataFrame,
                      ensemble_df: pd.DataFrame) -> Dict:
    """
    Create summary of results
    
    Returns:
        Dictionary with summary statistics
    """
    best_individual = get_best_model(individual_df, "MAE", minimize=True)
    best_ensemble = get_best_model(ensemble_df, "MAE", minimize=True)
    
    improvement = calculate_improvement(
        best_individual["MAE"],
        best_ensemble["MAE"]
    )
    
    return {
        "best_individual": {
            "model": best_individual["Model"],
            "MAE": best_individual["MAE"],
            "MMRE": best_individual["MMRE"]
        },
        "best_ensemble": {
            "model": best_ensemble["Model"],
            "MAE": best_ensemble["MAE"],
            "MMRE": best_ensemble["MMRE"]
        },
        "improvement_percentage": improvement,
        "ensemble_better": improvement > 0
    }


if __name__ == "__main__":
    # Test helpers
    print("Testing helper functions...")
    
    # Test format_time
    print(f"format_time(45.5): {format_time(45.5)}")
    print(f"format_time(125.5): {format_time(125.5)}")
    print(f"format_time(3725.5): {format_time(3725.5)}")
    
    # Test calculate_improvement
    print(f"calculate_improvement(100, 80): {calculate_improvement(100, 80):.2f}%")