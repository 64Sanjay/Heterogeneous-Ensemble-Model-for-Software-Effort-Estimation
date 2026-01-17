"""
Evaluation script for Software Effort Estimation models
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.ensemble_model import EnsembleModel, create_all_ensembles
from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel
from src.models.ml_models import get_all_ml_models
from src.evaluation.cross_validation import CrossValidator
from src.evaluation.metrics import calculate_all_metrics, compare_models
from src.utils.config import DATASETS, RESULTS_DIR
from src.utils.helpers import save_results_to_excel, print_results_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Software Effort Estimation models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cocomo81",
        choices=list(DATASETS.keys()),
        help="Dataset to use"
    )
    parser.add_argument(
        "--cv_type",
        type=str,
        default="kfold",
        choices=["loocv", "kfold"],
        help="Cross-validation type"
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds for k-fold CV"
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        default=True,
        help="Whether to scale features"
    )
    parser.add_argument(
        "--compare_all",
        action="store_true",
        default=True,
        help="Compare all models"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        default=True,
        help="Save results to Excel"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model (optional)"
    )
    
    return parser.parse_args()


def load_saved_model(model_path):
    """Load a saved model from disk"""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def evaluate_all_models(X, y, cv_type, n_splits):
    """Evaluate all models and return results"""
    
    # Initialize cross-validator
    cv = CrossValidator(cv_type=cv_type, n_splits=n_splits)
    
    # Create all models
    print("\nCreating models...")
    
    # Individual models
    individual_models = {
        "CBR": CBRModel(),
        "COCOMO": COCOMOModel(),
        **get_all_ml_models()
    }
    
    # Ensemble models
    ensemble_models = create_all_ensembles(combination_rule="median")
    
    # Evaluate
    print("\nEvaluating models...")
    ensemble_df, individual_df, ensemble_preds, individual_preds = cv.evaluate_ensemble_vs_individual(
        ensemble_models, individual_models, X, y
    )
    
    return {
        "ensemble_results": ensemble_df,
        "individual_results": individual_df,
        "ensemble_predictions": ensemble_preds,
        "individual_predictions": individual_preds
    }


def main():
    args = parse_args()
    
    print("=" * 70)
    print("SOFTWARE EFFORT ESTIMATION - MODEL EVALUATION")
    print("=" * 70)
    
    # Load data
    print(f"\n[1] Loading dataset: {args.dataset}")
    loader = DataLoader(args.dataset)
    df = loader.load_raw_data()
    
    # Preprocess
    print(f"\n[2] Preprocessing data (scale={args.scale})")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=args.scale)
    
    # Evaluate
    print(f"\n[3] Running {args.cv_type.upper()} evaluation...")
    
    if args.model_path:
        # Evaluate specific saved model
        print(f"Loading model from: {args.model_path}")
        model = load_saved_model(args.model_path)
        
        cv = CrossValidator(cv_type=args.cv_type, n_splits=args.n_splits)
        result = cv.evaluate_model(model, X, y)
        
        print("\nResults:")
        for metric, value in result["metrics"].items():
            print(f"  {metric}: {value:.4f}")
            
    else:
        # Evaluate all models
        results = evaluate_all_models(X, y, args.cv_type, args.n_splits)
        
        # Display results
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print("\n--- Individual Models ---")
        print_results_table(results["individual_results"])
        
        print("\n--- Ensemble Models ---")
        print_results_table(results["ensemble_results"])
        
        # Find best models
        best_individual = results["individual_results"].loc[
            results["individual_results"]['MAE'].idxmin()
        ]
        best_ensemble = results["ensemble_results"].loc[
            results["ensemble_results"]['MAE'].idxmin()
        ]
        
        print("\n" + "-" * 70)
        print(f"Best Individual Model: {best_individual['Model']} (MAE: {best_individual['MAE']:.2f})")
        print(f"Best Ensemble Model: {best_ensemble['Model']} (MAE: {best_ensemble['MAE']:.2f})")
        
        # Calculate improvement
        improvement = ((best_individual['MAE'] - best_ensemble['MAE']) / best_individual['MAE']) * 100
        if improvement > 0:
            print(f"\nEnsemble Improvement: {improvement:.2f}%")
        else:
            print(f"\nNote: Individual model performed better by {-improvement:.2f}%")
        
        # Save results
        if args.save_results:
            print(f"\n[4] Saving results...")
            output_file = RESULTS_DIR / f"{args.dataset}_{args.cv_type}_evaluation.xlsx"
            save_results_to_excel(
                output_file,
                results["individual_results"],
                results["ensemble_results"]
            )
            print(f"    Results saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    
    return results if not args.model_path else result


if __name__ == "__main__":
    results = main()