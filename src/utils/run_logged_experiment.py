"""
Run experiment with logging
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import argparse
from datetime import datetime

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.ensemble_model import EnsembleModel, create_all_ensembles
from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel
from src.models.ml_models import get_all_ml_models
from src.evaluation.cross_validation import CrossValidator
from src.utils.logger import setup_logger, log_experiment_start, log_experiment_end, log_model_result
from src.utils.config import DATASETS, RESULTS_DIR

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cocomo81", choices=list(DATASETS.keys()))
    parser.add_argument("--cv_type", default="kfold", choices=["loocv", "kfold"])
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()
    
    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger("experiment", f"{args.dataset}_{args.cv_type}_{timestamp}.log")
    
    # Log start
    config = {
        "dataset": args.dataset,
        "cv_type": args.cv_type,
        "n_splits": args.n_splits,
        "timestamp": timestamp
    }
    log_experiment_start(logger, config)
    
    # Load data
    logger.info(f"Loading dataset: {args.dataset}")
    loader = DataLoader(args.dataset)
    df = loader.load_raw_data()
    logger.info(f"  Samples: {len(df)}, Features: {len(df.columns) - 1}")
    
    # Preprocess
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
    # Cross-validation
    logger.info(f"Setting up {args.cv_type.upper()} cross-validation")
    cv = CrossValidator(cv_type=args.cv_type, n_splits=args.n_splits)
    
    # Individual models
    logger.info("Evaluating individual models...")
    individual_models = {
        "CBR": CBRModel(),
        "COCOMO": COCOMOModel(),
        **get_all_ml_models()
    }
    
    individual_results = []
    for name, model in individual_models.items():
        logger.info(f"  Evaluating {name}...")
        result = cv.evaluate_model(model, X, y)
        log_model_result(logger, name, result['metrics'])
        result['metrics']['Model'] = name
        individual_results.append(result['metrics'])
    
    # Ensemble models
    logger.info("Evaluating ensemble models...")
    ensemble_models = create_all_ensembles(combination_rule="median")
    
    ensemble_results = []
    for name, model in ensemble_models.items():
        logger.info(f"  Evaluating {name}...")
        result = cv.evaluate_model(model, X, y)
        log_model_result(logger, name, result['metrics'])
        result['metrics']['Model'] = name
        ensemble_results.append(result['metrics'])
    
    # Create DataFrames
    individual_df = pd.DataFrame(individual_results)
    ensemble_df = pd.DataFrame(ensemble_results)
    
    # Find best
    best_ind = individual_df.loc[individual_df['MAE'].idxmin()]
    best_ens = ensemble_df.loc[ensemble_df['MAE'].idxmin()]
    
    improvement = ((best_ind['MAE'] - best_ens['MAE']) / best_ind['MAE']) * 100
    
    # Log end
    results_summary = {
        "best_individual_model": best_ind['Model'],
        "best_individual_mae": best_ind['MAE'],
        "best_ensemble_model": best_ens['Model'],
        "best_ensemble_mae": best_ens['MAE'],
        "improvement_percent": improvement
    }
    log_experiment_end(logger, results_summary)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{args.dataset}_{args.cv_type}_results.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        individual_df.to_excel(writer, sheet_name='Individual_Models', index=False)
        ensemble_df.to_excel(writer, sheet_name='Ensemble_Models', index=False)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Best Individual: {best_ind['Model']} (MAE: {best_ind['MAE']:.2f})")
    print(f"Best Ensemble: {best_ens['Model']} (MAE: {best_ens['MAE']:.2f})")
    print(f"Improvement: {improvement:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
