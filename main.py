"""
Main script for Software Effort Estimation using Heterogeneous Ensemble
"""

import os
import warnings

# ============== GPU CONFIGURATION ==============
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"GPUs Available: {len(gpus)} Physical, {len(logical_gpus)} Logical")
    except RuntimeError as e:
        print(f"GPU Error: {e}")
else:
    print("No GPU found. Using CPU.")

tf.get_logger().setLevel('ERROR')
# ================================================

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.ensemble_model import EnsembleModel, create_all_ensembles
from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel
from src.models.ml_models import get_all_ml_models
from src.evaluation.cross_validation import CrossValidator
from src.evaluation.metrics import calculate_all_metrics
from src.utils.config import DATASETS, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Software Effort Estimation using Heterogeneous Ensemble"
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
        "--combination_rule",
        type=str,
        default="median",
        choices=["median", "linear", "mean"],
        help="Ensemble combination rule"
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        default=True,
        help="Whether to scale features"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        default=True,
        help="Whether to save results to Excel"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("HETEROGENEOUS ENSEMBLE MODEL FOR SOFTWARE EFFORT ESTIMATION")
    print("=" * 70)
    
    # Load and preprocess data
    print(f"\n[1] Loading dataset: {args.dataset}")
    loader = DataLoader(args.dataset)
    df = loader.load_raw_data()
    summary = loader.get_data_summary()
    
    print(f"    Samples: {summary['num_samples']}")
    print(f"    Features: {summary['num_features']}")
    print(f"    Target range: [{summary['target_stats']['min']:.1f}, {summary['target_stats']['max']:.1f}]")
    
    # Preprocess
    print(f"\n[2] Preprocessing data (scale={args.scale})")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=args.scale)
    
    # Setup cross-validation
    print(f"\n[3] Setting up {args.cv_type.upper()} cross-validation")
    cv = CrossValidator(cv_type=args.cv_type, n_splits=args.n_splits)
    
    # Create models
    print(f"\n[4] Creating models (combination rule: {args.combination_rule})")
    
    # Individual models
    individual_models = {
        "CBR": CBRModel(),
        "COCOMO": COCOMOModel(),
        **get_all_ml_models()
    }
    
    # Ensemble models
    ensemble_models = create_all_ensembles(combination_rule=args.combination_rule)
    
    # Evaluate
    print(f"\n[5] Evaluating models...")
    
    ensemble_df, individual_df, ensemble_preds, individual_preds = cv.evaluate_ensemble_vs_individual(
        ensemble_models, individual_models, X, y
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n--- Individual Models ---")
    print(individual_df.to_string(index=False))
    
    print("\n--- Ensemble Models ---")
    print(ensemble_df.to_string(index=False))
    
    # Find best models
    best_individual = individual_df.loc[individual_df['MAE'].idxmin()]
    best_ensemble = ensemble_df.loc[ensemble_df['MAE'].idxmin()]
    
    print("\n" + "-" * 70)
    print(f"Best Individual Model: {best_individual['Model']} (MAE: {best_individual['MAE']:.2f})")
    print(f"Best Ensemble Model: {best_ensemble['Model']} (MAE: {best_ensemble['MAE']:.2f})")
    
    # Calculate improvement
    improvement = ((best_individual['MAE'] - best_ensemble['MAE']) / best_individual['MAE']) * 100
    print(f"\nImprovement: {improvement:.2f}%")
    
    # Save results
    if args.save_results:
        print(f"\n[6] Saving results...")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_file = RESULTS_DIR / f"{args.dataset}_{args.cv_type}_results.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            individual_df.to_excel(writer, sheet_name='Individual_Models', index=False)
            ensemble_df.to_excel(writer, sheet_name='Ensemble_Models', index=False)
        
        print(f"    Results saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
    
    return ensemble_df, individual_df


if __name__ == "__main__":
    ensemble_results, individual_results = main()