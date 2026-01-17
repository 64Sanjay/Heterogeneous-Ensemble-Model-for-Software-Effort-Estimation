"""
Training script for Software Effort Estimation models
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from datetime import datetime

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.ensemble_model import EnsembleModel, create_all_ensembles
from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel
from src.models.ml_models import get_all_ml_models
from src.utils.config import DATASETS, RESULTS_DIR, RANDOM_SEED


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Software Effort Estimation models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cocomo81",
        choices=list(DATASETS.keys()),
        help="Dataset to use for training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["ensemble", "cbr", "cocomo", "xgboost", "ann", "knn", "svr", "all"],
        help="Model to train"
    )
    parser.add_argument(
        "--ml_model",
        type=str,
        default="XGBoost",
        choices=["XGBoost", "ANN", "KNN", "SVR", "LinearRegression"],
        help="ML model for ensemble"
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
        "--save_model",
        action="store_true",
        default=True,
        help="Whether to save trained model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/models",
        help="Directory to save trained models"
    )
    
    return parser.parse_args()


def train_single_model(model, X_train, y_train, model_name):
    """Train a single model and return training info"""
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.2f} seconds")
    
    return model, training_time


def save_model(model, model_name, output_dir):
    """Save trained model to disk"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"{model_name}_{timestamp}.pkl"
    
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    
    print(f"  Model saved to: {filename}")
    return filename


def main():
    args = parse_args()
    
    print("=" * 70)
    print("SOFTWARE EFFORT ESTIMATION - MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    print(f"\n[1] Loading dataset: {args.dataset}")
    loader = DataLoader(args.dataset)
    df = loader.load_raw_data()
    
    # Preprocess
    print(f"\n[2] Preprocessing data (scale={args.scale})")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=args.scale)
    
    print(f"    X shape: {X.shape}")
    print(f"    y shape: {y.shape}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    print(f"\n[3] Training models...")
    trained_models = {}
    training_times = {}
    
    if args.model == "ensemble" or args.model == "all":
        ensemble = EnsembleModel(
            ml_model_name=args.ml_model,
            combination_rule=args.combination_rule
        )
        model, train_time = train_single_model(ensemble, X, y, f"Ensemble_{args.ml_model}")
        trained_models[f"Ensemble_{args.ml_model}"] = model
        training_times[f"Ensemble_{args.ml_model}"] = train_time
        
    if args.model == "cbr" or args.model == "all":
        cbr = CBRModel()
        model, train_time = train_single_model(cbr, X, y, "CBR")
        trained_models["CBR"] = model
        training_times["CBR"] = train_time
        
    if args.model == "cocomo" or args.model == "all":
        cocomo = COCOMOModel()
        model, train_time = train_single_model(cocomo, X, y, "COCOMO")
        trained_models["COCOMO"] = model
        training_times["COCOMO"] = train_time
        
    if args.model in ["xgboost", "ann", "knn", "svr"] or args.model == "all":
        ml_models = get_all_ml_models()
        
        if args.model == "all":
            for name, ml_model in ml_models.items():
                model, train_time = train_single_model(ml_model, X, y, name)
                trained_models[name] = model
                training_times[name] = train_time
        else:
            model_map = {
                "xgboost": "XGBoost",
                "ann": "ANN",
                "knn": "KNN",
                "svr": "SVR"
            }
            name = model_map[args.model]
            model, train_time = train_single_model(ml_models[name], X, y, name)
            trained_models[name] = model
            training_times[name] = train_time
    
    # Save models
    if args.save_model:
        print(f"\n[4] Saving models...")
        for name, model in trained_models.items():
            save_model(model, name, args.output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nDataset: {args.dataset}")
    print(f"Samples: {len(y)}")
    print(f"Features: {X.shape[1]}")
    print(f"\nModels trained:")
    for name, train_time in training_times.items():
        print(f"  - {name}: {train_time:.2f}s")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    
    return trained_models


if __name__ == "__main__":
    trained_models = main()