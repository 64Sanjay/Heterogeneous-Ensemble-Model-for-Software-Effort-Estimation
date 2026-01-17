#!/usr/bin/env python
"""
Interactive Demo for Software Effort Estimation
Test the models with custom project parameters
"""

import os
import sys
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

# Suppress TensorFlow messages
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel
from src.models.ml_models import XGBoostModel, ANNModel, KNNModel, SVRModel
from src.models.ensemble_model import EnsembleModel
from src.evaluation.metrics import calculate_all_metrics


class EffortEstimationDemo:
    """Interactive demo for effort estimation"""
    
    def __init__(self):
        self.models = {}
        self.ensemble = None
        self.preprocessor = None
        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.scaler_mean = None
        self.scaler_std = None
        self.is_trained = False
        
    def load_and_train(self, dataset_name="cocomo81"):
        """Load data and train all models"""
        print(f"\n{'='*60}")
        print(f"Loading and Training Models on {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Load data
        print("\n[1/4] Loading dataset...")
        loader = DataLoader(dataset_name)
        df = loader.load_raw_data()
        self.feature_names = loader.get_feature_names()
        
        print(f"      Samples: {len(df)}, Features: {len(self.feature_names)}")
        
        # Get raw data for reference
        self.raw_df = df.copy()
        
        # Preprocess
        print("[2/4] Preprocessing...")
        self.preprocessor = DataPreprocessor()
        X, y = self.preprocessor.preprocess_pipeline(df, scale=True)
        
        # Store scaler parameters
        self.scaler_mean = self.preprocessor.scaler.mean_
        self.scaler_std = self.preprocessor.scaler.scale_
        
        self.X_train = X
        self.y_train = y
        
        # Train individual models
        print("[3/4] Training individual models...")
        self.models = {
            "CBR": CBRModel(k=5),
            "COCOMO": COCOMOModel(),
            "XGBoost": XGBoostModel(),
            "KNN": KNNModel(),
            "SVR": SVRModel()
        }
        
        for name, model in self.models.items():
            print(f"      Training {name}...")
            model.fit(X, y)
        
        # Train ensemble
        print("[4/4] Training ensemble model...")
        self.ensemble = EnsembleModel(ml_model_name="XGBoost", combination_rule="median")
        self.ensemble.fit(X, y)
        
        self.is_trained = True
        print("\n✓ All models trained successfully!")
        
    def show_feature_ranges(self):
        """Show the valid ranges for each feature"""
        print(f"\n{'='*60}")
        print("FEATURE RANGES (from training data)")
        print(f"{'='*60}")
        print(f"\n{'Feature':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
        print("-" * 60)
        
        for i, name in enumerate(self.feature_names):
            col_data = self.raw_df.iloc[:, i]
            print(f"{name:<15} {col_data.min():>10.2f} {col_data.max():>10.2f} "
                  f"{col_data.mean():>10.2f} {col_data.std():>10.2f}")
        
        print("-" * 60)
        print(f"{'actual':<15} {self.raw_df.iloc[:, -1].min():>10.2f} "
              f"{self.raw_df.iloc[:, -1].max():>10.2f} "
              f"{self.raw_df.iloc[:, -1].mean():>10.2f} "
              f"{self.raw_df.iloc[:, -1].std():>10.2f}")
    
    def predict_from_raw(self, raw_features: dict) -> dict:
        """
        Make prediction from raw (unscaled) feature values
        
        Args:
            raw_features: Dictionary of feature names and values
            
        Returns:
            Dictionary of predictions from each model
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call load_and_train() first.")
        
        # Create feature array in correct order
        feature_array = np.zeros(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            if name in raw_features:
                feature_array[i] = raw_features[name]
            else:
                # Use mean value if not provided
                feature_array[i] = self.raw_df.iloc[:, i].mean()
        
        # Scale features
        scaled_features = (feature_array - self.scaler_mean) / self.scaler_std
        scaled_features = scaled_features.reshape(1, -1)
        
        # Get predictions
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(scaled_features)[0]
            predictions[name] = max(0, pred)
        
        # Ensemble prediction
        ensemble_pred = self.ensemble.predict(scaled_features)[0]
        predictions["Ensemble"] = max(0, ensemble_pred)
        
        return predictions
    
    def interactive_prediction(self):
        """Interactive mode for making predictions"""
        print(f"\n{'='*60}")
        print("INTERACTIVE PREDICTION MODE")
        print(f"{'='*60}")
        print("\nEnter project parameters (press Enter for default values)")
        print("Type 'quit' to exit\n")
        
        while True:
            print("-" * 60)
            features = {}
            
            try:
                # Cost Driver inputs
                print("\n--- COST DRIVERS ---")
                
                # Reliability
                val = input(f"rely (Required Reliability) [0.75-1.40, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['rely'] = float(val) if val else 1.0
                
                # Data
                val = input(f"data (Database Size) [0.94-1.16, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['data'] = float(val) if val else 1.0
                
                # Complexity
                val = input(f"cplx (Complexity) [0.70-1.65, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['cplx'] = float(val) if val else 1.0
                
                # Time
                val = input(f"time (Execution Time Constraint) [1.0-1.66, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['time'] = float(val) if val else 1.0
                
                # Storage
                val = input(f"stor (Storage Constraint) [1.0-1.56, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['stor'] = float(val) if val else 1.0
                
                # Virtual Machine
                val = input(f"virt (VM Volatility) [0.87-1.30, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['virt'] = float(val) if val else 1.0
                
                # Turnaround
                val = input(f"turn (Turnaround Time) [0.87-1.15, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['turn'] = float(val) if val else 1.0
                
                print("\n--- PERSONNEL ATTRIBUTES ---")
                
                # Analyst Capability
                val = input(f"acap (Analyst Capability) [0.71-1.46, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['acap'] = float(val) if val else 1.0
                
                # Applications Experience
                val = input(f"aexp (Application Experience) [0.82-1.29, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['aexp'] = float(val) if val else 1.0
                
                # Programmer Capability
                val = input(f"pcap (Programmer Capability) [0.70-1.42, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['pcap'] = float(val) if val else 1.0
                
                # Virtual Machine Experience
                val = input(f"vexp (VM Experience) [0.90-1.21, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['vexp'] = float(val) if val else 1.0
                
                # Language Experience
                val = input(f"lexp (Language Experience) [0.95-1.14, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['lexp'] = float(val) if val else 1.0
                
                print("\n--- PROJECT ATTRIBUTES ---")
                
                # Modern Practices
                val = input(f"modp (Modern Practices) [0.82-1.24, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['modp'] = float(val) if val else 1.0
                
                # Tools
                val = input(f"tool (Software Tools) [0.83-1.24, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['tool'] = float(val) if val else 1.0
                
                # Schedule
                val = input(f"sced (Schedule Constraint) [1.0-1.23, default=1.0]: ").strip()
                if val.lower() == 'quit':
                    break
                features['sced'] = float(val) if val else 1.0
                
                print("\n--- SIZE ---")
                
                # Lines of Code
                val = input(f"loc (Lines of Code in KLOC) [required]: ").strip()
                if val.lower() == 'quit':
                    break
                if not val:
                    print("ERROR: LOC is required!")
                    continue
                features['loc'] = float(val)
                
                # Make predictions
                print(f"\n{'='*60}")
                print("EFFORT PREDICTIONS")
                print(f"{'='*60}")
                
                predictions = self.predict_from_raw(features)
                
                print(f"\n{'Model':<20} {'Predicted Effort':>20}")
                print("-" * 45)
                for model, pred in predictions.items():
                    if model == "Ensemble":
                        print("-" * 45)
                    print(f"{model:<20} {pred:>15.2f} person-months")
                
                print(f"\n★ Recommended Estimate (Ensemble): {predictions['Ensemble']:.2f} person-months")
                
            except ValueError as e:
                print(f"\nError: Invalid input - {e}")
                continue
            except KeyboardInterrupt:
                break
        
        print("\nExiting interactive mode...")
    
    def quick_test(self, loc: float, complexity: str = "nominal"):
        """
        Quick test with minimal inputs
        
        Args:
            loc: Lines of code (KLOC)
            complexity: 'low', 'nominal', 'high', or 'very_high'
        """
        complexity_map = {
            "low": {"cplx": 0.85, "time": 1.0, "rely": 0.88},
            "nominal": {"cplx": 1.0, "time": 1.0, "rely": 1.0},
            "high": {"cplx": 1.15, "time": 1.11, "rely": 1.15},
            "very_high": {"cplx": 1.30, "time": 1.30, "rely": 1.40}
        }
        
        if complexity not in complexity_map:
            complexity = "nominal"
        
        features = complexity_map[complexity].copy()
        features['loc'] = loc
        
        print(f"\n{'='*60}")
        print(f"QUICK ESTIMATION")
        print(f"{'='*60}")
        print(f"LOC: {loc} KLOC")
        print(f"Complexity: {complexity}")
        
        predictions = self.predict_from_raw(features)
        
        print(f"\n{'Model':<20} {'Predicted Effort':>20}")
        print("-" * 45)
        for model, pred in predictions.items():
            if model == "Ensemble":
                print("-" * 45)
            print(f"{model:<20} {pred:>15.2f} person-months")
        
        return predictions
    
    def batch_predict(self, projects: list) -> pd.DataFrame:
        """
        Predict effort for multiple projects
        
        Args:
            projects: List of dictionaries with project features
            
        Returns:
            DataFrame with predictions
        """
        results = []
        
        for i, project in enumerate(projects):
            predictions = self.predict_from_raw(project)
            row = {"Project": i + 1, **project, **predictions}
            results.append(row)
        
        return pd.DataFrame(results)
    
    def compare_with_actual(self, loc: float, actual_effort: float, **kwargs):
        """Compare predictions with actual effort"""
        features = {'loc': loc, **kwargs}
        predictions = self.predict_from_raw(features)
        
        print(f"\n{'='*60}")
        print("PREDICTION vs ACTUAL COMPARISON")
        print(f"{'='*60}")
        print(f"Actual Effort: {actual_effort:.2f} person-months")
        print(f"\n{'Model':<20} {'Predicted':>12} {'Error':>12} {'Error %':>10}")
        print("-" * 60)
        
        for model, pred in predictions.items():
            error = pred - actual_effort
            error_pct = (abs(error) / actual_effort) * 100 if actual_effort > 0 else 0
            print(f"{model:<20} {pred:>12.2f} {error:>+12.2f} {error_pct:>9.1f}%")
        
        return predictions


def main():
    """Main demo function"""
    print("\n" + "=" * 60)
    print("  SOFTWARE EFFORT ESTIMATION - INTERACTIVE DEMO")
    print("=" * 60)
    
    demo = EffortEstimationDemo()
    
    while True:
        print("\n" + "-" * 60)
        print("MENU")
        print("-" * 60)
        print("1. Load and train models (required first)")
        print("2. Show feature ranges")
        print("3. Interactive prediction (enter all parameters)")
        print("4. Quick test (LOC + complexity level)")
        print("5. Batch prediction (multiple projects)")
        print("6. Compare with actual effort")
        print("7. Run sample predictions")
        print("0. Exit")
        print("-" * 60)
        
        choice = input("\nEnter choice [0-7]: ").strip()
        
        if choice == "0":
            print("\nGoodbye!")
            break
            
        elif choice == "1":
            dataset = input("Dataset (cocomo81/nasa93) [cocomo81]: ").strip() or "cocomo81"
            demo.load_and_train(dataset)
            
        elif choice == "2":
            if not demo.is_trained:
                print("\n⚠ Please load and train models first (option 1)")
            else:
                demo.show_feature_ranges()
                
        elif choice == "3":
            if not demo.is_trained:
                print("\n⚠ Please load and train models first (option 1)")
            else:
                demo.interactive_prediction()
                
        elif choice == "4":
            if not demo.is_trained:
                print("\n⚠ Please load and train models first (option 1)")
            else:
                try:
                    loc = float(input("Enter LOC (KLOC): ").strip())
                    complexity = input("Complexity (low/nominal/high/very_high) [nominal]: ").strip() or "nominal"
                    demo.quick_test(loc, complexity)
                except ValueError:
                    print("Invalid input!")
                    
        elif choice == "5":
            if not demo.is_trained:
                print("\n⚠ Please load and train models first (option 1)")
            else:
                print("\nBatch prediction example:")
                projects = [
                    {"loc": 50, "cplx": 1.0, "rely": 1.0},
                    {"loc": 100, "cplx": 1.15, "rely": 1.15},
                    {"loc": 200, "cplx": 1.30, "rely": 1.40},
                ]
                results = demo.batch_predict(projects)
                print(results.to_string())
                
        elif choice == "6":
            if not demo.is_trained:
                print("\n⚠ Please load and train models first (option 1)")
            else:
                try:
                    loc = float(input("Enter LOC (KLOC): ").strip())
                    actual = float(input("Enter actual effort (person-months): ").strip())
                    demo.compare_with_actual(loc, actual)
                except ValueError:
                    print("Invalid input!")
                    
        elif choice == "7":
            if not demo.is_trained:
                print("\n⚠ Please load and train models first (option 1)")
            else:
                print("\n" + "=" * 60)
                print("SAMPLE PREDICTIONS")
                print("=" * 60)
                
                # Sample 1: Small project
                print("\n--- Sample 1: Small Project (10 KLOC, Low Complexity) ---")
                demo.quick_test(10, "low")
                
                # Sample 2: Medium project
                print("\n--- Sample 2: Medium Project (50 KLOC, Nominal Complexity) ---")
                demo.quick_test(50, "nominal")
                
                # Sample 3: Large project
                print("\n--- Sample 3: Large Project (200 KLOC, High Complexity) ---")
                demo.quick_test(200, "high")
                
                # Sample 4: Complex project
                print("\n--- Sample 4: Very Complex Project (500 KLOC, Very High Complexity) ---")
                demo.quick_test(500, "very_high")
        
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()
