# """
# Demo script for Software Effort Estimation
# Shows complete workflow from data loading to prediction
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path
# import sys
# import warnings
# warnings.filterwarnings('ignore')

# # Add project root to path
# sys.path.insert(0, str(Path(__file__).parent))

# from src.data.data_loader import DataLoader
# from src.data.preprocessor import DataPreprocessor
# from src.models.cbr_model import CBRModel
# from src.models.cocomo_model import COCOMOModel, PureCOCOMO
# from src.models.ml_models import XGBoostModel, ANNModel, KNNModel, SVRModel
# from src.models.ensemble_model import EnsembleModel
# from src.evaluation.metrics import calculate_all_metrics, calculate_mae, calculate_mmre
# from sklearn.model_selection import train_test_split


# def print_section(title):
#     """Print a section header"""
#     print("\n" + "=" * 70)
#     print(f" {title}")
#     print("=" * 70)


# def print_subsection(title):
#     """Print a subsection header"""
#     print(f"\n--- {title} ---")


# def demo_data_loading():
#     """Demonstrate data loading"""
#     print_section("1. DATA LOADING")
    
#     print("\nLoading COCOMO81 dataset...")
#     loader = DataLoader("cocomo81")
#     df = loader.load_raw_data()
    
#     print(f"\n✓ Dataset loaded successfully!")
#     print(f"\nDataset Overview:")
#     print(f"  • Samples: {len(df)}")
#     print(f"  • Features: {len(df.columns) - 1}")
#     print(f"  • Target: {df.columns[-1]}")
    
#     print(f"\nFeature Names:")
#     features = loader.get_feature_names()
#     for i, feat in enumerate(features):
#         print(f"  {i+1:2d}. {feat}")
    
#     print(f"\nFirst 5 rows:")
#     print(df.head().to_string())
    
#     print(f"\nTarget Variable Statistics:")
#     summary = loader.get_data_summary()
#     stats = summary['target_stats']
#     print(f"  • Min: {stats['min']:.1f}")
#     print(f"  • Max: {stats['max']:.1f}")
#     print(f"  • Mean: {stats['mean']:.1f}")
#     print(f"  • Median: {stats['median']:.1f}")
#     print(f"  • Std: {stats['std']:.1f}")
    
#     return loader, df


# def demo_preprocessing(df):
#     """Demonstrate preprocessing"""
#     print_section("2. PREPROCESSING")
    
#     preprocessor = DataPreprocessor()
    
#     print("\nApplying preprocessing pipeline...")
#     print("  • Handling missing values (mean imputation)")
#     print("  • Scaling features (StandardScaler)")
    
#     X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
#     print(f"\n✓ Preprocessing completed!")
#     print(f"\nProcessed Data:")
#     print(f"  • X shape: {X.shape}")
#     print(f"  • y shape: {y.shape}")
    
#     print(f"\nScaling Verification:")
#     print(f"  • X mean: {X.mean():.6f} (should be ~0)")
#     print(f"  • X std: {X.std():.6f} (should be ~1)")
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
    
#     print(f"\nTrain/Test Split:")
#     print(f"  • Training samples: {len(X_train)}")
#     print(f"  • Testing samples: {len(X_test)}")
    
#     return X_train, X_test, y_train, y_test, preprocessor


# def demo_individual_models(X_train, X_test, y_train, y_test):
#     """Demonstrate individual models"""
#     print_section("3. INDIVIDUAL MODELS")
    
#     models = {
#         "CBR": CBRModel(k=5),
#         "COCOMO": COCOMOModel(use_nn_correction=False),
#         "XGBoost": XGBoostModel(),
#         "KNN": KNNModel(),
#         "SVR": SVRModel()
#     }
    
#     results = []
#     predictions_dict = {}
    
#     for name, model in models.items():
#         print_subsection(f"Training {name}")
        
#         # Train
#         model.fit(X_train, y_train)
#         print(f"  ✓ Training completed in {model.training_time:.4f}s")
        
#         # Predict
#         predictions = model.predict(X_test)
#         predictions_dict[name] = predictions
        
#         # Evaluate
#         metrics = calculate_all_metrics(y_test, predictions)
#         results.append({
#             "Model": name,
#             "MAE": metrics["MAE"],
#             "MMRE": metrics["MMRE"],
#             "PRED(0.25)": metrics["PRED(0.25)"],
#             "Training Time": model.training_time
#         })
        
#         print(f"  • MAE: {metrics['MAE']:.2f}")
#         print(f"  • MMRE: {metrics['MMRE']:.4f}")
#         print(f"  • PRED(0.25): {metrics['PRED(0.25)']:.4f}")
    
#     # Summary table
#     print_subsection("Individual Models Summary")
#     results_df = pd.DataFrame(results)
#     print(results_df.to_string(index=False))
    
#     # Best model
#     best_idx = results_df["MAE"].idxmin()
#     best_model = results_df.loc[best_idx, "Model"]
#     best_mae = results_df.loc[best_idx, "MAE"]
#     print(f"\n★ Best Individual Model: {best_model} (MAE: {best_mae:.2f})")
    
#     return results_df, predictions_dict


# def demo_ensemble_models(X_train, X_test, y_train, y_test):
#     """Demonstrate ensemble models"""
#     print_section("4. ENSEMBLE MODELS")
    
#     print("\nEnsemble combines: CBR + COCOMO + ML Model")
#     print("Combination Rule: Median")
    
#     ml_models = ["XGBoost", "KNN", "SVR", "ANN"]
#     results = []
    
#     for ml_name in ml_models:
#         print_subsection(f"Ensemble with {ml_name}")
        
#         # Create ensemble
#         ensemble = EnsembleModel(
#             ml_model_name=ml_name,
#             combination_rule="median"
#         )
        
#         # Train
#         ensemble.fit(X_train, y_train)
#         print(f"  ✓ Training completed in {ensemble.training_time:.4f}s")
        
#         # Component times
#         times = ensemble.get_component_training_times()
#         print(f"    - CBR: {times['CBR']:.4f}s")
#         print(f"    - COCOMO: {times['COCOMO']:.4f}s")
#         print(f"    - {ml_name}: {times[ml_name]:.4f}s")
        
#         # Predict
#         predictions = ensemble.predict(X_test)
        
#         # Evaluate
#         metrics = calculate_all_metrics(y_test, predictions)
#         results.append({
#             "Ensemble": f"CBR+COCOMO+{ml_name}",
#             "MAE": metrics["MAE"],
#             "MMRE": metrics["MMRE"],
#             "PRED(0.25)": metrics["PRED(0.25)"],
#             "Training Time": ensemble.training_time
#         })
        
#         print(f"  • MAE: {metrics['MAE']:.2f}")
#         print(f"  • MMRE: {metrics['MMRE']:.4f}")
#         print(f"  • PRED(0.25): {metrics['PRED(0.25)']:.4f}")
    
#     # Summary table
#     print_subsection("Ensemble Models Summary")
#     results_df = pd.DataFrame(results)
#     print(results_df.to_string(index=False))
    
#     # Best ensemble
#     best_idx = results_df["MAE"].idxmin()
#     best_ensemble = results_df.loc[best_idx, "Ensemble"]
#     best_mae = results_df.loc[best_idx, "MAE"]
#     print(f"\n★ Best Ensemble Model: {best_ensemble} (MAE: {best_mae:.2f})")
    
#     return results_df


# def demo_comparison(individual_results, ensemble_results):
#     """Compare individual vs ensemble models"""
#     print_section("5. COMPARISON: INDIVIDUAL vs ENSEMBLE")
    
#     best_individual = individual_results.loc[individual_results["MAE"].idxmin()]
#     best_ensemble = ensemble_results.loc[ensemble_results["MAE"].idxmin()]
    
#     print(f"\nBest Individual Model:")
#     print(f"  • Model: {best_individual['Model']}")
#     print(f"  • MAE: {best_individual['MAE']:.2f}")
#     print(f"  • MMRE: {best_individual['MMRE']:.4f}")
    
#     print(f"\nBest Ensemble Model:")
#     print(f"  • Model: {best_ensemble['Ensemble']}")
#     print(f"  • MAE: {best_ensemble['MAE']:.2f}")
#     print(f"  • MMRE: {best_ensemble['MMRE']:.4f}")
    
#     # Calculate improvement
#     improvement_mae = ((best_individual['MAE'] - best_ensemble['MAE']) / best_individual['MAE']) * 100
#     improvement_mmre = ((best_individual['MMRE'] - best_ensemble['MMRE']) / best_individual['MMRE']) * 100
    
#     print(f"\nImprovement (Ensemble vs Individual):")
#     if improvement_mae > 0:
#         print(f"  • MAE: ↓ {improvement_mae:.2f}% (Ensemble is better)")
#     else:
#         print(f"  • MAE: ↑ {-improvement_mae:.2f}% (Individual is better)")
    
#     if improvement_mmre > 0:
#         print(f"  • MMRE: ↓ {improvement_mmre:.2f}% (Ensemble is better)")
#     else:
#         print(f"  • MMRE: ↑ {-improvement_mmre:.2f}% (Individual is better)")


# def demo_single_prediction(X_train, y_train, preprocessor):
#     """Demonstrate prediction for a single project"""
#     print_section("6. SINGLE PROJECT PREDICTION")
    
#     print("\nScenario: Predict effort for a new software project")
    
#     # Create a sample project (using mean values)
#     sample_project = X_train.mean(axis=0).reshape(1, -1)
    
#     print("\nProject Features (normalized):")
#     feature_names = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn',
#                      'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced', 'loc']
#     for i, (name, value) in enumerate(zip(feature_names, sample_project[0])):
#         print(f"  {name}: {value:.4f}")
    
#     # Create and train ensemble
#     print("\nTraining ensemble model...")
#     ensemble = EnsembleModel(ml_model_name="XGBoost", combination_rule="median")
#     ensemble.fit(X_train, y_train)
    
#     # Get predictions from each model
#     individual_preds = ensemble.predict_individual(sample_project)
#     ensemble_pred = ensemble.predict(sample_project)
    
#     print("\nPredictions:")
#     print(f"  • CBR Prediction: {individual_preds['CBR'][0]:.2f} person-months")
#     print(f"  • COCOMO Prediction: {individual_preds['COCOMO'][0]:.2f} person-months")
#     print(f"  • XGBoost Prediction: {individual_preds['XGBoost'][0]:.2f} person-months")
#     print(f"\n  ★ Ensemble Prediction (Median): {ensemble_pred[0]:.2f} person-months")


# def demo_cross_validation(X, y):
#     """Demonstrate cross-validation evaluation"""
#     print_section("7. CROSS-VALIDATION EVALUATION")
    
#     from src.evaluation.cross_validation import CrossValidator
    
#     print("\nRunning 5-Fold Cross-Validation...")
    
#     cv = CrossValidator(cv_type="kfold", n_splits=5)
    
#     # Evaluate ensemble
#     ensemble = EnsembleModel(ml_model_name="XGBoost", combination_rule="median")
#     result = cv.evaluate_model(ensemble, X, y)
    
#     print(f"\n✓ Cross-Validation Results (5-Fold):")
#     print(f"  • MAE: {result['metrics']['MAE']:.2f}")
#     print(f"  • RMSE: {result['metrics']['RMSE']:.2f}")
#     print(f"  • MMRE: {result['metrics']['MMRE']:.4f}")
#     print(f"  • MdMRE: {result['metrics']['MdMRE']:.4f}")
#     print(f"  • PRED(0.25): {result['metrics']['PRED(0.25)']:.4f}")
#     print(f"  • Average Training Time: {result['metrics']['Training_Time']:.4f}s")


# def main():
#     """Run complete demo"""
#     print("\n" + "=" * 70)
#     print(" HETEROGENEOUS ENSEMBLE MODEL FOR SOFTWARE EFFORT ESTIMATION")
#     print(" Demo Script")
#     print("=" * 70)
    
#     # 1. Data Loading
#     loader, df = demo_data_loading()
    
#     # 2. Preprocessing
#     X_train, X_test, y_train, y_test, preprocessor = demo_preprocessing(df)
    
#     # 3. Individual Models
#     individual_results, predictions_dict = demo_individual_models(
#         X_train, X_test, y_train, y_test
#     )
    
#     # 4. Ensemble Models
#     ensemble_results = demo_ensemble_models(X_train, X_test, y_train, y_test)
    
#     # 5. Comparison
#     demo_comparison(individual_results, ensemble_results)
    
#     # 6. Single Prediction
#     demo_single_prediction(X_train, y_train, preprocessor)
    
#     # 7. Cross-Validation
#     X, y = loader.get_features_and_target()
#     preprocessor2 = DataPreprocessor()
#     X_scaled, y = preprocessor2.preprocess_pipeline(df, scale=True)
#     demo_cross_validation(X_scaled, y)
    
#     # Final Summary
#     print_section("DEMO COMPLETED")
#     print("\nThis demo demonstrated:")
#     print("  1. ✓ Loading software effort estimation datasets")
#     print("  2. ✓ Preprocessing and feature scaling")
#     print("  3. ✓ Training individual models (CBR, COCOMO, XGBoost, KNN, SVR)")
#     print("  4. ✓ Training heterogeneous ensemble models")
#     print("  5. ✓ Comparing individual vs ensemble performance")
#     print("  6. ✓ Making predictions for new projects")
#     print("  7. ✓ Cross-validation evaluation")
    
#     print("\nTo run the full experiment:")
#     print("  python main.py --dataset cocomo81 --cv_type kfold --n_splits 5")
    
#     print("\n" + "=" * 70)


# if __name__ == "__main__":
#     main()
# -------------v2---------
"""
Demo script for Software Effort Estimation
Shows complete workflow from data loading to prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel
from src.models.ml_models import XGBoostModel, ANNModel, KNNModel, SVRModel
from src.models.ensemble_model import EnsembleModel
from src.evaluation.metrics import calculate_all_metrics
from sklearn.model_selection import train_test_split


def print_header(title):
    """Print a header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title):
    """Print a section header"""
    print(f"\n--- {title} ---")


def demo_data_loading():
    """Demonstrate data loading"""
    print_header("1. DATA LOADING")
    
    print("\nLoading COCOMO81 dataset...")
    loader = DataLoader("cocomo81")
    df = loader.load_raw_data()
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"\nDataset Overview:")
    print(f"  • Samples: {len(df)}")
    print(f"  • Features: {len(df.columns) - 1}")
    print(f"  • Target: {df.columns[-1]}")
    
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    
    summary = loader.get_data_summary()
    print(f"\nTarget Variable Statistics:")
    print(f"  • Min: {summary['target_stats']['min']:.1f}")
    print(f"  • Max: {summary['target_stats']['max']:.1f}")
    print(f"  • Mean: {summary['target_stats']['mean']:.1f}")
    print(f"  • Median: {summary['target_stats']['median']:.1f}")
    
    return loader, df


def demo_preprocessing(df):
    """Demonstrate preprocessing"""
    print_header("2. PREPROCESSING")
    
    preprocessor = DataPreprocessor()
    
    print("\nApplying preprocessing pipeline...")
    print("  • Handling missing values")
    print("  • Scaling features (StandardScaler)")
    
    X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
    print(f"\n✓ Preprocessing completed!")
    print(f"\nProcessed Data:")
    print(f"  • X shape: {X.shape}")
    print(f"  • y shape: {y.shape}")
    print(f"  • X mean: {X.mean():.6f} (should be ~0)")
    print(f"  • X std: {X.std():.6f} (should be ~1)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, preprocessor


def demo_individual_models(X_train, X_test, y_train, y_test):
    """Demonstrate individual models"""
    print_header("3. INDIVIDUAL MODELS")
    
    models = {
        "CBR": CBRModel(k=5),
        "COCOMO": COCOMOModel(use_nn_correction=False),
        "XGBoost": XGBoostModel(),
        "KNN": KNNModel(),
        "SVR": SVRModel()
    }
    
    results = []
    predictions_dict = {}
    
    for name, model in models.items():
        print_section(f"Training {name}")
        
        model.fit(X_train, y_train)
        print(f"  ✓ Training completed in {model.training_time:.4f}s")
        
        predictions = model.predict(X_test)
        predictions_dict[name] = predictions
        
        metrics = calculate_all_metrics(y_test, predictions)
        results.append({
            "Model": name,
            "MAE": metrics["MAE"],
            "MMRE": metrics["MMRE"],
            "PRED(0.25)": metrics["PRED(0.25)"],
            "Time(s)": model.training_time
        })
        
        print(f"  • MAE: {metrics['MAE']:.2f}")
        print(f"  • MMRE: {metrics['MMRE']:.4f}")
        print(f"  • PRED(0.25): {metrics['PRED(0.25)']:.4f}")
    
    print_section("Individual Models Summary")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    best_idx = results_df["MAE"].idxmin()
    best_model = results_df.loc[best_idx, "Model"]
    best_mae = results_df.loc[best_idx, "MAE"]
    print(f"\n★ Best Individual Model: {best_model} (MAE: {best_mae:.2f})")
    
    return results_df, predictions_dict


def demo_ensemble_models(X_train, X_test, y_train, y_test):
    """Demonstrate ensemble models"""
    print_header("4. ENSEMBLE MODELS")
    
    print("\nEnsemble Architecture: CBR + COCOMO + ML Model")
    print("Combination Rule: Median")
    
    ml_models = ["XGBoost", "KNN", "SVR"]
    results = []
    
    for ml_name in ml_models:
        print_section(f"Ensemble with {ml_name}")
        
        ensemble = EnsembleModel(
            ml_model_name=ml_name,
            combination_rule="median"
        )
        
        ensemble.fit(X_train, y_train)
        print(f"  ✓ Training completed in {ensemble.training_time:.4f}s")
        
        times = ensemble.get_component_training_times()
        print(f"    - CBR: {times['CBR']:.4f}s")
        print(f"    - COCOMO: {times['COCOMO']:.4f}s")
        print(f"    - {ml_name}: {times[ml_name]:.4f}s")
        
        predictions = ensemble.predict(X_test)
        
        metrics = calculate_all_metrics(y_test, predictions)
        results.append({
            "Ensemble": f"CBR+COCOMO+{ml_name}",
            "MAE": metrics["MAE"],
            "MMRE": metrics["MMRE"],
            "PRED(0.25)": metrics["PRED(0.25)"],
            "Time(s)": ensemble.training_time
        })
        
        print(f"  • MAE: {metrics['MAE']:.2f}")
        print(f"  • MMRE: {metrics['MMRE']:.4f}")
        print(f"  • PRED(0.25): {metrics['PRED(0.25)']:.4f}")
    
    print_section("Ensemble Models Summary")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    best_idx = results_df["MAE"].idxmin()
    best_ensemble = results_df.loc[best_idx, "Ensemble"]
    best_mae = results_df.loc[best_idx, "MAE"]
    print(f"\n★ Best Ensemble Model: {best_ensemble} (MAE: {best_mae:.2f})")
    
    return results_df


def demo_comparison(individual_results, ensemble_results):
    """Compare individual vs ensemble models"""
    print_header("5. COMPARISON: INDIVIDUAL vs ENSEMBLE")
    
    best_individual = individual_results.loc[individual_results["MAE"].idxmin()]
    best_ensemble = ensemble_results.loc[ensemble_results["MAE"].idxmin()]
    
    print(f"\nBest Individual Model:")
    print(f"  • Model: {best_individual['Model']}")
    print(f"  • MAE: {best_individual['MAE']:.2f}")
    print(f"  • MMRE: {best_individual['MMRE']:.4f}")
    
    print(f"\nBest Ensemble Model:")
    print(f"  • Model: {best_ensemble['Ensemble']}")
    print(f"  • MAE: {best_ensemble['MAE']:.2f}")
    print(f"  • MMRE: {best_ensemble['MMRE']:.4f}")
    
    improvement_mae = ((best_individual['MAE'] - best_ensemble['MAE']) / best_individual['MAE']) * 100
    
    print(f"\nImprovement Analysis:")
    if improvement_mae > 0:
        print(f"  ✓ Ensemble is {improvement_mae:.2f}% better in MAE")
    else:
        print(f"  ✗ Individual is {-improvement_mae:.2f}% better in MAE")


def demo_single_prediction(X_train, y_train):
    """Demonstrate prediction for a single project"""
    print_header("6. SINGLE PROJECT PREDICTION")
    
    print("\nScenario: Predict effort for a new software project")
    
    sample_project = X_train.mean(axis=0).reshape(1, -1)
    
    print("\nTraining ensemble model...")
    ensemble = EnsembleModel(ml_model_name="XGBoost", combination_rule="median")
    ensemble.fit(X_train, y_train)
    
    individual_preds = ensemble.predict_individual(sample_project)
    ensemble_pred = ensemble.predict(sample_project)
    
    print("\nPredictions for new project:")
    print(f"  • CBR Prediction: {individual_preds['CBR'][0]:.2f} person-months")
    print(f"  • COCOMO Prediction: {individual_preds['COCOMO'][0]:.2f} person-months")
    print(f"  • XGBoost Prediction: {individual_preds['XGBoost'][0]:.2f} person-months")
    print(f"\n  ★ Ensemble Prediction (Median): {ensemble_pred[0]:.2f} person-months")


def main():
    """Run complete demo"""
    print("\n" + "=" * 70)
    print(" HETEROGENEOUS ENSEMBLE MODEL FOR SOFTWARE EFFORT ESTIMATION")
    print(" Demo Script")
    print("=" * 70)
    
    try:
        # 1. Data Loading
        loader, df = demo_data_loading()
        
        # 2. Preprocessing
        X_train, X_test, y_train, y_test, preprocessor = demo_preprocessing(df)
        
        # 3. Individual Models
        individual_results, predictions_dict = demo_individual_models(
            X_train, X_test, y_train, y_test
        )
        
        # 4. Ensemble Models
        ensemble_results = demo_ensemble_models(X_train, X_test, y_train, y_test)
        
        # 5. Comparison
        demo_comparison(individual_results, ensemble_results)
        
        # 6. Single Prediction
        demo_single_prediction(X_train, y_train)
        
        # Final Summary
        print_header("DEMO COMPLETED SUCCESSFULLY!")
        print("\nThis demo demonstrated:")
        print("  1. ✓ Loading software effort estimation datasets")
        print("  2. ✓ Preprocessing and feature scaling")
        print("  3. ✓ Training individual models (CBR, COCOMO, XGBoost, KNN, SVR)")
        print("  4. ✓ Training heterogeneous ensemble models")
        print("  5. ✓ Comparing individual vs ensemble performance")
        print("  6. ✓ Making predictions for new projects")
        
        print("\nTo run the full experiment with cross-validation:")
        print("  python main.py --dataset cocomo81 --cv_type kfold --n_splits 5")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())