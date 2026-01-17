"""
Run complete experiments on all datasets with all configurations
"""

import os
import sys
import warnings
import pandas as pd
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.ensemble_model import EnsembleModel
from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel
from src.models.ml_models import get_all_ml_models
from src.evaluation.cross_validation import CrossValidator
from src.evaluation.metrics import calculate_all_metrics
from src.utils.config import DATASETS, RESULTS_DIR

def run_experiment(dataset_name, cv_type='kfold', n_splits=5):
    """Run experiment on a single dataset"""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    try:
        loader = DataLoader(dataset_name)
        df = loader.load_raw_data()
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, None
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
    # Cross-validation
    cv = CrossValidator(cv_type=cv_type, n_splits=n_splits)
    
    # Individual models
    individual_models = {
        "CBR": CBRModel(),
        "COCOMO": COCOMOModel(use_nn_correction=False),
        **get_all_ml_models()
    }
    
    # Ensemble models with different ML backends
    ensemble_configs = [
        ("Median", "median"),
        ("Mean", "mean"),
        ("Linear", "linear")
    ]
    
    all_results = []
    
    # Evaluate individual models
    print("\nEvaluating Individual Models...")
    for name, model in individual_models.items():
        print(f"  - {name}")
        result = cv.evaluate_model(model, X, y)
        result['metrics']['Model'] = name
        result['metrics']['Type'] = 'Individual'
        result['metrics']['Dataset'] = dataset_name
        all_results.append(result['metrics'])
    
    # Evaluate ensemble models
    print("\nEvaluating Ensemble Models...")
    for combo_name, combo_rule in ensemble_configs:
        for ml_name in ['XGBoost', 'ANN', 'KNN', 'SVR']:
            model_name = f"Ensemble_{ml_name}_{combo_name}"
            print(f"  - {model_name}")
            
            ensemble = EnsembleModel(
                ml_model_name=ml_name,
                combination_rule=combo_rule
            )
            result = cv.evaluate_model(ensemble, X, y)
            result['metrics']['Model'] = model_name
            result['metrics']['Type'] = 'Ensemble'
            result['metrics']['Dataset'] = dataset_name
            result['metrics']['Combination'] = combo_name
            all_results.append(result['metrics'])
    
    results_df = pd.DataFrame(all_results)
    return results_df

def main():
    print("="*70)
    print("COMPLETE EXPERIMENT SUITE")
    print("Software Effort Estimation - Heterogeneous Ensemble")
    print("="*70)
    
    all_results = []
    
    # Run on all available datasets
    for dataset_name in DATASETS.keys():
        try:
            results = run_experiment(dataset_name)
            if results is not None:
                all_results.append(results)
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")
    
    # Combine all results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        
        # Save to Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"complete_results_{timestamp}.xlsx"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            final_results.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Summary by dataset
            for dataset in final_results['Dataset'].unique():
                df_dataset = final_results[final_results['Dataset'] == dataset]
                df_dataset.to_excel(writer, sheet_name=dataset, index=False)
        
        print(f"\n{'='*70}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        print("\n--- BEST MODELS BY DATASET ---")
        for dataset in final_results['Dataset'].unique():
            df_dataset = final_results[final_results['Dataset'] == dataset]
            best = df_dataset.loc[df_dataset['MAE'].idxmin()]
            print(f"\n{dataset}:")
            print(f"  Best Model: {best['Model']}")
            print(f"  MAE: {best['MAE']:.2f}")
            print(f"  MMRE: {best['MMRE']:.4f}")
        
        return final_results
    
    return None

if __name__ == "__main__":
    results = main()
