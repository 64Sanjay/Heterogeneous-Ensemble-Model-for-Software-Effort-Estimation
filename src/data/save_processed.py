"""
Save processed datasets
"""

import sys
sys.path.insert(0, '.')

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.utils.config import DATASETS, PROCESSED_DATA_DIR
import pandas as pd
import numpy as np

def save_all_processed():
    """Process and save all datasets"""
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for dataset_name in DATASETS.keys():
        try:
            print(f"Processing {dataset_name}...")
            
            loader = DataLoader(dataset_name)
            df = loader.load_raw_data()
            
            preprocessor = DataPreprocessor()
            X, y = preprocessor.preprocess_pipeline(df, scale=True)
            
            # Create processed dataframe
            feature_names = loader.get_feature_names()
            processed_df = pd.DataFrame(X, columns=[f"{name}_scaled" for name in feature_names])
            processed_df['actual_effort'] = y
            
            # Save
            output_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
            processed_df.to_csv(output_path, index=False)
            print(f"  Saved: {output_path}")
            
            # Also save stats
            stats = {
                'feature': feature_names + ['actual_effort'],
                'mean': list(df.iloc[:, :-1].mean().values) + [df.iloc[:, -1].mean()],
                'std': list(df.iloc[:, :-1].std().values) + [df.iloc[:, -1].std()],
                'min': list(df.iloc[:, :-1].min().values) + [df.iloc[:, -1].min()],
                'max': list(df.iloc[:, :-1].max().values) + [df.iloc[:, -1].max()]
            }
            stats_df = pd.DataFrame(stats)
            stats_path = PROCESSED_DATA_DIR / f"{dataset_name}_stats.csv"
            stats_df.to_csv(stats_path, index=False)
            print(f"  Saved: {stats_path}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nAll datasets processed!")

if __name__ == "__main__":
    save_all_processed()
