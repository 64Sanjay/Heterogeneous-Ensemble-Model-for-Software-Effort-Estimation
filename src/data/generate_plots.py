"""
Generate publication-ready visualizations
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_model_comparison(individual_df, ensemble_df, dataset_name):
    """Create model comparison bar charts"""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Combine data
    individual_df = individual_df.copy()
    ensemble_df = ensemble_df.copy()
    individual_df['Type'] = 'Individual'
    ensemble_df['Type'] = 'Ensemble'
    
    all_data = pd.concat([individual_df, ensemble_df], ignore_index=True)
    
    colors = {'Individual': '#3498db', 'Ensemble': '#e74c3c'}
    
    # MAE
    ax = axes[0]
    x = range(len(all_data))
    bars = ax.bar(x, all_data['MAE'], color=[colors[t] for t in all_data['Type']])
    ax.set_xticks(x)
    ax.set_xticklabels(all_data['Model'], rotation=45, ha='right')
    ax.set_ylabel('MAE')
    ax.set_title('Mean Absolute Error (Lower is Better)')
    
    # MMRE
    ax = axes[1]
    bars = ax.bar(x, all_data['MMRE'], color=[colors[t] for t in all_data['Type']])
    ax.set_xticks(x)
    ax.set_xticklabels(all_data['Model'], rotation=45, ha='right')
    ax.set_ylabel('MMRE')
    ax.set_title('Mean Magnitude of Relative Error')
    
    # PRED(0.25)
    ax = axes[2]
    bars = ax.bar(x, all_data['PRED(0.25)'], color=[colors[t] for t in all_data['Type']])
    ax.set_xticks(x)
    ax.set_xticklabels(all_data['Model'], rotation=45, ha='right')
    ax.set_ylabel('PRED(0.25)')
    ax.set_title('Prediction Accuracy (Higher is Better)')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['Individual'], label='Individual'),
                       Patch(facecolor=colors['Ensemble'], label='Ensemble')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{dataset_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dataset_name}_comparison.png")


def plot_training_time(individual_df, ensemble_df, dataset_name):
    """Create training time comparison"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    individual_df = individual_df.copy()
    ensemble_df = ensemble_df.copy()
    individual_df['Type'] = 'Individual'
    ensemble_df['Type'] = 'Ensemble'
    
    all_data = pd.concat([individual_df, ensemble_df], ignore_index=True)
    
    colors = {'Individual': '#3498db', 'Ensemble': '#e74c3c'}
    
    x = range(len(all_data))
    bars = ax.bar(x, all_data['Training_Time'], color=[colors[t] for t in all_data['Type']])
    ax.set_xticks(x)
    ax.set_xticklabels(all_data['Model'], rotation=45, ha='right')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title(f'Training Time Comparison - {dataset_name}')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{dataset_name}_training_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dataset_name}_training_time.png")


def plot_metrics_heatmap(individual_df, ensemble_df, dataset_name):
    """Create metrics heatmap"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    metrics_cols = ['MAE', 'RMSE', 'MMRE', 'MdMRE', 'PRED(0.25)']
    
    # Individual models
    ind_metrics = individual_df.set_index('Model')[metrics_cols]
    sns.heatmap(ind_metrics, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[0])
    axes[0].set_title('Individual Models')
    
    # Ensemble models
    ens_metrics = ensemble_df.set_index('Model')[metrics_cols]
    sns.heatmap(ens_metrics, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[1])
    axes[1].set_title('Ensemble Models')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{dataset_name}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dataset_name}_heatmap.png")


def plot_boxplot_comparison(results_dict):
    """Create boxplot comparing MAE across datasets"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_for_plot = []
    for dataset, (ind_df, ens_df) in results_dict.items():
        for _, row in ind_df.iterrows():
            data_for_plot.append({
                'Dataset': dataset,
                'Model': row['Model'],
                'Type': 'Individual',
                'MAE': row['MAE']
            })
        for _, row in ens_df.iterrows():
            data_for_plot.append({
                'Dataset': dataset,
                'Model': row['Model'],
                'Type': 'Ensemble',
                'MAE': row['MAE']
            })
    
    plot_df = pd.DataFrame(data_for_plot)
    
    sns.boxplot(data=plot_df, x='Dataset', y='MAE', hue='Type', ax=ax)
    ax.set_title('MAE Distribution by Dataset and Model Type')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'all_datasets_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: all_datasets_boxplot.png")


def main():
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    results_dir = Path("experiments/results")
    results_dict = {}
    
    # Load results for each dataset
    for result_file in results_dir.glob("*_kfold_results.xlsx"):
        dataset_name = result_file.stem.replace("_kfold_results", "")
        print(f"\nProcessing: {dataset_name}")
        
        try:
            individual_df = pd.read_excel(result_file, sheet_name='Individual_Models')
            ensemble_df = pd.read_excel(result_file, sheet_name='Ensemble_Models')
            
            results_dict[dataset_name] = (individual_df, ensemble_df)
            
            # Generate plots
            plot_model_comparison(individual_df, ensemble_df, dataset_name)
            plot_training_time(individual_df, ensemble_df, dataset_name)
            plot_metrics_heatmap(individual_df, ensemble_df, dataset_name)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    # Generate combined plot
    if len(results_dict) > 1:
        plot_boxplot_comparison(results_dict)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
