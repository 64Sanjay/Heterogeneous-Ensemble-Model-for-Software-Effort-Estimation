"""
Generate comprehensive experiment report
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path
from datetime import datetime


def generate_report():
    print("="*70)
    print("GENERATING EXPERIMENT REPORT")
    print("="*70)
    
    results_dir = Path("experiments/results")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("# Software Effort Estimation - Experiment Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n---\n")
    
    all_results = []
    
    for result_file in results_dir.glob("*_kfold_results.xlsx"):
        dataset_name = result_file.stem.replace("_kfold_results", "")
        
        try:
            individual_df = pd.read_excel(result_file, sheet_name='Individual_Models')
            ensemble_df = pd.read_excel(result_file, sheet_name='Ensemble_Models')
            
            # Find best models
            best_ind = individual_df.loc[individual_df['MAE'].idxmin()]
            best_ens = ensemble_df.loc[ensemble_df['MAE'].idxmin()]
            
            improvement = ((best_ind['MAE'] - best_ens['MAE']) / best_ind['MAE']) * 100
            
            report_lines.append(f"## Dataset: {dataset_name.upper()}")
            report_lines.append("")
            report_lines.append("### Best Individual Model")
            report_lines.append(f"- **Model**: {best_ind['Model']}")
            report_lines.append(f"- **MAE**: {best_ind['MAE']:.2f}")
            report_lines.append(f"- **MMRE**: {best_ind['MMRE']:.4f}")
            report_lines.append(f"- **PRED(0.25)**: {best_ind['PRED(0.25)']:.4f}")
            report_lines.append("")
            report_lines.append("### Best Ensemble Model")
            report_lines.append(f"- **Model**: {best_ens['Model']}")
            report_lines.append(f"- **MAE**: {best_ens['MAE']:.2f}")
            report_lines.append(f"- **MMRE**: {best_ens['MMRE']:.4f}")
            report_lines.append(f"- **PRED(0.25)**: {best_ens['PRED(0.25)']:.4f}")
            report_lines.append("")
            
            if improvement > 0:
                report_lines.append(f"### Improvement: **{improvement:.2f}%** (Ensemble is better)")
            else:
                report_lines.append(f"### Improvement: **{-improvement:.2f}%** (Individual is better)")
            
            report_lines.append("")
            report_lines.append("### All Individual Models")
            report_lines.append("")
            report_lines.append("| Model | MAE | MMRE | PRED(0.25) |")
            report_lines.append("|-------|-----|------|------------|")
            for _, row in individual_df.iterrows():
                report_lines.append(f"| {row['Model']} | {row['MAE']:.2f} | {row['MMRE']:.4f} | {row['PRED(0.25)']:.4f} |")
            
            report_lines.append("")
            report_lines.append("### All Ensemble Models")
            report_lines.append("")
            report_lines.append("| Model | MAE | MMRE | PRED(0.25) |")
            report_lines.append("|-------|-----|------|------------|")
            for _, row in ensemble_df.iterrows():
                report_lines.append(f"| {row['Model']} | {row['MAE']:.2f} | {row['MMRE']:.4f} | {row['PRED(0.25)']:.4f} |")
            
            report_lines.append("\n---\n")
            
            # Collect for summary
            all_results.append({
                'Dataset': dataset_name,
                'Best_Individual': best_ind['Model'],
                'Best_Individual_MAE': best_ind['MAE'],
                'Best_Ensemble': best_ens['Model'],
                'Best_Ensemble_MAE': best_ens['MAE'],
                'Improvement': improvement
            })
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    # Summary table
    if all_results:
        report_lines.append("## Summary")
        report_lines.append("")
        report_lines.append("| Dataset | Best Individual | MAE | Best Ensemble | MAE | Improvement |")
        report_lines.append("|---------|-----------------|-----|---------------|-----|-------------|")
        for r in all_results:
            imp_str = f"+{r['Improvement']:.2f}%" if r['Improvement'] > 0 else f"{r['Improvement']:.2f}%"
            report_lines.append(f"| {r['Dataset']} | {r['Best_Individual']} | {r['Best_Individual_MAE']:.2f} | {r['Best_Ensemble']} | {r['Best_Ensemble_MAE']:.2f} | {imp_str} |")
    
    # Write report
    report_path = reports_dir / "experiment_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {report_path}")
    
    # Also print to console
    print("\n" + "="*70)
    print("REPORT PREVIEW")
    print("="*70)
    print('\n'.join(report_lines[:50]))
    if len(report_lines) > 50:
        print("\n... (see full report in file)")


if __name__ == "__main__":
    generate_report()
