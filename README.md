# Heterogeneous Ensemble Model for Software Effort Estimation

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible implementation of a heterogeneous ensemble approach for software effort estimation that combines algorithmic, memory-based, and machine learning models (COCOMO-II, Case-Based Reasoning, XGBoost, ANN, KNN, SVR). The goal is robust, accurate effort prediction across classical datasets (COCOMO81, NASA93, NASA60).

Table of Contents
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Install](#install)
- [Usage](#usage)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
  - [Python API (example)](#python-api-example)
- [Models & Ensemble](#models--ensemble)
- [Evaluation Metrics](#evaluation-metrics)
- [Datasets](#datasets)
- [Results (example)](#results-example)
- [Reproducibility & Experiments](#reproducibility--experiments)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Key Features
- Multiple base models:
  - COCOMO-II (algorithmic)
  - Case-Based Reasoning (CBR) (memory-based)
  - Machine learning models: XGBoost, ANN, KNN, SVR
- Heterogeneous ensemble combining different model types
- Multiple combination rules: median, linear (weighted), mean
- Supports common datasets: COCOMO81, NASA93, NASA60
- Evaluation: MAE, MMRE, MdMRE, PRED(x)
- Cross-validation: Leave-One-Out (LOOCV) and K-Fold
- Scripts and utilities for data loading, preprocessing, evaluation and experiments

## Project Structure
Root layout (high-level)
```
software-effort-estimation/
├── data/                  # Dataset files (CSV, ARFF, etc.)
├── src/
│   ├── data/              # Data loading & preprocessing modules
│   ├── models/            # Model implementations (cbr, cocomo, ml, ensemble)
│   ├── evaluation/        # Metrics & cross-validation utilities
│   └── utils/             # Config, logging, helpers
├── notebooks/             # Analysis and experiments (Jupyter)
├── experiments/           # Results, logs, saved models
├── main.py                # CLI entry point for experiments
├── requirements.txt       # Python dependencies
└── README.md
```

## Getting Started

### Requirements
- Python 3.8 or later
- Recommended: virtual environment
- GPU not required but helpful for large ANN experiments

### Install
Clone the repository and install dependencies:
```bash
git clone https://github.com/64Sanjay/Heterogeneous-Ensemble-Model-for-Software-Effort-Estimation.git
cd Heterogeneous-Ensemble-Model-for-Software-Effort-Estimation

# create and activate venv (Linux / macOS)
python -m venv venv
source venv/bin/activate

# (Windows PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

## Usage

### Command Line Interface (CLI)
Run a basic experiment (LOOCV on COCOMO81):
```bash
python main.py --dataset cocomo81 --cv_type loocv
```

Run a K-Fold experiment with specific options:
```bash
python main.py \
  --dataset nasa93 \
  --cv_type kfold \
  --n_splits 5 \
  --ml_model xgboost \
  --combination_rule median \
  --save_results \
  --random_seed 42
```

CLI flags (typical)
- `--dataset`: dataset name (cocomo81, nasa93, nasa60)
- `--cv_type`: `loocv` or `kfold`
- `--n_splits`: number of folds when `kfold` selected
- `--ml_model`: machine learning base model (xgboost, ann, knn, svr)
- `--combination_rule`: `median`, `mean`, `linear`
- `--save_results`: save experiment outputs to `experiments/`
- `--random_seed`: seed for reproducibility

Run `python main.py --help` for full CLI options.

### Python API (example)
A minimal example showing the loader and ensemble API:
```python
from src.data.data_loader import DataLoader
from src.models.ensemble_model import EnsembleModel
from sklearn.model_selection import train_test_split

# Load dataset
loader = DataLoader("cocomo81")
X, y = loader.get_features_and_target()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create ensemble
ensemble = EnsembleModel(ml_model_name="xgboost", combination_rule="median")

# Train and predict
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# Evaluate
from src.evaluation.metrics import evaluate_regression
metrics = evaluate_regression(y_test, y_pred)
print(metrics)
```

## Models & Ensemble
- COCOMO-II: algorithmic effort estimation based on scale and cost drivers; implemented to provide a baseline algorithmic prediction.
- Case-Based Reasoning (CBR): retrieves similar historical projects and adapts efforts; useful when similar cases exist.
- Machine Learning Models:
  - XGBoost (gradient boosting)
  - ANN (feed-forward neural network)
  - KNN (instance-based)
  - SVR (support vector regression)
- Ensemble strategies:
  - Mean: average of base model predictions
  - Median: median of base model predictions (robust to outliers)
  - Linear: weighted linear combination; weights can be learned or configured

## Evaluation Metrics
- MAE — Mean Absolute Error
- MMRE — Mean Magnitude of Relative Error
- MdMRE — Median Magnitude of Relative Error
- PRED(x) — Percentage of predictions within x relative error (e.g., PRED(0.25))

These metrics are implemented in `src/evaluation/metrics.py` and used in cross-validation experiments.

## Datasets
The repository includes or expects commonly used datasets for software effort estimation:
- COCOMO81
- NASA93
- NASA60

Each dataset should be placed (or linked) in `data/` and follow the expected schema for the provided data loader. See `src/data/` for specifics and example parsing code.

## Results (example)
Below is an example summary that can be produced by the evaluation pipeline (example values for illustration):

| Model                  | MAE     | MMRE | PRED(0.25) |
|------------------------|---------:|------:|-----------:|
| CBR                    | 648.93  | 1.81  | 0.11       |
| COCOMO-II              | 219.68  | 0.38  | 0.43       |
| XGBoost                | 319.20  | 1.38  | 0.21       |
| Ensemble (XGBoost)     | 284.41  | 0.81  | 0.25       |

Use `--save_results` to persist detailed per-fold results and summary statistics under `experiments/`.

## Reproducibility & Experiments
- Use the `--random_seed` option on CLI for deterministic splits.
- Save configurations and metrics for each experiment run in `experiments/` for later analysis.
- Notebooks under `notebooks/` provide reproducible analysis and plotting scripts for comparing models and rules.

Suggested experiment workflow:
1. Place dataset(s) in `data/`
2. Run experiments via `main.py` with desired flags
3. Collect saved results and run notebooks to visualize and compare

## Citation
If you use this code or results in academic work, please cite the accompanying paper or this repository:
```bibtex
@article{yourpaper2024,
  title={Heterogeneous Ensemble Model for Software Effort Estimation},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```
Replace with the actual publication details when available.

## Contributing
Contributions are welcome — bug reports, issues, and pull requests. Suggested guidelines:
1. Fork the repository
2. Create a feature branch (feature/issue-xxx)
3. Add tests or notebooks demonstrating the change
4. Open a pull request describing the change

Please follow the existing code style and update `requirements.txt` if adding dependencies.

## License
This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

## Contact
Repository: https://github.com/64Sanjay/Heterogeneous-Ensemble-Model-for-Software-Effort-Estimation

Author / Maintainer: 64Sanjay  
For questions or collaboration, open an issue or contact via GitHub.

```
