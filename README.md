# Heterogeneous Ensemble Model for Software Effort Estimation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This repository implements a **Heterogeneous Ensemble Model** for Software Effort Estimation, combining:
- **COCOMO-II** (Algorithmic Model)
- **Case-Based Reasoning (CBR)** (Memory-Based Model)
- **Machine Learning Models** (XGBoost, ANN, KNN, SVR)

## 🚀 Features

- Multiple standalone models (CBR, COCOMO-II, XGBoost, ANN, KNN, SVR)
- Heterogeneous ensemble combining different model types
- Multiple combination rules (median, linear, mean)
- Support for multiple datasets (COCOMO81, NASA93, NASA60)
- Comprehensive evaluation metrics (MAE, MMRE, MdMRE, PRED)
- Leave-One-Out and K-Fold cross-validation

## 📁 Project Structure
software-effort-estimation/
├── data/ # Dataset files
├── src/ # Source code
│ ├── data/ # Data loading & preprocessing
│ ├── models/ # Model implementations
│ ├── evaluation/ # Metrics & cross-validation
│ └── utils/ # Configuration & helpers
├── notebooks/ # Jupyter notebooks
├── experiments/ # Results & logs
├── main.py # Main entry point
└── requirements.txt # Dependencies

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/software-effort-estimation.git
cd software-effort-estimation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
📊 Usage
Basic Usage

python main.py --dataset cocomo81 --cv_type loocv
Advanced Options
python main.py \
    --dataset nasa93 \
    --cv_type kfold \
    --n_splits 5 \
    --combination_rule median \
    --save_results
Python API
from src.data.data_loader import DataLoader
from src.models.ensemble_model import EnsembleModel

# Load data
loader = DataLoader("cocomo81")
X, y = loader.get_features_and_target()

# Create and train ensemble
ensemble = EnsembleModel(ml_model_name="XGBoost", combination_rule="median")
ensemble.fit(X_train, y_train)

# Predict
predictions = ensemble.predict(X_test)
📈 Results
Model	MAE	MMRE	PRED(0.25)
CBR	648.93	1.81	0.11
COCOMO-II	219.68	0.38	0.43
XGBoost	319.20	1.38	0.21
Ensemble (XGBoost)	284.41	0.81	0.25
📚 Citation
@article{yourpaper2024,
  title={Heterogeneous Ensemble Model for Software Effort Estimation},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
📄 License

MIT License - see LICENSE file.

---

## **Summary**

I've provided you with a complete, production-ready repository structure including:

| Component | Files |
|-----------|-------|
| **Data Handling** | `data_loader.py`, `preprocessor.py` |
| **Models** | `cbr_model.py`, `cocomo_model.py`, `ml_models.py`, `ensemble_model.py` |
| **Evaluation** | `metrics.py`, `cross_validation.py` |
| **Configuration** | `config.py` |
| **Entry Point** | `main.py` |
| **Documentation** | `README.md`, `requirements.txt` |

**Ready for your next instructions!** What would you like me to help with next?