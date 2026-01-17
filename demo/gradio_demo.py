#!/usr/bin/env python
"""
Gradio Demo for Software Effort Estimation
Much simpler and more reliable than Flask!
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import gradio as gr

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel
from src.models.ml_models import XGBoostModel
from src.models.ensemble_model import EnsembleModel

# Global variables
models = {}
ensemble = None
feature_names = None
scaler_mean = None
scaler_std = None
raw_df = None

def init_models():
    global models, ensemble, feature_names, scaler_mean, scaler_std, raw_df
    
    print("Loading and training models...")
    
    loader = DataLoader("cocomo81")
    df = loader.load_raw_data()
    feature_names = loader.get_feature_names()
    raw_df = df.copy()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_pipeline(df, scale=True)
    
    scaler_mean = preprocessor.scaler.mean_
    scaler_std = preprocessor.scaler.scale_
    
    models["CBR"] = CBRModel(k=5)
    models["COCOMO"] = COCOMOModel()
    models["XGBoost"] = XGBoostModel()
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X, y)
    
    ensemble = EnsembleModel(ml_model_name="XGBoost", combination_rule="median")
    ensemble.fit(X, y)
    
    print("✓ Models ready!")


def predict_effort(loc, complexity):
    """Prediction function for Gradio"""
    
    complexity_map = {
        "Low": {"cplx": 0.85, "rely": 0.88},
        "Nominal": {"cplx": 1.0, "rely": 1.0},
        "High": {"cplx": 1.15, "rely": 1.15},
        "Very High": {"cplx": 1.30, "rely": 1.40}
    }
    
    features = complexity_map.get(complexity, {}).copy()
    features['loc'] = float(loc)
    
    # Build feature array
    feature_array = np.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        if name in features:
            feature_array[i] = features[name]
        else:
            feature_array[i] = raw_df.iloc[:, i].mean()
    
    # Scale
    scaled = (feature_array - scaler_mean) / scaler_std
    scaled = scaled.reshape(1, -1)
    
    # Predict
    results = []
    for name, model in models.items():
        pred = max(0, float(model.predict(scaled)[0]))
        results.append(f"**{name}**: {pred:.1f} person-months")
    
    ensemble_pred = max(0, float(ensemble.predict(scaled)[0]))
    results.append(f"\n**⭐ Ensemble (Recommended)**: {ensemble_pred:.1f} person-months")
    
    # Add interpretation
    if ensemble_pred < 50:
        interpretation = f"\n\n📅 *Small project - ~{int(ensemble_pred/2)} months with 2 developers*"
    elif ensemble_pred < 200:
        interpretation = f"\n\n📅 *Medium project - ~{int(ensemble_pred/4)} months with 4 developers*"
    else:
        interpretation = f"\n\n📅 *Large project - ~{int(ensemble_pred/8)} months with 8 developers*"
    
    return "\n".join(results) + interpretation


# Initialize models
init_models()

# Create Gradio interface
demo = gr.Interface(
    fn=predict_effort,
    inputs=[
        gr.Number(label="Lines of Code (KLOC)", value=100, minimum=1, maximum=10000),
        gr.Dropdown(
            choices=["Low", "Nominal", "High", "Very High"],
            label="Project Complexity",
            value="Nominal"
        )
    ],
    outputs=gr.Markdown(label="Effort Estimation Results"),
    title="🖥️ Software Effort Estimation",
    description="Heterogeneous Ensemble Model for estimating software development effort",
    examples=[
        [10, "Low"],
        [50, "Nominal"],
        [100, "High"],
        [500, "Very High"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Launching Gradio Demo...")
    print("=" * 60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # This creates a public URL!
    )
