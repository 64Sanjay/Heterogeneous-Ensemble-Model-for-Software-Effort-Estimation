#!/usr/bin/env python
"""
Web-based Demo for Software Effort Estimation
Run with: python web_demo.py
Open browser: http://localhost:5000
"""

import os
import sys
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import numpy as np

try:
    from flask import Flask, render_template_string, request, jsonify
except ImportError:
    print("Flask not installed. Installing...")
    os.system('pip install flask')
    from flask import Flask, render_template_string, request, jsonify

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.cbr_model import CBRModel
from src.models.cocomo_model import COCOMOModel
from src.models.ml_models import XGBoostModel
from src.models.ensemble_model import EnsembleModel

app = Flask(__name__)

# Global variables for models
models = {}
ensemble = None
preprocessor = None
feature_names = None
scaler_mean = None
scaler_std = None
raw_df = None
is_initialized = False

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Software Effort Estimation Demo</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            color: rgba(255,255,255,0.9);
            margin-bottom: 30px;
        }
        .container {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .container h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #444;
        }
        .hint {
            font-size: 12px;
            color: #888;
            margin-top: 4px;
        }
        input[type="number"], select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="number"]:focus, select:focus {
            border-color: #667eea;
            outline: none;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 18px;
            font-weight: 600;
            width: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102,126,234,0.4);
        }
        button:active {
            transform: translateY(0);
        }
        .results {
            margin-top: 20px;
            animation: fadeIn 0.5s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .result-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            border-bottom: 1px solid #eee;
            transition: background-color 0.2s;
        }
        .result-item:hover {
            background-color: #f8f9fa;
        }
        .result-item.ensemble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            margin-top: 15px;
        }
        .result-item.ensemble:hover {
            background: linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%);
        }
        .model-name {
            font-weight: 500;
        }
        .effort-value {
            font-size: 18px;
            font-weight: 700;
        }
        .section-title {
            color: #666;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .status {
            text-align: center;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .info-box {
            background: #e8f4fd;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 0 8px 8px 0;
        }
        .footer {
            text-align: center;
            color: rgba(255,255,255,0.7);
            margin-top: 30px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>🖥️ Software Effort Estimation</h1>
    <p class="subtitle">Heterogeneous Ensemble Model - Interactive Demo</p>
    
    <div class="container">
        <h3>📊 Quick Estimation</h3>
        <div class="info-box">
            Enter the project size and complexity level for a quick effort estimate.
        </div>
        <div class="grid">
            <div class="form-group">
                <label for="loc">Lines of Code (KLOC) *</label>
                <input type="number" id="loc" value="100" min="1" max="10000" step="1" required>
                <div class="hint">Thousands of lines of code (1 KLOC = 1000 lines)</div>
            </div>
            <div class="form-group">
                <label for="complexity">Project Complexity</label>
                <select id="complexity">
                    <option value="low">🟢 Low - Simple, well-understood</option>
                    <option value="nominal" selected>🟡 Nominal - Average complexity</option>
                    <option value="high">🟠 High - Complex requirements</option>
                    <option value="very_high">🔴 Very High - Highly complex</option>
                </select>
                <div class="hint">Overall project complexity level</div>
            </div>
        </div>
        <button onclick="quickPredict()">🚀 Estimate Effort</button>
    </div>
    
    <div class="container">
        <h3>⚙️ Advanced Parameters</h3>
        <div class="info-box">
            Fine-tune your estimation with COCOMO cost drivers.
        </div>
        <div class="grid">
            <div class="form-group">
                <label for="rely">Reliability Required</label>
                <input type="number" id="rely" value="1.0" min="0.75" max="1.40" step="0.01">
                <div class="hint">0.75 (low) to 1.40 (very high)</div>
            </div>
            <div class="form-group">
                <label for="cplx">Product Complexity</label>
                <input type="number" id="cplx" value="1.0" min="0.70" max="1.65" step="0.01">
                <div class="hint">0.70 (very low) to 1.65 (extra high)</div>
            </div>
            <div class="form-group">
                <label for="time">Execution Time Constraint</label>
                <input type="number" id="time" value="1.0" min="1.0" max="1.66" step="0.01">
                <div class="hint">1.0 (nominal) to 1.66 (extra high)</div>
            </div>
            <div class="form-group">
                <label for="acap">Analyst Capability</label>
                <input type="number" id="acap" value="1.0" min="0.71" max="1.46" step="0.01">
                <div class="hint">0.71 (very high) to 1.46 (very low)</div>
            </div>
            <div class="form-group">
                <label for="pcap">Programmer Capability</label>
                <input type="number" id="pcap" value="1.0" min="0.70" max="1.42" step="0.01">
                <div class="hint">0.70 (very high) to 1.42 (very low)</div>
            </div>
            <div class="form-group">
                <label for="tool">Tool Usage</label>
                <input type="number" id="tool" value="1.0" min="0.83" max="1.24" step="0.01">
                <div class="hint">0.83 (very high) to 1.24 (very low)</div>
            </div>
        </div>
        <button onclick="advancedPredict()">🔧 Estimate with Advanced Parameters</button>
    </div>
    
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <p>Calculating estimates...</p>
    </div>
    
    <div class="container results" id="results" style="display: none;">
        <h3>📈 Estimation Results</h3>
        <div id="results-content"></div>
    </div>
    
    <div class="footer">
        <p>Heterogeneous Ensemble Model for Software Effort Estimation</p>
        <p>NIT Warangal - 2024</p>
    </div>
    
    <script>
        function showLoading() {
            document.getElementById('loading').classList.add('active');
            document.getElementById('results').style.display = 'none';
        }
        
        function hideLoading() {
            document.getElementById('loading').classList.remove('active');
        }
        
        function quickPredict() {
            showLoading();
            const loc = document.getElementById('loc').value;
            const complexity = document.getElementById('complexity').value;
            
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({loc: loc, complexity: complexity, mode: 'quick'})
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                showResults(data);
            })
            .catch(error => {
                hideLoading();
                alert('Error: ' + error);
            });
        }
        
        function advancedPredict() {
            showLoading();
            const params = {
                loc: document.getElementById('loc').value,
                rely: document.getElementById('rely').value,
                cplx: document.getElementById('cplx').value,
                time: document.getElementById('time').value,
                acap: document.getElementById('acap').value,
                pcap: document.getElementById('pcap').value,
                tool: document.getElementById('tool').value,
                mode: 'advanced'
            };
            
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                showResults(data);
            })
            .catch(error => {
                hideLoading();
                alert('Error: ' + error);
            });
        }
        
        function showResults(data) {
            const resultsDiv = document.getElementById('results');
            const contentDiv = document.getElementById('results-content');
            
            let html = '<p class="section-title">Individual Model Predictions:</p>';
            
            const modelIcons = {
                'CBR': '🔍',
                'COCOMO': '📐',
                'XGBoost': '🌳'
            };
            
            for (const [model, value] of Object.entries(data.predictions)) {
                if (model !== 'Ensemble') {
                    const icon = modelIcons[model] || '📊';
                    html += `<div class="result-item">
                        <span class="model-name">${icon} ${model}</span>
                        <span class="effort-value">${value.toFixed(1)} person-months</span>
                    </div>`;
                }
            }
            
            html += '<p class="section-title">Recommended Estimate:</p>';
            html += `<div class="result-item ensemble">
                <span class="model-name">⭐ Ensemble Prediction</span>
                <span class="effort-value">${data.predictions.Ensemble.toFixed(1)} person-months</span>
            </div>`;
            
            // Add interpretation
            const effort = data.predictions.Ensemble;
            let interpretation = '';
            if (effort < 50) {
                interpretation = '📅 Small project - approximately ' + Math.ceil(effort/2) + ' months with 2 developers';
            } else if (effort < 200) {
                interpretation = '📅 Medium project - approximately ' + Math.ceil(effort/4) + ' months with 4 developers';
            } else if (effort < 500) {
                interpretation = '📅 Large project - approximately ' + Math.ceil(effort/8) + ' months with 8 developers';
            } else {
                interpretation = '📅 Very large project - requires significant team and time investment';
            }
            
            html += `<div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                <strong>Interpretation:</strong><br>
                ${interpretation}
            </div>`;
            
            contentDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""


def init_models():
    """Initialize and train models"""
    global models, ensemble, preprocessor, feature_names, scaler_mean, scaler_std, raw_df, is_initialized
    
    print("Loading and training models...")
    
    try:
        loader = DataLoader("cocomo81")
        df = loader.load_raw_data()
        feature_names = loader.get_feature_names()
        raw_df = df.copy()
        
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_pipeline(df, scale=True)
        
        scaler_mean = preprocessor.scaler.mean_
        scaler_std = preprocessor.scaler.scale_
        
        models = {
            "CBR": CBRModel(k=5),
            "COCOMO": COCOMOModel(),
            "XGBoost": XGBoostModel(),
        }
        
        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X, y)
        
        print("  Training Ensemble...")
        ensemble = EnsembleModel(ml_model_name="XGBoost", combination_rule="median")
        ensemble.fit(X, y)
        
        is_initialized = True
        print("✓ Models ready!")
        return True
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False


def predict(features_dict):
    """Make prediction"""
    if not is_initialized:
        return {"error": "Models not initialized"}
    
    feature_array = np.zeros(len(feature_names))
    
    for i, name in enumerate(feature_names):
        if name in features_dict:
            feature_array[i] = float(features_dict[name])
        else:
            feature_array[i] = raw_df.iloc[:, i].mean()
    
    scaled = (feature_array - scaler_mean) / scaler_std
    scaled = scaled.reshape(1, -1)
    
    predictions = {}
    for name, model in models.items():
        pred = model.predict(scaled)[0]
        predictions[name] = max(0, float(pred))
    
    predictions["Ensemble"] = max(0, float(ensemble.predict(scaled)[0]))
    
    return predictions


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/health')
def health():
    return jsonify({"status": "ok", "models_ready": is_initialized})


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.json
        
        if data.get('mode') == 'quick':
            complexity_map = {
                "low": {"cplx": 0.85, "time": 1.0, "rely": 0.88},
                "nominal": {"cplx": 1.0, "time": 1.0, "rely": 1.0},
                "high": {"cplx": 1.15, "time": 1.11, "rely": 1.15},
                "very_high": {"cplx": 1.30, "time": 1.30, "rely": 1.40}
            }
            features = complexity_map.get(data.get('complexity', 'nominal'), {}).copy()
            features['loc'] = float(data.get('loc', 100))
        else:
            features = {k: float(v) for k, v in data.items() if k != 'mode'}
        
        predictions = predict(features)
        
        return jsonify({
            'predictions': predictions,
            'input': features,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  SOFTWARE EFFORT ESTIMATION - WEB DEMO")
    print("=" * 60)
    
    if init_models():
        print("\n" + "=" * 60)
        print("  🌐 Web Demo Running!")
        print("=" * 60)
        print("\n  Open your browser:")
        print("    → http://localhost:5000")
        print("    → http://127.0.0.1:5000")
        print("\n  Press CTRL+C to stop the server")
        print("=" * 60 + "\n")
        
        # Run Flask app
        app.run(
            debug=False, 
            host='127.0.0.1',  # Changed to localhost only for security
            port=5000,
            threaded=True
        )
    else:
        print("Failed to initialize models. Exiting.")
        sys.exit(1)
