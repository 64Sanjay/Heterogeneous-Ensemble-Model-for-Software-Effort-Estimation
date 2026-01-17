#!/usr/bin/env python
"""
Simple test script for the demo
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from interactive_demo import EffortEstimationDemo


def main():
    print("=" * 60)
    print("TESTING EFFORT ESTIMATION DEMO")
    print("=" * 60)
    
    # Initialize demo
    demo = EffortEstimationDemo()
    
    # Load and train
    demo.load_and_train("cocomo81")
    
    # Show feature ranges
    demo.show_feature_ranges()
    
    # Run sample predictions
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Test 1: Small project
    print("\n--- Test 1: Small Project ---")
    demo.quick_test(10, "low")
    
    # Test 2: Medium project
    print("\n--- Test 2: Medium Project ---")
    demo.quick_test(50, "nominal")
    
    # Test 3: Large project
    print("\n--- Test 3: Large Project ---")
    demo.quick_test(200, "high")
    
    # Test 4: Custom parameters
    print("\n--- Test 4: Custom Parameters ---")
    custom_features = {
        'loc': 100,
        'rely': 1.15,
        'cplx': 1.30,
        'acap': 0.86,
        'pcap': 0.86,
    }
    predictions = demo.predict_from_raw(custom_features)
    
    print(f"\nCustom Project Parameters:")
    for k, v in custom_features.items():
        print(f"  {k}: {v}")
    
    print(f"\nPredictions:")
    for model, pred in predictions.items():
        print(f"  {model}: {pred:.2f} person-months")
    
    # Test 5: Compare with actual
    print("\n--- Test 5: Compare with Actual ---")
    demo.compare_with_actual(loc=113, actual_effort=2040, rely=0.88, cplx=0.70)
    
    # Test 6: Batch prediction
    print("\n--- Test 6: Batch Prediction ---")
    projects = [
        {"loc": 25, "cplx": 0.85},
        {"loc": 50, "cplx": 1.00},
        {"loc": 100, "cplx": 1.15},
        {"loc": 200, "cplx": 1.30},
    ]
    results = demo.batch_predict(projects)
    print("\nBatch Results:")
    print(results[['Project', 'loc', 'cplx', 'CBR', 'XGBoost', 'Ensemble']].to_string())
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
