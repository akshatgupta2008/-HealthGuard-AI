"""
HealthGuard AI: 4-Algorithm Model Benchmark & Evaluation Runner
----------------------------------------------------------------
Trains the 4-algorithm ML pipeline on patient records and prints formatted
evaluation summaries, cross-validation metrics, and feature Gain importances.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from src.model import FourAlgorithmPipeline, DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH

def run_ml_benchmark():
    print("=" * 70)
    print("     HEALTHGUARD AI 4-ALGORITHM ML BENCHMARK & EVALUATION     ")
    print("=" * 70)
    
    if not DEFAULT_DATASET_PATH.exists():
        print(f"[Error] Dataset not found at '{DEFAULT_DATASET_PATH.resolve()}'")
        return

    print(f"\n[1] Reading patient dataset from: {DEFAULT_DATASET_PATH.resolve()}")
    df = pd.read_csv(DEFAULT_DATASET_PATH)
    print(f"    Raw Dataset Observations: {len(df)} patient records, 15 features.")
    
    # 2. Train 4-Algorithm Pipeline
    pipeline = FourAlgorithmPipeline(dataset_path=str(DEFAULT_DATASET_PATH))
    metrics = pipeline.train(df)
    
    # Serialize model to disk
    pipeline.save(str(DEFAULT_MODEL_PATH))
    
    # 3. Print Benchmark Metrics Summary Table
    print("\n" + "=" * 70)
    print("          4-ALGORITHM EVALUATION METRICS SUMMARY TABLE          ")
    print("=" * 70)
    
    results = [
        {
            "Model Paradigm": "Logistic / Ridge Regression (Baseline)",
            "Accuracy": f"{metrics['logistic_regression']['accuracy']*100:.2f}%",
            "AUC-ROC": f"{metrics['logistic_regression']['auc_roc']:.4f}",
            "Precision": f"{metrics['logistic_regression']['precision']:.4f}",
            "Recall": f"{metrics['logistic_regression']['recall']:.4f}"
        },
        {
            "Model Paradigm": "XGBoost Ensemble Engine (Production)",
            "Accuracy": f"{metrics['xgboost']['accuracy']*100:.2f}%",
            "AUC-ROC": f"{metrics['xgboost']['auc_roc']:.4f}",
            "Precision": f"{metrics['xgboost']['precision']:.4f}",
            "Recall": f"{metrics['xgboost']['recall']:.4f}"
        }
    ]
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print("------------------------------------------------------------------------\n")
    
    # 4. Print Selected RFE Features & Top XGBoost Gain Feature Importances
    print(f"[4] RFE Top {metrics['rfe_selected_count']} Selected Features:")
    for feat in pipeline.rfe_selected_feature_names:
        print(f"    - {feat}")
        
    print("\n[5] Production XGBoost Feature Weight Distribution (Gain):")
    xgb_imp = metrics['xgboost']['feature_importances']
    sorted_imp = sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp:
        print(f"    - {feat:32s}: {imp*100:6.2f}%")
        
    print("\n" + "=" * 70)
    print("             BENCHMARK EVALUATION COMPLETE                     ")
    print("=" * 70)

if __name__ == '__main__':
    run_ml_benchmark()
