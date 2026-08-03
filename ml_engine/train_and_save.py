import pandas as pd
import os
import sys

# Add directory to sys.path
sys.path.append(os.path.dirname(__file__))

from pipeline import FourAlgorithmPipeline, CLUSTER_PERSONAS

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'HealthGuard_Readmission_Data', 'readmission_dataset.csv')
    model_save_path = os.path.join(os.path.dirname(__file__), 'model_pipeline.joblib')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    print(f"[Train] Reading patient dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"[Train] Dataset shape: {df.shape}")
    
    pipeline = FourAlgorithmPipeline(dataset_path=dataset_path, n_clusters=4, n_rfe_features=15)
    metrics = pipeline.train(df)
    
    pipeline.save(model_save_path)
    
    print("\n========================================================")
    print("      4-ALGORITHM PIPELINE TRAINING METRICS SUMMARY      ")
    print("========================================================")
    print(f"Total Patient Records: {metrics['total_samples']}")
    print(f"Baseline Readmission Rate: {metrics['readmission_rate_overall']*100:.2f}%")
    print(f"RFE Selected Features ({metrics['rfe_selected_count']}): {pipeline.rfe_selected_feature_names}")
    print("\n[Algorithm 3: Logistic Regression Baseline]")
    print(f"  - Accuracy: {metrics['logistic_regression']['accuracy']*100:.2f}%")
    print(f"  - AUC-ROC:  {metrics['logistic_regression']['auc_roc']:.4f}")
    print(f"  - Precision:{metrics['logistic_regression']['precision']:.4f}")
    print(f"  - Recall:   {metrics['logistic_regression']['recall']:.4f}")
    
    print("\n[Algorithm 4: XGBoost Ensemble Engine]")
    print(f"  - Accuracy: {metrics['xgboost']['accuracy']*100:.2f}%")
    print(f"  - AUC-ROC:  {metrics['xgboost']['auc_roc']:.4f}")
    print(f"  - Precision:{metrics['xgboost']['precision']:.4f}")
    print(f"  - Recall:   {metrics['xgboost']['recall']:.4f}")
    print("========================================================")

if __name__ == '__main__':
    main()
