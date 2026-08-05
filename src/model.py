"""
HealthGuard AI: 4-Algorithm Machine Learning Pipeline Module
-------------------------------------------------------------
Implements sequential 4-algorithm architecture:
1. Stage 1: Unsupervised K-Means Patient Segmentation (k=4)
2. Stage 2: Recursive Feature Elimination (RFE, top 15 features)
3. Stage 3: Ridge Logistic Regression Baseline & Explainable Risk Coefficients
4. Stage 4: High-Precision XGBoost Ensemble Engine & Gain Importances
"""

import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb

from src.data_processing import load_and_preprocess_raw_data, NUMERICAL_COLS, CATEGORICAL_COLS

DEFAULT_DATASET_PATH = Path(__file__).parent.parent / "data" / "readmission_dataset.csv"
DEFAULT_MODEL_PATH = Path(__file__).parent / "model_pipeline.joblib"

CLUSTER_PERSONAS = {
    0: {
        "name": "High Risk Elderly Comorbid",
        "badge": "Critical Risk Persona",
        "description": "Older patient demographic with elevated prior admissions, multiple chronic comorbidities, and extended hospital length-of-stay.",
        "risk_level": "High"
    },
    1: {
        "name": "Acute Emergency High-Procedure",
        "badge": "Acute Care Persona",
        "description": "Patients admitted via Emergency with heavy diagnostic lab work, high medication counts, and acute clinical interventions.",
        "risk_level": "High"
    },
    2: {
        "name": "Moderate Risk Chronic Care",
        "badge": "Managed Care Persona",
        "description": "Middle-to-older age patients with recurring outpatient/urgent visits, stable vital parameters, and baseline comorbidity.",
        "risk_level": "Moderate"
    },
    3: {
        "name": "Low Risk Elective Recovery",
        "badge": "Low Risk Persona",
        "description": "Younger/mid-age elective surgery or planned routine stay patients with minimal prior readmissions and smooth discharge trajectory.",
        "risk_level": "Low"
    }
}

class FourAlgorithmPipeline:
    def __init__(self, dataset_path: str = None, n_clusters: int = 4, n_rfe_features: int = 15):
        self.dataset_path = str(dataset_path) if dataset_path else str(DEFAULT_DATASET_PATH)
        self.n_clusters = n_clusters
        self.n_rfe_features = n_rfe_features
        
        self.preprocessor = None
        self.kmeans_model = None
        self.rfe_selector = None
        self.logistic_model = None
        self.xgboost_model = None
        
        self.feature_names = []
        self.rfe_selected_feature_names = []
        self.rfe_history = []
        self.cluster_centroids = {}
        self.cluster_counts = {}
        self.metrics = {}
        self.is_trained = False

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains the full 4-stage ML pipeline on the input dataset."""
        print("[ML Engine] Starting 4-Algorithm Pipeline Training...")
        X_raw, y = load_and_preprocess_raw_data(df)
        
        num_cols = [c for c in NUMERICAL_COLS if c in X_raw.columns]
        cat_cols = [c for c in CATEGORICAL_COLS if c in X_raw.columns]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
            ]
        )
        
        X_processed = self.preprocessor.fit_transform(X_raw)
        cat_encoder = self.preprocessor.named_transformers_['cat']
        encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        self.feature_names = num_cols + encoded_cat_names
        
        # STAGE 1: K-Means Clustering
        print("[ML Engine] Stage 1: Running K-Means Patient Segmentation...")
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans_clusters = self.kmeans_model.fit_predict(X_processed)
        
        cluster_ohe = pd.get_dummies(kmeans_clusters, prefix='patient_segment').astype(float).values
        segment_feature_names = [f'patient_segment_Cluster_{i}' for i in range(self.n_clusters)]
        
        X_augmented = np.hstack([X_processed, cluster_ohe])
        augmented_feature_names = self.feature_names + segment_feature_names
        
        for i in range(self.n_clusters):
            mask = (kmeans_clusters == i)
            self.cluster_counts[i] = int(np.sum(mask))
            sub_df = X_raw[mask]
            self.cluster_centroids[i] = {
                "avg_age": float(sub_df['age'].mean()) if 'age' in sub_df else 0,
                "avg_hospital_days": float(sub_df['time_in_hospital'].mean()) if 'time_in_hospital' in sub_df else 0,
                "avg_prior_admissions": float(sub_df['num_prior_admissions'].mean()) if 'num_prior_admissions' in sub_df else 0,
                "avg_lab_procedures": float(sub_df['num_lab_procedures'].mean()) if 'num_lab_procedures' in sub_df else 0,
                "avg_medications": float(sub_df['num_medications'].mean()) if 'num_medications' in sub_df else 0,
                "comorbidity_rate": float(sub_df['has_comorbidity'].mean()) if 'has_comorbidity' in sub_df else 0,
                "readmission_rate": float(y[mask].mean()) if y is not None else 0,
                "count": int(np.sum(mask))
            }

        # STAGE 2: Recursive Feature Elimination (RFE)
        print(f"[ML Engine] Stage 2: Running Recursive Feature Elimination (RFE) to pick top {self.n_rfe_features} features...")
        base_estimator = LogisticRegression(max_iter=1000, random_state=42)
        
        self.rfe_history = []
        n_curr_features = X_augmented.shape[1]
        step = 1
        curr_indices = np.arange(n_curr_features)
        while len(curr_indices) > self.n_rfe_features:
            temp_model = LogisticRegression(max_iter=500, random_state=42)
            temp_model.fit(X_augmented[:, curr_indices], y)
            coefs = np.abs(temp_model.coef_[0])
            acc = float(accuracy_score(y, temp_model.predict(X_augmented[:, curr_indices])))
            
            worst_local_idx = np.argmin(coefs)
            dropped_feature = augmented_feature_names[curr_indices[worst_local_idx]]
            
            self.rfe_history.append({
                "step": step,
                "remaining_features_count": len(curr_indices),
                "dropped_feature": dropped_feature,
                "dropped_importance": float(coefs[worst_local_idx]),
                "step_accuracy": round(acc, 4)
            })
            
            curr_indices = np.delete(curr_indices, worst_local_idx)
            step += 1

        self.rfe_selector = RFE(estimator=base_estimator, n_features_to_select=self.n_rfe_features)
        X_rfe = self.rfe_selector.fit_transform(X_augmented, y)
        rfe_support = self.rfe_selector.support_
        self.rfe_selected_feature_names = [augmented_feature_names[i] for i, supp in enumerate(rfe_support) if supp]

        # STAGE 3: Ridge Logistic Regression Baseline
        print("[ML Engine] Stage 3: Training Logistic Baseline Regression...")
        self.logistic_model = LogisticRegression(max_iter=1000, random_state=42)
        self.logistic_model.fit(X_rfe, y)
        y_pred_log = self.logistic_model.predict(X_rfe)
        y_prob_log = self.logistic_model.predict_proba(X_rfe)[:, 1]
        
        # STAGE 4: XGBoost Ensemble Engine
        print("[ML Engine] Stage 4: Training XGBoost Predictive Engine...")
        self.xgboost_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.xgboost_model.fit(X_rfe, y)
        y_pred_xgb = self.xgboost_model.predict(X_rfe)
        y_prob_xgb = self.xgboost_model.predict_proba(X_rfe)[:, 1]

        self.metrics = {
            "logistic_regression": {
                "accuracy": float(accuracy_score(y, y_pred_log)),
                "auc_roc": float(roc_auc_score(y, y_prob_log)),
                "precision": float(precision_score(y, y_pred_log, zero_division=0)),
                "recall": float(recall_score(y, y_pred_log, zero_division=0)),
                "confusion_matrix": confusion_matrix(y, y_pred_log).tolist(),
                "coefficients": {name: float(coef) for name, coef in zip(self.rfe_selected_feature_names, self.logistic_model.coef_[0])}
            },
            "xgboost": {
                "accuracy": float(accuracy_score(y, y_pred_xgb)),
                "auc_roc": float(roc_auc_score(y, y_prob_xgb)),
                "precision": float(precision_score(y, y_pred_xgb, zero_division=0)),
                "recall": float(recall_score(y, y_pred_xgb, zero_division=0)),
                "confusion_matrix": confusion_matrix(y, y_pred_xgb).tolist(),
                "feature_importances": {name: float(imp) for name, imp in zip(self.rfe_selected_feature_names, self.xgboost_model.feature_importances_)}
            },
            "total_samples": len(df),
            "readmission_rate_overall": float(y.mean()),
            "rfe_selected_count": len(self.rfe_selected_feature_names)
        }
        
        self.is_trained = True
        print("[ML Engine] Training complete! Metrics calculated successfully.")
        return self.metrics

    def predict_patient(self, patient_dict: Dict[str, Any], weight_logistic: float = 0.35, weight_xgb: float = 0.65) -> Dict[str, Any]:
        """Runs single patient prediction through all 4 pipeline stages."""
        if not self.is_trained:
            raise RuntimeError("ML Pipeline is not trained yet!")

        df_patient = pd.DataFrame([patient_dict])
        X_raw, _ = load_and_preprocess_raw_data(df_patient)
        
        X_processed = self.preprocessor.transform(X_raw)
        cluster_id = int(self.kmeans_model.predict(X_processed)[0])
        cluster_info = CLUSTER_PERSONAS.get(cluster_id, {
            "name": f"Segment Cluster {cluster_id}",
            "badge": "Patient Group",
            "description": "Patient categorized into cluster segment.",
            "risk_level": "Moderate"
        })
        cluster_stats = self.cluster_centroids.get(cluster_id, {})
        
        cluster_ohe = np.zeros((1, self.n_clusters))
        cluster_ohe[0, cluster_id] = 1.0
        X_augmented = np.hstack([X_processed, cluster_ohe])
        
        rfe_support = self.rfe_selector.support_
        X_rfe = X_augmented[:, rfe_support]
        
        logistic_prob = float(self.logistic_model.predict_proba(X_rfe)[0, 1])
        logistic_coefs = self.logistic_model.coef_[0]
        
        feature_contributions = []
        for feat_name, val, coef in zip(self.rfe_selected_feature_names, X_rfe[0], logistic_coefs):
            contrib = float(val * coef)
            feature_contributions.append({
                "feature": feat_name,
                "value": round(float(val), 3),
                "coefficient": round(float(coef), 4),
                "contribution": round(contrib, 4),
                "impact": "Increases Risk" if contrib > 0 else "Decreases Risk"
            })
        feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        xgb_prob = float(self.xgboost_model.predict_proba(X_rfe)[0, 1])
        xgb_importances = self.xgboost_model.feature_importances_
        xgb_top_drivers = []
        for feat_name, imp in zip(self.rfe_selected_feature_names, xgb_importances):
            xgb_top_drivers.append({
                "feature": feat_name,
                "importance": round(float(imp), 4)
            })
        xgb_top_drivers.sort(key=lambda x: x['importance'], reverse=True)

        total_w = weight_logistic + weight_xgb
        w_log_norm = weight_logistic / total_w
        w_xgb_norm = weight_xgb / total_w
        ensemble_score = float(w_log_norm * logistic_prob + w_xgb_norm * xgb_prob)
        
        if ensemble_score >= 0.70:
            risk_tier = "Critical Risk"
            recommendation = "Immediate Clinical Intervention: Schedule mandatory outpatient follow-up within 48-72 hours, assign care manager, and optimize discharge regimen."
            badge_color = "#ef4444"
        elif ensemble_score >= 0.45:
            risk_tier = "High Risk"
            recommendation = "Enhanced Monitoring: Recommend medication reconciliation call within 5 days and telehealth check-in at 14 days."
            badge_color = "#f97316"
        elif ensemble_score >= 0.25:
            risk_tier = "Moderate Risk"
            recommendation = "Standard Post-Discharge Care: Provide standard discharge instructions and standard primary care follow-up."
            badge_color = "#eab308"
        else:
            risk_tier = "Low Risk"
            recommendation = "Routine Recovery Pathway: Patient demonstrates favorable clinical profile with low expected readmission probability."
            badge_color = "#10b981"
            
        return {
            "patient_input": patient_dict,
            "pipeline_stages": {
                "stage1_kmeans": {
                    "cluster_id": cluster_id,
                    "persona": cluster_info,
                    "cluster_centroids": cluster_stats
                },
                "stage2_rfe": {
                    "total_features_evaluated": X_augmented.shape[1],
                    "selected_features_count": len(self.rfe_selected_feature_names),
                    "selected_features": self.rfe_selected_feature_names
                },
                "stage3_logistic_regression": {
                    "readmission_probability": round(logistic_prob, 4),
                    "percentage": round(logistic_prob * 100, 1),
                    "top_feature_contributions": feature_contributions[:8]
                },
                "stage4_xgboost": {
                    "readmission_probability": round(xgb_prob, 4),
                    "percentage": round(xgb_prob * 100, 1),
                    "top_importance_drivers": xgb_top_drivers[:8]
                }
            },
            "ensemble_result": {
                "ensemble_score": round(ensemble_score, 4),
                "readmission_risk_percentage": round(ensemble_score * 100, 1),
                "risk_tier": risk_tier,
                "badge_color": badge_color,
                "recommendation": recommendation,
                "weights_used": {
                    "logistic_weight": round(w_log_norm, 2),
                    "xgboost_weight": round(w_xgb_norm, 2)
                }
            }
        }

    def save(self, filepath: str = str(DEFAULT_MODEL_PATH)):
        """Persists trained model pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[ML Engine] Saved trained 4-Algorithm Pipeline to {filepath}")

    @staticmethod
    def load(filepath: str = str(DEFAULT_MODEL_PATH)) -> "FourAlgorithmPipeline":
        """Loads trained model pipeline from disk."""
        pipeline = joblib.load(filepath)
        print(f"[ML Engine] Successfully loaded 4-Algorithm Pipeline from {filepath}")
        return pipeline

def train_and_serialize_pipeline(dataset_path: Path = DEFAULT_DATASET_PATH, model_save_path: Path = DEFAULT_MODEL_PATH) -> FourAlgorithmPipeline:
    """Helper function to load data, fit pipeline, and persist to disk."""
    df = pd.read_csv(dataset_path)
    pipeline = FourAlgorithmPipeline(dataset_path=str(dataset_path))
    pipeline.train(df)
    pipeline.save(str(model_save_path))
    return pipeline

if __name__ == '__main__':
    train_and_serialize_pipeline()
