"""
HealthGuard AI: FastAPI Serving Microservice Module
---------------------------------------------------
Exposes HTTP REST API endpoints for real-time patient readmission risk scoring,
model metric inspections, 4-algorithm metadata, and retrains triggering.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model import FourAlgorithmPipeline, CLUSTER_PERSONAS, DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH

app = FastAPI(
    title="HealthGuard AI - 4-Algorithm ML Readmission Microservice",
    version="2.0.0",
    description="Python ML engine leveraging K-Means, RFE, Ridge Logistic Regression, and XGBoost for patient readmission analytics."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_instance: Optional[FourAlgorithmPipeline] = None

def get_pipeline() -> FourAlgorithmPipeline:
    global pipeline_instance
    if pipeline_instance is not None and pipeline_instance.is_trained:
        return pipeline_instance
        
    if DEFAULT_MODEL_PATH.exists():
        try:
            pipeline_instance = FourAlgorithmPipeline.load(str(DEFAULT_MODEL_PATH))
            return pipeline_instance
        except Exception as e:
            print(f"[FastAPI] Warning loading model file: {e}")
            
    if DEFAULT_DATASET_PATH.exists():
        print("[FastAPI] Training model pipeline on startup...")
        df = pd.read_csv(DEFAULT_DATASET_PATH)
        pipeline_instance = FourAlgorithmPipeline(dataset_path=str(DEFAULT_DATASET_PATH))
        pipeline_instance.train(df)
        pipeline_instance.save(str(DEFAULT_MODEL_PATH))
        return pipeline_instance
    else:
        raise HTTPException(status_code=500, detail="Dataset not found to train ML Pipeline.")

class PatientInput(BaseModel):
    patient_name: Optional[str] = Field("Eleanor Vance", example="Eleanor Vance")
    name: Optional[str] = Field("Eleanor Vance", example="Eleanor Vance")
    age: int = Field(..., example=75)
    gender: str = Field("Female", example="Female")
    admission_type: str = Field("Emergency", example="Emergency")
    primary_diagnosis_code: str = Field("E11", example="E11")
    num_prior_admissions: int = Field(2, example=2)
    time_in_hospital: int = Field(5, example=5)
    num_lab_procedures: float = Field(45.0, example=45.0)
    num_medications: float = Field(20.0, example=20.0)
    has_comorbidity: int = Field(1, example=1)
    discharge_disposition: str = Field("Home", example="Home")
    insurance_type: str = Field("Medicare", example="Medicare")
    hospital_id: int = Field(1, example=1)
    weight_logistic: Optional[float] = 0.35
    weight_xgb: Optional[float] = 0.65

    class Config:
        extra = "allow"

@app.on_event("startup")
def startup_event():
    try:
        get_pipeline()
        print("[FastAPI] Startup complete. ML Pipeline is loaded & ready!")
    except Exception as e:
        print(f"[FastAPI] Startup training postponed: {e}")

@app.get("/health")
def health_check():
    pipeline = get_pipeline()
    return {
        "status": "online",
        "service": "HealthGuard AI 4-Algorithm ML Microservice (FastAPI + XGBoost + scikit-learn)",
        "is_trained": pipeline.is_trained if pipeline else False,
        "n_clusters": pipeline.n_clusters if pipeline else 4,
        "n_rfe_features": pipeline.n_rfe_features if pipeline else 15
    }

@app.get("/metrics")
def get_metrics():
    pipeline = get_pipeline()
    return {
        "metrics": pipeline.metrics,
        "selected_features": pipeline.rfe_selected_feature_names,
        "cluster_centroids": pipeline.cluster_centroids,
        "cluster_personas": CLUSTER_PERSONAS,
        "rfe_history": pipeline.rfe_history
    }

@app.get("/pipeline-details")
def get_pipeline_details():
    pipeline = get_pipeline()
    return {
        "pipeline_description": {
            "algorithm1": {
                "name": "K-Means Clustering",
                "type": "Unsupervised Learning",
                "role": "Discovers hidden patient personas & segment clusters to enrich patient features.",
                "n_clusters": pipeline.n_clusters,
                "personas": CLUSTER_PERSONAS,
                "centroids": pipeline.cluster_centroids
            },
            "algorithm2": {
                "name": "Recursive Feature Elimination (RFE)",
                "type": "Recursive Selection",
                "role": "Recursively prunes weakest predictors until top critical features remain.",
                "history": pipeline.rfe_history,
                "selected_features": pipeline.rfe_selected_feature_names
            },
            "algorithm3": {
                "name": "Logistic / Ridge Regression",
                "type": "Baseline Supervised Learning",
                "role": "Generates interpretable baseline probabilities & linear symptom risk weights.",
                "metrics": pipeline.metrics.get("logistic_regression", {})
            },
            "algorithm4": {
                "name": "XGBoost Classifier",
                "type": "Advanced Ensemble Learning",
                "role": "Captures non-linear feature interactions and high-dimensional clinical relationships.",
                "metrics": pipeline.metrics.get("xgboost", {})
            }
        }
    }

@app.post("/predict")
def predict_patient_readmission(patient: PatientInput):
    pipeline = get_pipeline()
    patient_dict = patient.dict()
    w_log = patient_dict.pop("weight_logistic", 0.35)
    w_xgb = patient_dict.pop("weight_xgb", 0.65)
    p_name = patient_dict.get("patient_name") or patient_dict.get("name") or "Anonymous Patient"
    patient_dict["patient_name"] = p_name
    patient_dict["name"] = p_name
    
    result = pipeline.predict_patient(patient_dict, weight_logistic=w_log, weight_xgb=w_xgb)
    return result

@app.post("/train")
def retrain_pipeline():
    global pipeline_instance
    df = pd.read_csv(DEFAULT_DATASET_PATH)
    pipeline_instance = FourAlgorithmPipeline(dataset_path=str(DEFAULT_DATASET_PATH))
    metrics = pipeline_instance.train(df)
    pipeline_instance.save(str(DEFAULT_MODEL_PATH))
    return {"status": "success", "message": "ML Pipeline retrained successfully!", "metrics": metrics}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
