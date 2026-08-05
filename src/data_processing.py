"""
HealthGuard AI: Clinical Data Processing & Feature Ingestion Module
--------------------------------------------------------------------
Handles clinical dataset loading, missing value imputations, categorical encoding,
and feature matrix/target vector separation.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Tuple

DEFAULT_DATASET_PATH = Path(__file__).parent.parent / "data" / "readmission_dataset.csv"
TARGET_COL = 'readmitted_within_30days'
DROP_COLS = ['patient_id', 'patient_name', 'name', 'days_to_readmission', TARGET_COL]

NUMERICAL_COLS = [
    'age', 'num_prior_admissions', 'time_in_hospital', 
    'num_lab_procedures', 'num_medications', 'has_comorbidity', 'hospital_id'
]
CATEGORICAL_COLS = [
    'gender', 'admission_type', 'primary_diagnosis_code', 
    'discharge_disposition', 'insurance_type'
]

def load_and_preprocess_raw_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleans raw patient dataframe and splits feature matrix from target vector.
    
    Returns:
        X_raw (pd.DataFrame): Unprocessed feature matrix with handled NaNs.
        y (pd.Series): Target 30-day readmission vector (or None if inference).
    """
    df_clean = df.copy()
    
    # Handle missing numeric values safely (handles 1-row dataframes)
    if 'num_lab_procedures' in df_clean.columns:
        med_labs = df_clean['num_lab_procedures'].median()
        if pd.isna(med_labs):
            med_labs = 40.0
        df_clean['num_lab_procedures'] = df_clean['num_lab_procedures'].fillna(med_labs)
        
    if 'num_medications' in df_clean.columns:
        med_meds = df_clean['num_medications'].median()
        if pd.isna(med_meds):
            med_meds = 15.0
        df_clean['num_medications'] = df_clean['num_medications'].fillna(med_meds)
        
    for col in NUMERICAL_COLS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
            
    for col in CATEGORICAL_COLS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("Unknown")
        
    X_raw = df_clean.drop(columns=[c for c in DROP_COLS if c in df_clean.columns])
    y = df_clean[TARGET_COL] if TARGET_COL in df_clean.columns else None
    
    return X_raw, y

if __name__ == '__main__':
    if DEFAULT_DATASET_PATH.exists():
        df = pd.read_csv(DEFAULT_DATASET_PATH)
        X, y = load_and_preprocess_raw_data(df)
        print("[SUCCESS] Data Processing Module Verified!")
        print(f" -> Cleaned Dataset Shape: {X.shape}")
        print(f" -> Target Mean Readmission Rate: {y.mean()*100:.2f}%")
