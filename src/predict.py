"""
HealthGuard AI: Interactive Patient Readmission Risk Predictor CLI
------------------------------------------------------------------
Interactive CLI tool for evaluating real-time 30-day hospital readmission risk
using the trained 4-Algorithm ML Engine (K-Means + RFE + Logistic Reg + XGBoost).
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from src.model import FourAlgorithmPipeline, DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH

PRESET_PROFILES = {
    "1": {
        "name": "High Risk Comorbid Patient (Eleanor Vance)",
        "data": {
            "patient_name": "Eleanor Vance",
            "age": 78,
            "gender": "Female",
            "admission_type": "Emergency",
            "primary_diagnosis_code": "I10",
            "num_prior_admissions": 5,
            "time_in_hospital": 9,
            "num_lab_procedures": 62,
            "num_medications": 24,
            "has_comorbidity": 1,
            "discharge_disposition": "Transfer",
            "insurance_type": "Medicare",
            "hospital_id": 102
        }
    },
    "2": {
        "name": "Moderate Risk Chronic Care Patient (Arthur Pendelton)",
        "data": {
            "patient_name": "Arthur Pendelton",
            "age": 64,
            "gender": "Male",
            "admission_type": "Urgent",
            "primary_diagnosis_code": "E11",
            "num_prior_admissions": 2,
            "time_in_hospital": 4,
            "num_lab_procedures": 38,
            "num_medications": 14,
            "has_comorbidity": 1,
            "discharge_disposition": "Home",
            "insurance_type": "Private",
            "hospital_id": 105
        }
    },
    "3": {
        "name": "Low Risk Elective Recovery Patient (Sophia Martinez)",
        "data": {
            "patient_name": "Sophia Martinez",
            "age": 34,
            "gender": "Female",
            "admission_type": "Elective",
            "primary_diagnosis_code": "J45",
            "num_prior_admissions": 0,
            "time_in_hospital": 2,
            "num_lab_procedures": 18,
            "num_medications": 6,
            "has_comorbidity": 0,
            "discharge_disposition": "Home",
            "insurance_type": "Private",
            "hospital_id": 101
        }
    }
}

def load_or_train_pipeline() -> FourAlgorithmPipeline:
    if DEFAULT_MODEL_PATH.exists():
        try:
            return FourAlgorithmPipeline.load(str(DEFAULT_MODEL_PATH))
        except Exception as e:
            print(f"[Warning] Could not load model file: {e}")
            
    if DEFAULT_DATASET_PATH.exists():
        print("[Info] Training model pipeline...")
        df = pd.read_csv(DEFAULT_DATASET_PATH)
        pipeline = FourAlgorithmPipeline(dataset_path=str(DEFAULT_DATASET_PATH))
        pipeline.train(df)
        pipeline.save(str(DEFAULT_MODEL_PATH))
        return pipeline
    else:
        raise FileNotFoundError(f"Dataset not found at '{DEFAULT_DATASET_PATH}'")

def display_prediction_results(result: Dict[str, Any]):
    input_data = result["patient_input"]
    stages = result["pipeline_stages"]
    ensemble = result["ensemble_result"]
    
    print("\n" + "=" * 70)
    print(f"  PATIENT READMISSION RISK ASSESSMENT: {input_data.get('patient_name', 'Patient')}")
    print("=" * 70)
    print(f"Demographics: {input_data['age']} y/o {input_data['gender']} | Admission: {input_data['admission_type']} ({input_data['primary_diagnosis_code']})")
    print(f"Clinical Vitals: {input_data['time_in_hospital']} days in hospital, {input_data['num_prior_admissions']} prior admissions, Comorbidities: {'Yes' if input_data['has_comorbidity'] else 'No'}")
    
    print("\n" + "-" * 70)
    print("  STAGE 1: K-MEANS PATIENT SEGMENTATION & PERSONA CLUSTER")
    print("-" * 70)
    p_info = stages["stage1_kmeans"]["persona"]
    print(f"  Cluster ID    : Segment #{stages['stage1_kmeans']['cluster_id']}")
    print(f"  Persona Name  : {p_info['name']} [{p_info['badge']}]")
    print(f"  Description   : {p_info['description']}")
    
    print("\n" + "-" * 70)
    print("  STAGE 2 & 3: LOGISTIC BASELINE REGRESSION (EXPLAINABLE DRIVERS)")
    print("-" * 70)
    log_stage = stages["stage3_logistic_regression"]
    print(f"  Baseline Probability: {log_stage['percentage']}% ({log_stage['readmission_probability']:.4f})")
    print("  Top Symptom Risk Contributions:")
    for contrib in log_stage["top_feature_contributions"][:5]:
        direction = "[+] Risk" if contrib["impact"] == "Increases Risk" else "[-] Risk"
        print(f"    - {contrib['feature']:32s} : {contrib['contribution']:+.4f} ({direction})")
        
    print("\n" + "-" * 70)
    print("  STAGE 4: XGBoost ENSEMBLE ENGINE DRIVERS")
    print("-" * 70)
    xgb_stage = stages["stage4_xgboost"]
    print(f"  XGBoost Probability : {xgb_stage['percentage']}% ({xgb_stage['readmission_probability']:.4f})")
    print("  Top Gain Feature Importances:")
    for driver in xgb_stage["top_importance_drivers"][:5]:
        print(f"    - {driver['feature']:32s} : {driver['importance']*100:6.2f}% weight")
        
    print("\n" + "=" * 70)
    print("  ENSEMBLE READMISSION RISK SCORE & CLINICAL RECOMMENDATION")
    print("=" * 70)
    print(f"  Weighted Ensemble Score : {ensemble['readmission_risk_percentage']}% ({ensemble['ensemble_score']:.4f})")
    print(f"  Assigned Risk Tier      : {ensemble['risk_tier']}")
    print(f"  Clinical Recommendation : {ensemble['recommendation']}")
    print("=" * 70 + "\n")

def run_interactive_cli():
    print("=" * 70)
    print("    HEALTHGUARD AI: 4-ALGORITHM ML READMISSION PREDICTOR CLI    ")
    print("=" * 70)
    
    pipeline = load_or_train_pipeline()
    
    while True:
        print("\nSelect an option:")
        print("  1. Evaluate Eleanor Vance (High Risk Elderly Comorbid)")
        print("  2. Evaluate Arthur Pendelton (Moderate Risk Chronic Care)")
        print("  3. Evaluate Sophia Martinez (Low Risk Elective Recovery)")
        print("  4. Enter Custom Patient Vitals")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice in PRESET_PROFILES:
            profile = PRESET_PROFILES[choice]
            print(f"\nEvaluating preset: {profile['name']}...")
            result = pipeline.predict_patient(profile["data"])
            display_prediction_results(result)
        elif choice == "4":
            print("\n--- Enter Custom Patient Parameters ---")
            try:
                p_name = input("Patient Name [Default: Jane Doe]: ").strip() or "Jane Doe"
                age = int(input("Age (18-100) [Default: 65]: ").strip() or "65")
                gender = input("Gender (Female/Male) [Default: Female]: ").strip() or "Female"
                adm_type = input("Admission Type (Emergency/Elective/Urgent) [Default: Emergency]: ").strip() or "Emergency"
                diag = input("Primary Diagnosis Code (I10/E11/J45/M54) [Default: I10]: ").strip() or "I10"
                priors = int(input("Number of Prior Admissions (0-15) [Default: 2]: ").strip() or "2")
                stay = int(input("Time in Hospital (days 1-14) [Default: 5]: ").strip() or "5")
                comorbid = int(input("Has Comorbidities? (1 for Yes, 0 for No) [Default: 1]: ").strip() or "1")
                disch = input("Discharge Disposition (Home/Transfer/SNF) [Default: Home]: ").strip() or "Home"
                ins = input("Insurance Type (Medicare/Private/Medicaid) [Default: Medicare]: ").strip() or "Medicare"
                
                custom_patient = {
                    "patient_name": p_name,
                    "age": age,
                    "gender": gender,
                    "admission_type": adm_type,
                    "primary_diagnosis_code": diag,
                    "num_prior_admissions": priors,
                    "time_in_hospital": stay,
                    "num_lab_procedures": 40.0,
                    "num_medications": 15.0,
                    "has_comorbidity": comorbid,
                    "discharge_disposition": disch,
                    "insurance_type": ins,
                    "hospital_id": 101
                }
                result = pipeline.predict_patient(custom_patient)
                display_prediction_results(result)
            except Exception as e:
                print(f"[Error] Invalid input: {e}")
        elif choice == "5":
            print("Exiting Predictor. Goodbye!")
            break
        else:
            print("Invalid choice! Please select 1-5.")

if __name__ == '__main__':
    run_interactive_cli()
