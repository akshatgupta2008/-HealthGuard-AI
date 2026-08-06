import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.model import CLUSTER_PERSONAS, DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, FourAlgorithmPipeline


st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
<style>
  .hero { background: linear-gradient(90deg,#0f172a,#0f766e); color: #fff; padding: 20px; border-radius: 12px; }
  .card { background: #fff; padding: 16px; border-radius: 12px; box-shadow: 0 6px 18px rgba(2,6,23,0.06); }
  .section-title { font-size: 1rem; font-weight: 700; margin-bottom: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_PATIENT = {
    "patient_name": "Custom Patient",
    "age": 62,
    "gender": "Female",
    "admission_type": "Emergency",
    "primary_diagnosis_code": "I10",
    "num_prior_admissions": 2,
    "time_in_hospital": 5,
    "num_lab_procedures": 40,
    "num_medications": 15,
    "has_comorbidity": 1,
    "discharge_disposition": "Home",
    "insurance_type": "Medicare",
    "hospital_id": 101,
}


@st.cache_resource
def load_artifacts():
    if not DEFAULT_DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DEFAULT_DATASET_PATH}")

    df = pd.read_csv(DEFAULT_DATASET_PATH)

    if DEFAULT_MODEL_PATH.exists():
        try:
            pipeline = FourAlgorithmPipeline.load(str(DEFAULT_MODEL_PATH))
            return pipeline, df
        except Exception:
            pass

    pipeline = FourAlgorithmPipeline(dataset_path=str(DEFAULT_DATASET_PATH))
    pipeline.train(df)
    pipeline.save(str(DEFAULT_MODEL_PATH))
    return pipeline, df


def build_patient_form(defaults):
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient name", value=defaults["patient_name"])
            age = st.number_input("Age", min_value=18, max_value=100, value=int(defaults["age"]))
            gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(defaults["gender"]))
            admission_type = st.selectbox(
                "Admission type",
                ["Emergency", "Elective", "Urgent"],
                index=["Emergency", "Elective", "Urgent"].index(defaults["admission_type"]),
            )
            diagnosis = st.text_input("Primary diagnosis code", value=defaults["primary_diagnosis_code"])
            prior_admissions = st.number_input(
                "Prior admissions", min_value=0, max_value=20, value=int(defaults["num_prior_admissions"])
            )

        with col2:
            time_in_hospital = st.number_input(
                "Time in hospital (days)", min_value=1, max_value=30, value=int(defaults["time_in_hospital"])
            )
            num_lab_procedures = st.number_input(
                "Number of lab procedures", min_value=0, max_value=200, value=int(defaults["num_lab_procedures"])
            )
            num_medications = st.number_input(
                "Number of medications", min_value=0, max_value=100, value=int(defaults["num_medications"])
            )
            has_comorbidity = st.selectbox("Has comorbidity", [0, 1], index=int(defaults["has_comorbidity"]))
            discharge_disposition = st.selectbox(
                "Discharge disposition", ["Home", "Transfer", "SNF"], index=0
            )
            insurance_type = st.selectbox(
                "Insurance type", ["Medicare", "Private", "Medicaid"], index=0
            )

        submitted = st.form_submit_button("Generate readmission risk")

    payload = {
        "patient_name": patient_name,
        "age": age,
        "gender": gender,
        "admission_type": admission_type,
        "primary_diagnosis_code": diagnosis,
        "num_prior_admissions": prior_admissions,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "has_comorbidity": has_comorbidity,
        "discharge_disposition": discharge_disposition,
        "insurance_type": insurance_type,
        "hospital_id": defaults["hospital_id"],
    }
    return submitted, payload


st.markdown(
    """
    <div class="hero">
        <h1>HealthGuard AI</h1>
        <p>Clinical readmission risk demo with EDA, clustering, RFE, Logistic Regression, and XGBoost.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

pipeline, df = load_artifacts()

tabs = st.tabs(["Overview", "Live Predictor", "Dataset"])

with tabs[0]:
    st.markdown("## Overview")
    st.write(
        "HealthGuard AI is a notebook-backed healthcare ML project that predicts 30-day hospital readmission risk and surfaces interpretable drivers for each patient."
    )

    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric("Dataset rows", f"{len(df):,}")
    with metrics_cols[1]:
        st.metric("Overall readmission", f"{df['readmitted_within_30days'].mean() * 100:.1f}%")
    with metrics_cols[2]:
        st.metric("Selected features", pipeline.metrics.get("rfe_selected_count", 0))

    st.markdown("### Project scope")
    st.markdown(
        "- Exploratory data analysis\n- Feature engineering and selection\n- K-Means patient segmentation\n- Logistic Regression and XGBoost comparison\n- Ensemble risk scoring"
    )

with tabs[1]:
    st.markdown("## Live Predictor")
    st.info("Enter patient values and generate a readmission risk estimate.")

    submitted, patient_payload = build_patient_form(DEFAULT_PATIENT)

    if submitted:
        result = pipeline.predict_patient(patient_payload)
        ensemble = result["ensemble_result"]
        stages = result["pipeline_stages"]
        persona = stages["stage1_kmeans"]["persona"]

        score_cols = st.columns(4)
        with score_cols[0]:
            st.metric("Risk tier", ensemble["risk_tier"])
        with score_cols[1]:
            st.metric("Ensemble risk", f"{ensemble['readmission_risk_percentage']:.1f}%")
        with score_cols[2]:
            st.metric("Logistic score", f"{stages['stage3_logistic_regression']['percentage']:.1f}%")
        with score_cols[3]:
            st.metric("XGBoost score", f"{stages['stage4_xgboost']['percentage']:.1f}%")

        st.success(ensemble["recommendation"])
        st.write(f"Patient segment: **{persona['name']}**  |  {persona['description']}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Top logistic drivers")
            logistic_df = pd.DataFrame(stages["stage3_logistic_regression"]["top_feature_contributions"])
            st.dataframe(logistic_df, use_container_width=True, hide_index=True)
        with right:
            st.markdown("#### Top XGBoost drivers")
            xgb_df = pd.DataFrame(stages["stage4_xgboost"]["top_importance_drivers"])
            st.dataframe(xgb_df, use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown("## Dataset")
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)

    st.markdown("### Patient personas")
    st.write(CLUSTER_PERSONAS)
