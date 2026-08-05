import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.model import CLUSTER_PERSONAS, DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, FourAlgorithmPipeline
from src.predict import PRESET_PROFILES


st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🛡️",
    layout="wide",
)

# Minimal styling for a clean, focused presentation
st.markdown(
    """
<style>
  .hero { background: linear-gradient(90deg,#0f172a,#0f766e); color: #fff; padding: 20px; border-radius: 10px; }
  .feature { padding: 8px 0; }
  .small-card { background: #fff; padding: 12px; border-radius: 10px; box-shadow: 0 6px 18px rgba(2,6,23,0.06); }
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
        cols = st.columns(2)
st.markdown(
    """
    <div class="hero">
        <h1>HealthGuard AI</h1>
        <p>Focused clinical readmission risk demo — clean, shareable, and interview-ready.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Overview", "Live Predictor", "Notebooks"])

with tabs[0]:
    st.markdown("## Overview")
    st.markdown("HealthGuard AI is a compact end-to-end demonstration of a clinical readmission risk pipeline.")
    st.markdown("### Key features")
    st.markdown("- Patient segmentation with KMeans personas\\n- Feature selection with RFE\\n- Interpretable logistic regression + XGBoost ensemble\\n- Streamlit demo with a live predictor")

    with st.container():
        st.markdown("### Model & Data at a glance")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Dataset rows", f"{len(df):,}")
        with cols[1]:
            st.metric("Overall readmission", f"{df['readmitted_within_30days'].mean() * 100:.1f}%")
        with cols[2]:
            st.metric("Selected features", pipeline.metrics.get("rfe_selected_count", 0))

    st.markdown("### How to use this demo")
    st.markdown(
        "1) Use the Live Predictor to simulate patient scenarios. 2) Review the executed notebooks for methodology and training details."
    )

with tabs[1]:
    st.markdown("## Live Predictor")
    st.info("Enter values and click 'Generate readmission risk' to run the HealthGuard AI pipeline.")
    submitted, patient_payload = build_patient_form(default_patient)
    if submitted:
        result = pipeline.predict_patient(patient_payload)
        ensemble = result["ensemble_result"]
        stages = result["pipeline_stages"]
        persona = stages["stage1_kmeans"]["persona"]

        a, b, c, d = st.columns(4)
        with a:
            st.metric("Risk tier", ensemble["risk_tier"])
        with b:
            st.metric("Ensemble risk", f'{ensemble["readmission_risk_percentage"]:.1f}%')
        with c:
            st.metric("Logistic score", f'{stages["stage3_logistic_regression"]["percentage"]:.1f}%')
        with d:
            st.metric("XGBoost score", f'{stages["stage4_xgboost"]["percentage"]:.1f}%')

        st.success(ensemble["recommendation"])
        st.write(f"Patient segment: **{persona['name']}**  |  {persona['description']}")

        detail_left, detail_right = st.columns(2)
        with detail_left:
            st.markdown("#### Top logistic drivers")
            logistic_df = pd.DataFrame(stages["stage3_logistic_regression"]["top_feature_contributions"])
            st.dataframe(logistic_df, use_container_width=True, hide_index=True)
        with detail_right:
            st.markdown("#### Top XGBoost drivers")
            xgb_df = pd.DataFrame(stages["stage4_xgboost"]["top_importance_drivers"])
            st.dataframe(xgb_df, use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown("## Notebooks")
    st.markdown("Executed notebooks with narrative and training steps are included below.")
    nb1 = 'notebooks/01_exploratory_data_analysis.executed.ipynb'
    nb2 = 'notebooks/02_model_training_and_eval.executed.ipynb'
    st.markdown(f"- [Exploratory analysis]({nb1})")
    st.markdown(f"- [Training & evaluation]({nb2})")

    st.markdown("If you want these exported to HTML or pushed to a remote branch, tell me and I can do that next.")
with col4:
    st.markdown(f'<div class="card"><div class="section-title">Selected features</div><div style="font-size:1.7rem;font-weight:800;">{pipeline.metrics.get("rfe_selected_count", 0)}</div><div>after RFE</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.15, 0.85])

with left:
    st.markdown('<div class="card"><div class="section-title">Project snapshot</div></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <span class="pill">Python for data</span>
        <span class="pill">Regression</span>
        <span class="pill">Classification</span>
        <span class="pill">Clustering</span>
        <span class="pill">Metrics</span>
        <span class="pill">scikit-learn</span>
        <span class="pill">Kaggle-ready narrative</span>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "This version keeps the project focused on a single Streamlit entrypoint. It shows the dataset, model metrics, patient segmentation, and an interactive risk score without any frontend or backend stack."
    )

with right:
    st.markdown('<div class="card"><div class="section-title">Model facts</div></div>', unsafe_allow_html=True)
    st.write(f"Training rows: {pipeline.metrics.get('total_samples', len(df)):,}")
    st.write(f"Overall readmission rate: {pipeline.metrics.get('readmission_rate_overall', 0) * 100:.1f}%")
    st.write(f"Top segment example: {CLUSTER_PERSONAS.get(0, {}).get('name', 'Patient segment')}")

st.write("")
st.markdown("### Interactive Risk Predictor")
submitted, patient_payload = build_patient_form(default_patient)

if submitted:
    result = pipeline.predict_patient(patient_payload)
    ensemble = result["ensemble_result"]
    stages = result["pipeline_stages"]
    persona = stages["stage1_kmeans"]["persona"]

    a, b, c, d = st.columns(4)
    with a:
        st.metric("Risk tier", ensemble["risk_tier"])
    with b:
        st.metric("Ensemble risk", f'{ensemble["readmission_risk_percentage"]:.1f}%')
    with c:
        st.metric("Logistic score", f'{stages["stage3_logistic_regression"]["percentage"]:.1f}%')
    with d:
        st.metric("XGBoost score", f'{stages["stage4_xgboost"]["percentage"]:.1f}%')

    st.success(ensemble["recommendation"])
    st.write(f"Patient segment: **{persona['name']}**  |  {persona['description']}")

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("#### Top logistic drivers")
        logistic_df = pd.DataFrame(stages["stage3_logistic_regression"]["top_feature_contributions"])
        st.dataframe(logistic_df, use_container_width=True, hide_index=True)
    with detail_right:
        st.markdown("#### Top XGBoost drivers")
        xgb_df = pd.DataFrame(stages["stage4_xgboost"]["top_importance_drivers"])
        st.dataframe(xgb_df, use_container_width=True, hide_index=True)

st.write("")
st.markdown("### Dataset Preview")
st.dataframe(df.head(12), use_container_width=True, hide_index=True)

st.write("")
st.markdown("### How to present this project")
st.write(
    "1. Frame the problem as a binary classification task. 2. Explain the preprocessing and feature selection pipeline. 3. Report accuracy, precision, recall, and AUC. 4. Show one or two Kaggle-style case studies with a clear solution anatomy."
)
