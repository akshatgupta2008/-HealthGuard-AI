import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.model import CLUSTER_PERSONAS, DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, FourAlgorithmPipeline
from src.predict import PRESET_PROFILES


st.set_page_config(
    page_title="Data Science Portfolio | HealthGuard AI",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
<style>
    .main { background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%); }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0f766e 100%);
        color: white;
        padding: 28px 30px;
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
    }
    .hero h1 { margin: 0; font-size: 2.2rem; }
    .hero p { margin: 10px 0 0; color: rgba(255,255,255,0.84); font-size: 1rem; }
    .card {
        background: rgba(255,255,255,0.84);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 18px;
        padding: 18px 18px 14px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }
    .pill {
        display: inline-block;
        margin: 0 8px 8px 0;
        padding: 5px 12px;
        border-radius: 999px;
        background: #e2e8f0;
        color: #0f172a;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .section-title { font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem; }
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

        with cols[0]:
            patient_name = st.text_input("Patient name", value=defaults["patient_name"])
            age = st.number_input("Age", min_value=18, max_value=100, value=int(defaults["age"]))
            gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=["Female", "Male", "Other"].index(defaults["gender"]) if defaults["gender"] in ["Female", "Male", "Other"] else 0)
            admission_type = st.selectbox("Admission type", ["Emergency", "Urgent", "Elective"], index=["Emergency", "Urgent", "Elective"].index(defaults["admission_type"]) if defaults["admission_type"] in ["Emergency", "Urgent", "Elective"] else 0)
            primary_diagnosis_code = st.selectbox("Primary diagnosis code", ["I10", "E11", "J45", "M54"], index=["I10", "E11", "J45", "M54"].index(defaults["primary_diagnosis_code"]) if defaults["primary_diagnosis_code"] in ["I10", "E11", "J45", "M54"] else 0)
            num_prior_admissions = st.number_input("Prior admissions", min_value=0, max_value=15, value=int(defaults["num_prior_admissions"]))

        with cols[1]:
            time_in_hospital = st.number_input("Days in hospital", min_value=1, max_value=30, value=int(defaults["time_in_hospital"]))
            num_lab_procedures = st.number_input("Lab procedures", min_value=0, max_value=200, value=int(defaults["num_lab_procedures"]))
            num_medications = st.number_input("Medications", min_value=0, max_value=100, value=int(defaults["num_medications"]))
            has_comorbidity = st.selectbox("Comorbidity", [0, 1], index=int(defaults["has_comorbidity"]))
            discharge_disposition = st.selectbox("Discharge disposition", ["Home", "Transfer", "SNF"], index=["Home", "Transfer", "SNF"].index(defaults["discharge_disposition"]) if defaults["discharge_disposition"] in ["Home", "Transfer", "SNF"] else 0)
            insurance_type = st.selectbox("Insurance type", ["Medicare", "Private", "Medicaid"], index=["Medicare", "Private", "Medicaid"].index(defaults["insurance_type"]) if defaults["insurance_type"] in ["Medicare", "Private", "Medicaid"] else 0)

        hospital_id = st.number_input("Hospital ID", min_value=1, max_value=999, value=int(defaults["hospital_id"]))
        submitted = st.form_submit_button("Generate readmission risk")

    payload = {
        "patient_name": patient_name,
        "age": int(age),
        "gender": gender,
        "admission_type": admission_type,
        "primary_diagnosis_code": primary_diagnosis_code,
        "num_prior_admissions": int(num_prior_admissions),
        "time_in_hospital": int(time_in_hospital),
        "num_lab_procedures": int(num_lab_procedures),
        "num_medications": int(num_medications),
        "has_comorbidity": int(has_comorbidity),
        "discharge_disposition": discharge_disposition,
        "insurance_type": insurance_type,
        "hospital_id": int(hospital_id),
    }

    return submitted, payload


try:
    pipeline, df = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

profile_options = {meta["name"]: meta["data"] for meta in PRESET_PROFILES.values()}
selected_profile = st.sidebar.selectbox("Preset patient profile", ["Custom"] + list(profile_options.keys()))
default_patient = profile_options.get(selected_profile, DEFAULT_PATIENT)

st.sidebar.markdown("### Core Data Science Skills")
st.sidebar.markdown("- Python: pandas, NumPy, matplotlib, seaborn")
st.sidebar.markdown("- ML: regression, classification, clustering")
st.sidebar.markdown("- Metrics: accuracy, precision, recall, AUC")
st.sidebar.markdown("- scikit-learn workflow and feature selection")
st.sidebar.markdown("- Kaggle-style problem anatomy and solution writing")

st.markdown(
    """
<div class="hero">
    <h1>Data Science Portfolio | HealthGuard AI</h1>
    <p>Streamlit-only project for demonstrating Python for data, scikit-learn, model evaluation, and a full end-to-end machine learning workflow.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="card"><div class="section-title">Dataset</div><div style="font-size:1.7rem;font-weight:800;">{len(df):,}</div><div>patient records</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="card"><div class="section-title">Readmission rate</div><div style="font-size:1.7rem;font-weight:800;">{df["readmitted_within_30days"].mean() * 100:.1f}%</div><div>target prevalence</div></div>', unsafe_allow_html=True)
with col3:
    xgb_metrics = pipeline.metrics.get("xgboost", {})
    st.markdown(f'<div class="card"><div class="section-title">AUC-ROC</div><div style="font-size:1.7rem;font-weight:800;">{xgb_metrics.get("auc_roc", 0):.3f}</div><div>production model</div></div>', unsafe_allow_html=True)
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
