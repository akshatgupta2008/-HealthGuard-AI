import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing import load_and_preprocess_raw_data
from src.model import FourAlgorithmPipeline, DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, CLUSTER_PERSONAS
from src.predict import PRESET_PROFILES

# Page Configuration
st.set_page_config(
    page_title="HealthGuard AI - Clinical CareHub 2.0",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic CSS Theme
st.markdown("""
<style>
    /* Main Background & Fonts */
    .main {
        background-color: #0e1117;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Container */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 16px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Pipeline Step Box */
    .step-box {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .step-title {
        font-weight: 700;
        color: #e2e8f0;
        font-size: 1.05rem;
    }
    .step-desc {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 4px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load dataset and pipeline model safely
@st.cache_resource
def load_pipeline_and_data():
    if not DEFAULT_DATASET_PATH.exists():
        st.error(f"Dataset not found at {DEFAULT_DATASET_PATH}")
        st.stop()
        
    df_raw = pd.read_csv(DEFAULT_DATASET_PATH)
    
    if DEFAULT_MODEL_PATH.exists():
        try:
            pipeline = FourAlgorithmPipeline.load(str(DEFAULT_MODEL_PATH))
            return pipeline, df_raw
        except Exception as e:
            st.warning(f"Re-training pipeline due to load warning: {e}")
            
    pipeline = FourAlgorithmPipeline(dataset_path=str(DEFAULT_DATASET_PATH))
    pipeline.train(df_raw)
    pipeline.save(str(DEFAULT_MODEL_PATH))
    return pipeline, df_raw

pipeline, df_raw = load_pipeline_and_data()

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🏥 HealthGuard AI: Clinical CareHub 2.0</div>
    <div class="hero-subtitle">
        4-Algorithm Machine Learning Pipeline for 30-Day Hospital Readmission Risk Prediction & Patient Segmentation
    </div>
    <div style="display: flex; gap: 12px; flex-wrap: wrap;">
        <span style="background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid #3b82f6; padding: 4px 12px; border-radius: 16px; font-size: 0.82rem; font-weight: 600;">
            ✓ ML Pipeline: Active & Online
        </span>
        <span style="background: rgba(168, 85, 247, 0.2); color: #d8b4fe; border: 1px solid #a855f7; padding: 4px 12px; border-radius: 16px; font-size: 0.82rem; font-weight: 600;">
            📊 Dataset: 10,000 Patient Records
        </span>
        <span style="background: rgba(236, 72, 153, 0.2); color: #fbcfe8; border: 1px solid #ec4899; padding: 4px 12px; border-radius: 16px; font-size: 0.82rem; font-weight: 600;">
            🎯 Model AUC-ROC: 0.6672
        </span>
        <span style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid #10b981; padding: 4px 12px; border-radius: 16px; font-size: 0.82rem; font-weight: 600;">
            🤖 XGBoost Production Engine
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls & Information
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/hospital.png", width=64)
    st.title("CareHub Control Center")
    
    st.markdown("---")
    st.subheader("⚙️ Ensemble Weights")
    weight_xgb = st.slider("XGBoost Weight", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    weight_log = st.slider("Logistic Regression Weight", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    
    st.markdown("---")
    st.subheader("💡 Preset Patient Profiles")
    preset_choice = st.selectbox(
        "Quick Evaluate Patient",
        ["Select a preset...", "Eleanor Vance (High Risk)", "Arthur Pendelton (Moderate Risk)", "Sophia Martinez (Low Risk)"]
    )
    
    st.markdown("---")
    st.markdown("### 📌 Pipeline Specs")
    st.markdown("- **Stage 1**: K-Means ($k=4$)")
    st.markdown("- **Stage 2**: Pure RFE ($n=15$)")
    st.markdown("- **Stage 3**: Ridge Logistic Baseline")
    st.markdown("- **Stage 4**: XGBoost Engine")

# Main Navigation Tabs
tab_landing, tab_predict, tab_segmentation, tab_analytics, tab_cohort = st.tabs([
    "🏥 Executive Overview",
    "🔮 Risk Predictor",
    "📊 Patient Segmentation",
    "⚡ Model Analytics",
    "📁 Patient Cohort Explorer"
])

# ==========================================
# TAB 1: EXECUTIVE OVERVIEW & LANDING PAGE
# ==========================================
with tab_landing:
    st.markdown("### 📌 Executive Overview & Core Metrics")
    
    # Top KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">10,000</div>
            <div class="metric-lbl">Total EHR Patient Cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val" style="color: #60a5fa;">0.6672</div>
            <div class="metric-lbl">Production AUC-ROC</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val" style="color: #34d399;">73.53%</div>
            <div class="metric-lbl">Ensemble Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val" style="color: #f472b6;">22.38%</div>
            <div class="metric-lbl">Top Feature Driver (Comorbidity)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4-Algorithm Pipeline Visual Architecture
    st.markdown("### ⚙️ 4-Algorithm Sequential ML Architecture")
    
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("""
        <div class="step-box" style="border-left-color: #3b82f6;">
            <div class="step-title">Stage 1: Unsupervised K-Means Patient Segmentation (k=4)</div>
            <div class="step-desc">Discovers latent clinical personas across demographic & EHR parameters, injecting 'Patient Segment' features directly into downstream training.</div>
        </div>
        <div class="step-box" style="border-left-color: #8b5cf6;">
            <div class="step-title">Stage 2: Pure Recursive Feature Elimination (RFE)</div>
            <div class="step-desc">Prunes noisy predictors step-by-step to extract the top 15 high-signal features for predictive engines.</div>
        </div>
        <div class="step-box" style="border-left-color: #ec4899;">
            <div class="step-title">Stage 3: Ridge Logistic Regression Baseline</div>
            <div class="step-desc">Provides linear log-odds interpretable risk baseline & symptom coefficient contributions for clinical compliance.</div>
        </div>
        <div class="step-box" style="border-left-color: #10b981;">
            <div class="step-title">Stage 4: High-Precision XGBoost Ensemble Engine</div>
            <div class="step-desc">Models complex non-linear feature interactions and gain importances for high-accuracy 30-day readmission scoring.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("#### 📊 Model Benchmark Comparison")
        metrics_df = pd.DataFrame([
            {"Model Paradigm": "Logistic Regression (Baseline)", "Accuracy": "72.93%", "AUC-ROC": "0.5821", "Precision": "0.0000", "Recall": "0.0000"},
            {"Model Paradigm": "XGBoost Engine (Production)", "Accuracy": "73.53%", "AUC-ROC": "0.6672", "Precision": "0.8333", "Recall": "0.0277"}
        ])
        st.table(metrics_df)
        
        st.info("💡 **Key Finding**: The XGBoost Ensemble model delivers an **14.6% improvement in AUC-ROC** over linear baselines by capturing multi-morbidity interactions.")

# ==========================================
# TAB 2: LIVE PATIENT RISK PREDICTOR
# ==========================================
with tab_predict:
    st.markdown("### 🔮 Interactive Clinical Patient Readmission Calculator")
    st.caption("Adjust patient demographics and clinical vitals to run real-time inference through the 4-algorithm ML pipeline.")
    
    # Check if sidebar preset was chosen
    default_vals = PRESET_PROFILES["1"]["data"]
    if preset_choice == "Eleanor Vance (High Risk)":
        default_vals = PRESET_PROFILES["1"]["data"]
    elif preset_choice == "Arthur Pendelton (Moderate Risk)":
        default_vals = PRESET_PROFILES["2"]["data"]
    elif preset_choice == "Sophia Martinez (Low Risk)":
        default_vals = PRESET_PROFILES["3"]["data"]
        
    col_input, col_result = st.columns([1, 1.1])
    
    with col_input:
        st.markdown("#### 👤 Patient Profile Inputs")
        
        p_name = st.text_input("Patient Name", value=default_vals.get("patient_name", "Eleanor Vance"))
        
        c_i1, c_i2 = st.columns(2)
        with c_i1:
            age = st.slider("Age (Years)", 18, 100, int(default_vals.get("age", 65)))
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if default_vals.get("gender") == "Female" else 1)
            insurance = st.selectbox("Insurance Type", ["Medicare", "Private", "Medicaid"], index=["Medicare", "Private", "Medicaid"].index(default_vals.get("insurance_type", "Medicare")))
        with c_i2:
            admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"], index=["Emergency", "Elective", "Urgent"].index(default_vals.get("admission_type", "Emergency")))
            primary_diag = st.selectbox("Primary Diagnosis Code", ["I10", "E11", "J45", "M54"], index=["I10", "E11", "J45", "M54"].index(default_vals.get("primary_diagnosis_code", "I10")))
            discharge = st.selectbox("Discharge Disposition", ["Home", "Transfer", "SNF"], index=["Home", "Transfer", "SNF"].index(default_vals.get("discharge_disposition", "Transfer") if default_vals.get("discharge_disposition") in ["Home", "Transfer", "SNF"] else 1))

        st.markdown("#### 🩺 Clinical Vitals & Trajectory")
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, int(default_vals.get("time_in_hospital", 5)))
            num_prior = st.slider("Prior Admissions (Past Year)", 0, 15, int(default_vals.get("num_prior_admissions", 2)))
        with c_v2:
            num_labs = st.slider("Lab Procedures", 1, 100, int(default_vals.get("num_lab_procedures", 40)))
            num_meds = st.slider("Medications Prescribed", 1, 50, int(default_vals.get("num_medications", 15)))
            
        has_comorb = st.radio("Has Chronic Comorbidities?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", index=0 if default_vals.get("has_comorbidity", 1) == 1 else 1)

        patient_payload = {
            "patient_name": p_name,
            "age": age,
            "gender": gender,
            "admission_type": admission_type,
            "primary_diagnosis_code": primary_diag,
            "num_prior_admissions": num_prior,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": float(num_labs),
            "num_medications": float(num_meds),
            "has_comorbidity": has_comorb,
            "discharge_disposition": discharge,
            "insurance_type": insurance,
            "hospital_id": 101
        }
        
    with col_result:
        st.markdown("#### 📊 Real-Time ML Pipeline Assessment")
        
        # Run prediction
        res = pipeline.predict_patient(patient_payload, weight_logistic=weight_log, weight_xgb=weight_xgb)
        ensemble = res["ensemble_result"]
        stages = res["pipeline_stages"]
        
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ensemble["readmission_risk_percentage"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "30-Day Readmission Risk Score", 'font': {'size': 18, 'color': "#e2e8f0"}},
            number={'suffix': "%", 'font': {'size': 36, 'color': ensemble['badge_color']}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': ensemble['badge_color']},
                'bgcolor': "rgba(15, 23, 42, 0.8)",
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.2)'},
                    {'range': [25, 45], 'color': 'rgba(234, 179, 8, 0.2)'},
                    {'range': [45, 70], 'color': 'rgba(249, 115, 22, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                ]
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Risk Badge & Recommendation Box
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid {ensemble['badge_color']}; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-weight: 600;">Assigned Risk Classification:</span>
                <span style="background-color: {ensemble['badge_color']}20; color: {ensemble['badge_color']}; border: 1px solid {ensemble['badge_color']}; padding: 4px 14px; border-radius: 20px; font-weight: 700;">
                    {ensemble['risk_tier']}
                </span>
            </div>
            <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.5; margin-top: 8px;">
                <strong>📋 Protocol:</strong> {ensemble['recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stage 1 Persona Badge
        p_persona = stages["stage1_kmeans"]["persona"]
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border-left: 4px solid #60a5fa; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
            <div style="color: #93c5fd; font-weight: 700; font-size: 0.9rem;">Stage 1 Patient Persona: {p_persona['name']}</div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 2px;">{p_persona['description']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Top Drivers Tabs
        st.markdown("#### 💡 Primary Risk Drivers")
        t_xgb, t_log = st.tabs(["XGBoost Gain Importances", "Logistic Coefficients"])
        
        with t_xgb:
            top_xgb = pd.DataFrame(stages["stage4_xgboost"]["top_importance_drivers"][:6])
            top_xgb["importance_pct"] = top_xgb["importance"] * 100
            fig_xgb = px.bar(
                top_xgb,
                x="importance_pct",
                y="feature",
                orientation='h',
                labels={'importance_pct': 'Weight Gain (%)', 'feature': 'Feature Name'},
                color="importance_pct",
                color_continuous_scale="Blues"
            )
            fig_xgb.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_xgb, use_container_width=True)
            
        with t_log:
            top_log = pd.DataFrame(stages["stage3_logistic_regression"]["top_feature_contributions"][:6])
            fig_log = px.bar(
                top_log,
                x="contribution",
                y="feature",
                orientation='h',
                color="impact",
                color_discrete_map={"Increases Risk": "#ef4444", "Decreases Risk": "#10b981"},
                labels={'contribution': 'Log-Odds Contribution', 'feature': 'Feature Name'}
            )
            fig_log.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_log, use_container_width=True)

# ==========================================
# TAB 3: PATIENT SEGMENTATION (K-MEANS)
# ==========================================
with tab_segmentation:
    st.markdown("### 📊 Unsupervised Patient Segmentation (K-Means Clustering)")
    st.caption("Identifies 4 clinical personas to capture risk patterns across length-of-stay, prior admissions, and age.")
    
    # 4 Persona Cards
    st.markdown("#### 🏷️ Patient Cluster Personas")
    col_p0, col_p1, col_p2, col_p3 = st.columns(4)
    
    personas = [
        (col_p0, CLUSTER_PERSONAS[0], "#ef4444"),
        (col_p1, CLUSTER_PERSONAS[1], "#f97316"),
        (col_p2, CLUSTER_PERSONAS[2], "#eab308"),
        (col_p3, CLUSTER_PERSONAS[3], "#10b981")
    ]
    
    for col, p, color in personas:
        with col:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.6); border-top: 4px solid {color}; border-radius: 8px; padding: 14px; height: 180px;">
                <div style="color: {color}; font-weight: 700; font-size: 0.95rem;">{p['name']}</div>
                <span style="font-size: 0.75rem; background: {color}20; color: {color}; padding: 2px 8px; border-radius: 10px; font-weight: 600;">{p['badge']}</span>
                <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 8px;">{p['description']}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Cluster Visualization Plotly
    c_scat, c_heat = st.columns([1.2, 1])
    
    # Run K-means prediction on cleaned raw dataset for scatter
    X_clean, _ = load_and_preprocess_raw_data(df_raw)
    X_proc = pipeline.preprocessor.transform(X_clean)
    clusters = pipeline.kmeans_model.predict(X_proc)
    df_scatter = df_raw.copy()
    df_scatter['Cluster'] = [f"Cluster #{c}: {CLUSTER_PERSONAS[c]['name']}" for c in clusters]
    
    with c_scat:
        st.markdown("#### 📌 3D Cluster Distribution (Stay vs Admissions vs Age)")
        fig_3d = px.scatter_3d(
            df_scatter.head(1500),
            x='time_in_hospital',
            y='num_prior_admissions',
            z='age',
            color='Cluster',
            opacity=0.7,
            size_max=8,
            height=440
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with c_heat:
        st.markdown("#### 📈 Cluster Centroid Comparison")
        centroids_list = []
        for cid, stats in pipeline.cluster_centroids.items():
            centroids_list.append({
                "Cluster": f"Cluster #{cid}",
                "Avg Stay (Days)": stats.get("avg_hospital_days", 0),
                "Prior Admissions": stats.get("avg_prior_admissions", 0),
                "Comorbidity Rate (%)": stats.get("comorbidity_rate", 0) * 100,
                "Readmission Rate (%)": stats.get("readmission_rate", 0) * 100
            })
        df_cent = pd.DataFrame(centroids_list)
        
        fig_cent = px.bar(
            df_cent,
            x="Cluster",
            y=["Avg Stay (Days)", "Prior Admissions", "Readmission Rate (%)"],
            barmode="group",
            height=400
        )
        fig_cent.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cent, use_container_width=True)

# ==========================================
# TAB 4: RFE TRACE & MODEL ANALYTICS
# ==========================================
with tab_analytics:
    st.markdown("### ⚡ RFE Trace & Feature Selection Analytics")
    
    col_rfe, col_imp = st.columns([1, 1])
    
    with col_rfe:
        st.markdown("#### 📉 RFE Elimination Step Accuracy Trace")
        rfe_hist_df = pd.DataFrame(pipeline.rfe_history)
        if not rfe_hist_df.empty:
            fig_rfe = px.line(
                rfe_hist_df,
                x="step",
                y="step_accuracy",
                markers=True,
                labels={"step": "RFE Elimination Step", "step_accuracy": "Step Accuracy"},
                title="Model Accuracy Trajectory During Feature Elimination"
            )
            fig_rfe.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=340)
            st.plotly_chart(fig_rfe, use_container_width=True)
        else:
            st.info("RFE trace data ready.")
            
    with col_imp:
        st.markdown("#### 🏆 All 15 Selected Features (XGBoost Gain)")
        xgb_all_imp = pd.DataFrame([
            {"Feature": name, "Importance": imp * 100}
            for name, imp in zip(pipeline.rfe_selected_feature_names, pipeline.xgboost_model.feature_importances_)
        ]).sort_values(by="Importance", ascending=True)
        
        fig_all_xgb = px.bar(
            xgb_all_imp,
            x="Importance",
            y="Feature",
            orientation='h',
            labels={"Importance": "Gain Importance (%)"},
            color="Importance",
            color_continuous_scale="Purples"
        )
        fig_all_xgb.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=340)
        st.plotly_chart(fig_all_xgb, use_container_width=True)

# ==========================================
# TAB 5: PATIENT COHORT EXPLORER & INTELLIGENCE CENTER
# ==========================================
with tab_cohort:
    st.markdown("### 📁 Comprehensive Patient Cohort Explorer & Clinical Intelligence Center")
    st.caption("Filter, search, batch-score, and perform deep-dive clinical analysis across the 10,000 record patient EHR dataset.")
    
    # Filter Bar Controls
    with st.expander("🔍 Advanced Cohort Filtering Controls", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            search_query = st.text_input("🔎 Search Patient Name / ID", placeholder="e.g. Eleanor or 101")
            f_adm = st.multiselect("Admission Type", df_raw['admission_type'].unique(), default=list(df_raw['admission_type'].unique()))
        with f_col2:
            f_diag = st.multiselect("Diagnosis Code", df_raw['primary_diagnosis_code'].unique(), default=list(df_raw['primary_diagnosis_code'].unique()))
            f_ins = st.multiselect("Insurance Type", df_raw['insurance_type'].unique(), default=list(df_raw['insurance_type'].unique()))
        with f_col3:
            age_range = st.slider("Age Range", int(df_raw['age'].min()), int(df_raw['age'].max()), (int(df_raw['age'].min()), int(df_raw['age'].max())))
            stay_range = st.slider("Hospital Stay (Days)", int(df_raw['time_in_hospital'].min()), int(df_raw['time_in_hospital'].max()), (int(df_raw['time_in_hospital'].min()), int(df_raw['time_in_hospital'].max())))
        with f_col4:
            f_comorb = st.radio("Comorbidity Filter", ["All", "Comorbid Only", "Non-Comorbid"], index=0)
            f_disch = st.multiselect("Discharge Disposition", df_raw['discharge_disposition'].unique(), default=list(df_raw['discharge_disposition'].unique()))

    # Apply Filters
    df_filtered = df_raw.copy()
    
    if search_query:
        search_lower = search_query.lower()
        df_filtered = df_filtered[
            df_filtered['patient_name'].astype(str).str.lower().str.contains(search_lower) |
            df_filtered['patient_id'].astype(str).str.lower().str.contains(search_lower)
        ]
        
    df_filtered = df_filtered[
        (df_filtered['admission_type'].isin(f_adm)) &
        (df_filtered['primary_diagnosis_code'].isin(f_diag)) &
        (df_filtered['insurance_type'].isin(f_ins)) &
        (df_filtered['discharge_disposition'].isin(f_disch)) &
        (df_filtered['age'] >= age_range[0]) & (df_filtered['age'] <= age_range[1]) &
        (df_filtered['time_in_hospital'] >= stay_range[0]) & (df_filtered['time_in_hospital'] <= stay_range[1])
    ]
    
    if f_comorb == "Comorbid Only":
        df_filtered = df_filtered[df_filtered['has_comorbidity'] == 1]
    elif f_comorb == "Non-Comorbid":
        df_filtered = df_filtered[df_filtered['has_comorbidity'] == 0]
        
    # Cohort Summary Statistics Cards
    st.markdown("#### 📊 Selected Cohort Overview")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{len(df_filtered):,}</div>
            <div class="metric-lbl">Matching Patients</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        avg_stay = df_filtered['time_in_hospital'].mean() if not df_filtered.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #60a5fa;">{avg_stay:.1f} Days</div>
            <div class="metric-lbl">Avg Hospital Stay</div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        com_rate = (df_filtered['has_comorbidity'].mean() * 100) if not df_filtered.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #f472b6;">{com_rate:.1f}%</div>
            <div class="metric-lbl">Comorbidity Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        readm_rate = (df_filtered['readmitted_within_30days'].mean() * 100) if ('readmitted_within_30days' in df_filtered and not df_filtered.empty) else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #34d399;">{readm_rate:.1f}%</div>
            <div class="metric-lbl">Historical Readmission Rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Cohort Data Table with ML Scores Batch Scoring Button
    st.markdown("#### 📋 Cohort Data & Batch ML Risk Evaluation")
    
    batch_score = st.checkbox("⚡ Compute Live 4-Algorithm ML Risk Scores & Personas for Cohort", value=True)
    
    if batch_score and not df_filtered.empty:
        with st.spinner("Scoring patient cohort through ML pipeline..."):
            # Sample max 500 for fast UI response if larger
            scoring_subset = df_filtered.head(500).copy()
            
            scores = []
            tiers = []
            personas = []
            
            for _, row in scoring_subset.iterrows():
                p_dict = row.to_dict()
                eval_res = pipeline.predict_patient(p_dict, weight_logistic=weight_log, weight_xgb=weight_xgb)
                scores.append(eval_res["ensemble_result"]["readmission_risk_percentage"])
                tiers.append(eval_res["ensemble_result"]["risk_tier"])
                personas.append(eval_res["pipeline_stages"]["stage1_kmeans"]["persona"]["name"])
                
            scoring_subset["ML Risk Score (%)"] = scores
            scoring_subset["Assigned Risk Tier"] = tiers
            scoring_subset["Patient Segment Persona"] = personas
            
            display_df = scoring_subset
    else:
        display_df = df_filtered
        
    st.dataframe(display_df, use_container_width=True, height=360)
    
    # Export Button
    csv_export = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Cohort Data with ML Scores to CSV",
        data=csv_export,
        file_name="healthguard_cohort_ml_scored.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Patient Deep-Dive Inspector
    if not display_df.empty:
        st.markdown("### 🩺 Individual Patient Clinical Deep-Dive")
        patient_names = display_df['patient_name'].tolist() if 'patient_name' in display_df.columns else [f"Patient #{i}" for i in display_df['patient_id']]
        selected_patient_name = st.selectbox("Select Patient to Inspect", patient_names)
        
        # Get patient row
        p_row = display_df[display_df['patient_name'] == selected_patient_name].iloc[0] if 'patient_name' in display_df.columns else display_df.iloc[0]
        p_dict = p_row.to_dict()
        
        inspect_res = pipeline.predict_patient(p_dict, weight_logistic=weight_log, weight_xgb=weight_xgb)
        ins_ens = inspect_res["ensemble_result"]
        ins_stg = inspect_res["pipeline_stages"]
        
        pi1, pi2 = st.columns([1, 1.2])
        with pi1:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="font-size: 1.3rem; font-weight: 800; color: #f8fafc;">{p_dict.get('patient_name', 'Patient Dossier')}</div>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
                    {p_dict['age']} y/o {p_dict['gender']} • Admission: {p_dict['admission_type']} ({p_dict['primary_diagnosis_code']})
                </div>
                <hr style="border-color: rgba(255,255,255,0.1); margin: 12px 0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.88rem;">
                    <div><strong>Hospital Stay:</strong> {p_dict['time_in_hospital']} days</div>
                    <div><strong>Prior Admissions:</strong> {p_dict['num_prior_admissions']}</div>
                    <div><strong>Lab Procedures:</strong> {p_dict['num_lab_procedures']}</div>
                    <div><strong>Medications:</strong> {p_dict['num_medications']}</div>
                    <div><strong>Comorbidities:</strong> {'Yes' if p_dict['has_comorbidity'] else 'No'}</div>
                    <div><strong>Insurance:</strong> {p_dict['insurance_type']}</div>
                </div>
                <hr style="border-color: rgba(255,255,255,0.1); margin: 12px 0;">
                <div style="background: rgba(15, 23, 42, 0.8); padding: 12px; border-radius: 8px; border-left: 4px solid {ins_ens['badge_color']};">
                    <div style="font-weight: 700; color: {ins_ens['badge_color']}; font-size: 1.1rem;">
                        Readmission Score: {ins_ens['readmission_risk_percentage']}% ({ins_ens['risk_tier']})
                    </div>
                    <div style="color: #e2e8f0; font-size: 0.85rem; margin-top: 4px;">
                        <strong>Protocol:</strong> {ins_ens['recommendation']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with pi2:
            st.markdown("#### 💡 Patient Symptom & Risk Contribution Breakdown")
            ins_drivers = pd.DataFrame(ins_stg["stage4_xgboost"]["top_importance_drivers"][:6])
            ins_drivers["imp_pct"] = ins_drivers["importance"] * 100
            
            fig_ins = px.bar(
                ins_drivers,
                x="imp_pct",
                y="feature",
                orientation='h',
                labels={'imp_pct': 'Risk Driver Weight (%)', 'feature': 'Predictor'},
                title=f"Top Risk Drivers for {p_dict.get('patient_name', 'Patient')}",
                color="imp_pct",
                color_continuous_scale="Purples"
            )
            fig_ins.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_ins, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "HealthGuard AI CareHub 2.0 • Powered by K-Means, RFE, Ridge Logistic Regression & XGBoost • Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
