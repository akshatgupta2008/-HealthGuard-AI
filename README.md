# 🏥 HealthGuard AI: 4-Algorithm Clinical Readmission Engine & Data Science Application

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![ML Engine](https://img.shields.io/badge/ML--Engine-XGBoost%20%7C%20K--Means%20%7C%20RFE-purple.svg)
![UI](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-Passing%20100%25-brightgreen.svg)

An end-to-end Data Science and Machine Learning project combining **Unsupervised Patient Segmentation (K-Means)**, **Pure Recursive Feature Elimination (RFE)**, **Interpretable Baseline Modeling (Ridge Logistic Regression)**, and **High-Precision Predictive Engines (XGBoost)** for 30-day hospital readmission risk prediction ($AUC\text{-}ROC = 0.6672$) on 10,000 Patient Electronic Health Records (EHR).

---

## 📌 Executive Summary & Business Impact

Unplanned 30-day hospital readmissions impose multi-billion dollar costs on healthcare systems and trigger substantial penalties under CMS Hospital Readmission Reduction Programs (HRRP). **HealthGuard AI** addresses this challenge by deploying a domain-constrained **4-Algorithm Machine Learning Pipeline** that combines linear explainability (log-odds risk coefficients) with high-precision non-linear ensemble models (XGBoost Gain importances).

```
[ 10,000 Patient Clinical EHR Dataset (15 Features) ]
                         │
                         ▼
┌───────────────────────────────────────────────────┐
│ Stage 1: Unsupervised K-Means Clustering (k=4)    │  <-- Patient Segmentation & Persona Discovery
│ Appends latent "Patient Segment" features         │
└────────────────────────┬──────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────┐
│ Stage 2: Recursive Feature Elimination (RFE)      │  <-- Feature Selection & Noise Pruning
│ Isolates top 15 highest-signal predictors         │
└────────────────────────┬──────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌───────────────────────┐       ┌───────────────────────┐
│ Stage 3: Ridge        │       │ Stage 4: High-        │  <-- Dual Predictive Modeling
│ Logistic Baseline     │       │ Precision XGBoost     │
│ Log-Odds Coefficients │       │ Gain Importances      │
└───────────┬───────────┘       └───────────┬───────────┘
            │                               │
            └────────────────┬──────────────┘
                             ▼
┌───────────────────────────────────────────────────┐
│ Weighted Ensemble Readmission Score & XAI Drivers │  <-- Streamlit Clinical Intelligence App
│ Real-Time Patient Risk Calculator & Cohort Viewer │      (app.py)
└───────────────────────────────────────────────────┘
```

---

## 📊 Model Performance Benchmark

Evaluated on **10,000 Patient Records**:

| Model Paradigm | Accuracy | AUC-ROC | Precision | Recall | Primary Analytical Strengths |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | `72.93%` | `0.5821` | `0.0000` | `0.0000` | Linear log-odds symptom risk coefficients for clinical auditability. |
| **XGBoost Ensemble (Production)** | **`73.53%`** | **`0.6672`** | **`0.8333`** | **`0.0277`** | **$+14.6\%$ AUC-ROC boost** by modeling multi-morbidity non-linear feature interactions. |

---

## 🔑 Production Feature Weight Distribution (XGBoost Gain)

Top predictive feature weights identified by the production XGBoost Engine:

1. **Has Comorbidity** (`22.38%`): Primary clinical chronic comorbidity indicator.
2. **Length of Hospital Stay** (`6.32%`): Inpatient stay duration in days.
3. **Patient Segment Cluster #1** (`6.11%`): Acute Emergency High-Procedure cohort persona.
4. **Insurance Type (Medicare)** (`6.09%`): Primary Medicare coverage tier.
5. **Patient Segment Cluster #0** (`5.93%`): High-Risk Elderly Comorbid cohort persona.
6. **Admission Type (Emergency)** (`5.84%`): Emergency admission channel.
7. **Primary Diagnosis (J45 Respiratory)** (`5.72%`): Asthma/Respiratory primary diagnosis code.

---

## 🖥️ Streamlit Data Science & Clinical Application

The application (`app.py`) provides an interactive Data Science dashboard built with Streamlit and Plotly:

- **🏥 Executive Overview**: Hero KPI metrics, visual 4-Algorithm Pipeline flow diagram, model benchmark comparison table, and quick preset patient evaluators.
- **🔮 Real-Time Risk Predictor**: Interactive calculator allowing clinicians to adjust patient age, stay, admissions, lab tests, and comorbidities to compute ensemble risk scores, risk tier badges (`Critical`, `High`, `Moderate`, `Low`), and clinical intervention recommendations.
- **📊 K-Means Patient Segmentation**: Unsupervised cluster persona analysis, interactive Plotly 3D scatter plots (`time_in_hospital` vs `num_prior_admissions` vs `age`), and centroid bar charts.
- **⚡ Feature Selection Analytics**: Step-by-step RFE feature elimination accuracy trajectory plot and full 15-feature XGBoost Gain ranking.
- **📁 Patient Cohort Explorer**: Filterable 10,000-record dataset table with real-time batch ML risk scoring, individual patient inspection cards, and CSV export.

---

## 📁 Repository Layout

```
Carehub2.0/
├── app.py                        # Streamlit Data Science Landing Page & Dashboard
├── data/                         # Patient EHR Dataset
│   └── readmission_dataset.csv   # Raw patient dataset (10,000 records)
│
├── notebooks/                    # Data Science EDA & Model Benchmark Notebooks
│   ├── 01_exploratory_data_analysis.ipynb # EDA, demographic distributions & heatmaps
│   └── 02_model_training_and_eval.ipynb  # 4-algorithm ML benchmark & feature importance plots
│
├── src/                          # Modular Python Machine Learning Package
│   ├── __init__.py
│   ├── data_processing.py        # Data cleaning, imputations & preprocessor
│   ├── model.py                  # FourAlgorithmPipeline class (K-Means + RFE + Logistic + XGBoost)
│   ├── train_and_evaluate.py     # Benchmark execution script
│   └── predict.py                # Interactive CLI risk predictor
│
├── tests/                        # Automated PyTest / Unittest Suite
│   └── run_tests.py              # Automated ML test suite (100% passing)
│
├── docs/                         # Data Science Technical Documentation
│   └── healthguard_solution_anatomy.md # Architecture blueprint & problem anatomy
│
├── requirements.txt              # Python ML & Streamlit dependencies
├── start_app.bat                 # One-click Streamlit launch script
└── README.md
```

---

## 🚀 Quickstart & Execution Guide

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated ML Unit Test Suite
```bash
python tests/run_tests.py
```

### 3. Execute Data Science Benchmark Script
```bash
python src/train_and_evaluate.py
```

### 4. Run Interactive CLI Predictor
```bash
python src/predict.py
```

### 5. Launch Streamlit Data Science Web Application
```bash
streamlit run app.py
```
*Or execute launcher:*
```cmd
.\start_app.bat
```
*Live at: `http://localhost:8501`*

---

## 🗣️ Data Science Interview Pitch & Key Talking Points

* **Problem Formulation**: Framed 30-day hospital readmission as a supervised binary classification task augmented by unsupervised clustering features.
* **Feature Engineering & Leakage Prevention**: Engineered patient cluster features using K-Means ($k=4$) on normalized demographic/clinical variables. Strictly separated target `readmitted_within_30days` and leak-prone `days_to_readmission` columns prior to preprocessing.
* **Dimensionality Reduction**: Used pure Recursive Feature Elimination (RFE) to prune noisy predictors and isolate the top 15 most informative features.
* **Explainability (XAI)**: Combined Ridge Logistic Regression linear coefficients (log-odds impact) with XGBoost Gain importances to provide clinicians with clear risk explanations alongside high-precision predictions.

---

## 📝 License
Distributed under the MIT License.
