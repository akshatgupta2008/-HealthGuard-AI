# HealthGuard AI - 4-Algorithm ML Healthcare Readmission Pipeline 🏥🤖

An end-to-end Data Science and Machine Learning project combining Unsupervised Patient Segmentation (**K-Means**), Pure Recursive Feature Elimination (**RFE**), Interpretable Baseline Modeling (**Logistic/Ridge Regression**), and High-Precision Predictive Engines (**XGBoost**) for 30-day hospital readmission risk prediction ($AUC\text{-}ROC = 0.6672$).

---

## 📌 Project Overview & Data Science Objectives

Medical readmission datasets contain complex, noisy EHR variables (history, lab procedures, comorbidities, insurance). **HealthGuard AI** addresses this by engineering a domain-constrained **4-Algorithm Machine Learning Pipeline**:

```
[ Patient Clinical EHR Dataset (10,000 Patient Records, 15 Features) ]
                 │
                 ▼
┌──────────────────────────────────────────┐
│  Stage 1: K-Means Clustering (k=4)       │  <-- Unsupervised Patient Segmentation
│  Appends "Patient Segment" Feature       │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│  Stage 2: Recursive Feature Elimination   │  <-- Pure Recursion (RFE)
│  Prunes noise, isolates top 15 features  │
└────────────────┬─────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌────────────────┐  ┌────────────────┐
│ Stage 3: Ridge │  │ Stage 4:       │  <-- Dual Predictive Modeling
│ Logistic Regr. │  │ XGBoost Engine │
│ Baseline &     │  │ High Precision │
│ Coefficients   │  │ Gain Import.   │
└───────┬────────┘  └───────┬────────┘
        │                   │
        └─────────┬─────────┘
                  ▼
┌──────────────────────────────────────────┐
│  Weighted Ensemble Readmission Score     │  <-- Interactive Clinical Dashboard &
│  & Feature Weight Explainability (XAI)   │      Intervention Protocol
└──────────────────────────────────────────┘
```

---

## 📊 Model Benchmark & Performance Metrics

Evaluated on **10,000 Patient Records**:

| Model Paradigm | Accuracy | AUC-ROC | Precision | Recall | Primary Strengths |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | `72.93%` | `0.5821` | Baseline | Baseline | Linear log-odds symptom risk coefficients |
| **XGBoost Ensemble Engine (Production)** | **`73.53%`** | **`0.6672`** | **`0.8333`** | Elevated | Non-linear interaction modeling & Gain importances |

---

## 🔑 Production Feature Weight Distribution (Gain)

Top feature importance drivers identified by production XGBoost Engine:
* **Has Comorbidity** (`22.38%`): Clinical chronic comorbidity indicator.
* **Length of Stay** (`6.32%`): Total days in hospital.
* **Patient Segment Cluster #1** (`6.11%`): Acute Emergency High-Procedure cohort persona.
* **Insurance Type (Medicare)** (`6.09%`): Medicare primary insurance tier.
* **Patient Segment Cluster #0** (`5.93%`): High-Risk Elderly Comorbid cohort persona.

---

## 📁 Repository Layout

```
Carehub2.0/
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
│   ├── model.py                  # FourAlgorithmPipeline class & joblib serialization
│   ├── train_and_evaluate.py     # Benchmark execution script
│   ├── predict.py                # Interactive CLI risk predictor
│   ├── api.py                    # FastAPI serving microservice
│   └── model_pipeline.joblib     # Serialized production model
│
├── tests/                        # Automated PyTest / Unittest Suite
│   └── run_tests.py              # Automated test suite (100% passing)
│
├── docs/                         # Technical Documentation & Interview Preparation
│   ├── healthguard_solution_anatomy.md # Problem statement, business context & architecture
│   └── ds_interview_cheat_sheet.md      # Resume bullet points, pitches & technical Q&A
│
├── requirements.txt              # Python ML dependencies
└── README.md
```

---

## ⚡ FastAPI Serving Microservice Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /health` | `GET` | Returns ML microservice online status and pipeline configuration. |
| `GET /metrics` | `GET` | Exposes 4-algorithm metrics, cluster centroids, RFE trace, and coefficients. |
| `GET /pipeline-details` | `GET` | Returns structured 4-algorithm architecture metadata. |
| `POST /predict` | `POST` | Exposes real-time single-patient risk evaluation and explainable risk weights. |
| `POST /train` | `POST` | Triggers background retraining of the 4-algorithm pipeline. |

---

## 🚀 Quickstart Guide

### 1. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Test Suite:
```bash
python tests/run_tests.py
```

### 3. Run Data Science Benchmark Script:
```bash
python src/train_and_evaluate.py
```

### 4. Run Interactive Terminal Predictor CLI:
```bash
python src/predict.py
```

### 5. Launch FastAPI Microservice Server:
```bash
python src/api.py
```
* Interactive API Documentation will be live at: `http://127.0.0.1:8000/docs`

---

## 🐳 Docker / Full-Stack Launch
```cmd
.\start_app.bat
```
* Interactive Clinical Dashboard live at: `http://localhost:5173/`

---

## 📝 License
Distributed under the MIT License.
