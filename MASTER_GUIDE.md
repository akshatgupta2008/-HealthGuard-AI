# HealthGuard AI - Comprehensive A-to-Z Project Master Guide 📚🏥

Welcome to **HealthGuard AI**, an advanced full-stack Machine Learning System designed for clinical 30-day patient hospital readmission risk prediction, feature optimization, and model explainability.

---

## 📋 Table of Contents
1. [Executive Summary & What the Project Does](#1-executive-summary--what-the-project-does)
2. [Project Directory Structure: Where is What?](#2-project-directory-structure-where-is-what)
3. [The 4-Algorithm Pipeline Architecture Deep-Dive](#3-the-4-algorithm-pipeline-architecture-deep-dive)
4. [Microservices & System Communication Flow](#4-microservices--system-communication-flow)
5. [Frontend Dashboard Views & Features](#5-frontend-dashboard-views--features)
6. [API Endpoints Reference](#6-api-endpoints-reference)
7. [How to Run & Operate the System](#7-how-to-run--operate-the-system)

---

## 1. Executive Summary & What the Project Does

### The Problem
Hospital readmissions within 30 days of discharge represent a major quality and cost challenge in healthcare. However, medical datasets contain hundreds of complex, noisy EHR variables (demographics, prior visits, lab values, prescriptions, diagnostic codes) making simple linear models inaccurate and complex black-box models unexplainable to doctors.

### The Solution: HealthGuard AI
HealthGuard AI resolves this by deploying a **sequential 4-algorithm ML architecture**:
1. **Unsupervised Patient Personas (`K-Means`)** group similar patients together.
2. **Pure Programming Recursion (`RFE`)** strips away dataset noise and isolates the top 15 critical predictors.
3. **Interpretable Baseline (`Logistic Regression`)** provides linear symptom weights so clinicians understand why predictions were made.
4. **High-Precision Non-Linear Engine (`XGBoost`)** captures complex symptom interactions.
5. **Dynamic Ensemble Weighting** blends both models into a unified 30-Day Readmission Risk Score and clinical action protocol.

---

## 2. Project Directory Structure: Where is What?

```
Carehub2.0/
├── HealthGuard_Readmission_Data/
│   └── readmission_dataset.csv     <-- Raw Dataset (10,000 Patient Records, 15 Columns)
│
├── ml_engine/                      <-- Python 3.14 ML Microservice
│   ├── pipeline.py                 <-- Core 4-Algorithm Pipeline Class & Logic
│   ├── train_and_save.py           <-- Training Script & Model Persistence Engine
│   ├── main.py                     <-- FastAPI Microservice Server (Port 8000)
│   ├── requirements.txt            <-- Python Dependencies (scikit-learn, xgboost, pandas, etc.)
│   └── model_pipeline.joblib       <-- Serialized Trained Model Weights & Preprocessors
│
├── backend/                        <-- Express Node.js API Gateway
│   ├── server.js                   <-- REST Proxy API, Patient Presets, & Fallback Engine (Port 5000)
│   ├── package.json                <-- Node Dependencies (express, axios, cors)
│   └── package-lock.json
│
├── frontend/                       <-- React 19 + Vite + Tailwind v4 Dashboard
│   ├── src/
│   │   ├── App.jsx                 <-- Main Dashboard UI (5 Tabs, Charts, Controls, Form)
│   │   ├── index.css               <-- Glassmorphism Styling & Design System
│   │   └── main.jsx                <-- React Entry Point
│   ├── index.html                  <-- Google Fonts & Base HTML Template
│   ├── vite.config.js              <-- Vite Configuration with @tailwindcss/vite Plugin
│   ├── package.json                <-- Frontend Dependencies (lucide-react, recharts, tailwindcss)
│   └── dist/                       <-- Production Build Bundle
│
├── package.json                    <-- Root Multi-Service Orchestrator (npm start)
├── start_app.bat                   <-- Windows 1-Click Batch File Launcher
├── .gitignore                      <-- Excludes node_modules, build outputs, & venv
├── README.md                       <-- Main GitHub Documentation
└── MASTER_GUIDE.md                 <-- This Complete A-to-Z Guide
```

---

## 3. The 4-Algorithm Pipeline Architecture Deep-Dive

Because the output of one algorithm feeds into the next, execution order is critical:

```
[ Raw Dataset (10,000 Records) ]
               │
               ▼
┌──────────────────────────────────────────┐
│  Stage 1: K-Means Clustering (k=4)       │  <-- Unsupervised Patient Personas
│  Assigns "Patient Segment ID" Feature    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Stage 2: Recursive Feature Elimination   │  <-- Pure Recursion (RFE)
│  Strips noise, selects top 15 features   │
└──────────────────┬───────────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
┌──────────────────┐  ┌──────────────────┐
│ Stage 3: Ridge / │  │ Stage 4: XGBoost │  <-- Dual Model Training
│ Logistic Regr.   │  │ Non-Linear Engine│
│ Baseline Weights │  │ High Precision   │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
┌──────────────────────────────────────────┐
│  Weighted Ensemble Risk Score            │  <-- Interactive Weight Slider
│  Final Risk Score = w1*Log + w2*XGB      │      Clinician Recommendation
└──────────────────────────────────────────┘
```

### Algorithm 1: K-Means Clustering (Unsupervised Learning)
- **File**: `ml_engine/pipeline.py` (`train()` method)
- **Role**: Groups patients into 4 distinct persona clusters based on vitals, length of stay, prior admissions, and comorbidities.
- **Clusters**:
  - `Segment 0: High Risk Elderly Comorbid` (Older age, extended stay, high comorbidities).
  - `Segment 1: Acute Emergency High-Procedure` (Emergency admission, high lab count).
  - `Segment 2: Moderate Risk Chronic Care` (Urgent visits, baseline comorbidity).
  - `Segment 3: Low Risk Elective Recovery` (Younger elective surgery patients).
- **Output**: Appends `patient_segment_Cluster_0..3` into feature matrix.

### Algorithm 2: Recursive Feature Elimination - RFE (Pure Programming Recursion)
- **File**: `ml_engine/pipeline.py` (`RFE` loop)
- **Role**: Pure recursive elimination loop:
  $$\text{Subset } S_k \xrightarrow{\text{Fit Model}} \text{Weights } w \xrightarrow{\text{Drop Worst } f_i} \text{Subset } S_{k-1} \dots$$
  Recursively drops the single least important feature and retrains until reaching $n=15$ top predictors.
- **Output**: 15 pruned features.

### Algorithm 3: Logistic / Ridge Regression (Baseline Supervised Learning)
- **File**: `ml_engine/pipeline.py` (`LogisticRegression`)
- **Role**: Fits baseline model on the 15 RFE features. Computes exact linear coefficients ($w_i$).
- **Output**: Baseline probability score $P_{\text{Logistic}}$ + interpretable symptom weights.

### Algorithm 4: XGBoost (Advanced Ensemble Engine)
- **File**: `ml_engine/pipeline.py` (`xgb.XGBClassifier`)
- **Role**: Trains 100 gradient-boosted decision trees on the 15 RFE features to capture non-linear symptom interactions.
- **Output**: Non-linear probability score $P_{\text{XGBoost}}$ + feature gain importances.

---

## 4. Microservices & System Communication Flow

```
[ User Browser ]
   │
   ▼ (Port 5173)
┌──────────────────────────────────────────┐
│ React Vite Clinical Dashboard            │
└──────────────────┬───────────────────────┘
                   │ HTTP REST Requests
                   ▼ (Port 5000)
┌──────────────────────────────────────────┐
│ Express Node.js Gateway                  │  <-- Serves presets, caches audit history,
└──────────────────┬───────────────────────┘      runs fallback engine if ML boots.
                   │ Internal Proxy
                   ▼ (Port 8000)
┌──────────────────────────────────────────┐
│ Python FastAPI ML Microservice           │  <-- Loads Joblib pipeline, runs real-time
└──────────────────────────────────────────┘      4-algorithm inference.
```

---

## 5. Frontend Dashboard Views & Features

Access at **`http://localhost:5173/`**:

1. **Live Patient Assessor**:
   - 1-Click Sample Patient Presets (*Eleanor Vance High Risk, Marcus Brody Low Risk, Sofia Rodriguez Moderate Risk, David Kim Acute Emergency*).
   - Organized 3-section input form (*Demographics, Admission & Diagnosis, Hospital Utilization*).
   - Interactive Model Weighting Slider (*0% to 100% Logistic vs XGBoost*).
   - Animated 4-step pipeline execution sequence.
   - Radial Risk Score Gauge + Clinical Triage Recommendation Protocol.

2. **4-Algorithm Flow Architecture**:
   - Theoretical flowchart cards explaining how data passes through K-Means $\rightarrow$ RFE $\rightarrow$ Logistic $\rightarrow$ XGBoost.

3. **Explainable AI & Symptom Drivers**:
   - Horizontal Recharts bar charts comparing Logistic Regression coefficients against XGBoost gain importances.

4. **K-Means Patient Segments**:
   - Multi-dimensional Radar / Spider Chart comparing the 4 clusters across Age, Hospital Stay, Admissions, Lab Count, and Comorbidities.

5. **Model Metrics & RFE Analytics**:
   - Accuracy & AUC-ROC metrics for both models + RFE Recursive Elimination Line Chart.

6. **Audit History Log**:
   - Real-time searchable log of all evaluated patients with timestamp, individual algorithm probabilities, ensemble risk score, and risk badges.

---

## 6. API Endpoints Reference

### Express Gateway (`http://localhost:5000`)
- `GET /api/health` -> System health and FastAPI engine connectivity.
- `GET /api/samples` -> Curated sample patient profiles.
- `GET /api/metrics` -> Model accuracy, AUC-ROC, and RFE trace.
- `POST /api/assess` -> Triggers full 4-stage pipeline prediction.
- `GET /api/history` -> Recent patient assessment audit trail.

---

## 7. How to Run & Operate the System

### Single-Command Start (Recommended)
```bash
npm start
```
*or double-click:*
```cmd
.\start_app.bat
```

### Manual 3-Terminal Start
1. **ML Microservice**: `python -E ml_engine/main.py`
2. **Backend Gateway**: `node backend/server.js`
3. **Frontend Dashboard**: `cd frontend && npm run dev`
