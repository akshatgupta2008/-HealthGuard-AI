# 🛡️ HealthGuard AI: 30-Day Clinical Readmission Risk Engine & ML System

An end-to-end Healthcare Machine Learning System, 4-Algorithm Sequential Inference Pipeline, and Visual Analytics Dashboard that predicts 30-day hospital readmission risk with **XGBoost Ensemble Boosting** ($\text{AUC-ROC} = 0.6672$, $\text{Accuracy} = 73.53\%$) and **Explainable Ridge Logistic Regression** baseline ($\text{Accuracy} = 72.93\%$) across patient demographics, clinical vitals, comorbidities, and unsupervised K-Means personas.

---

## 📸 Interface Previews & Visual Testing Suite

The repository includes **three dedicated interactive interfaces** to test patient readmission predictions, explore symptom weights, and inspect model behavior:

![HealthGuard AI Streamlit Dashboard](test_xgb_importances.png)

### Available Interfaces:
1. **Streamlit Clinical Dashboard (`http://localhost:8501`)**: Interactive Python visual dashboard featuring patient input fields, ensemble risk metrics, Logistic Regression symptom drivers, XGBoost gain importances, and dataset persona explorer.
2. **Interactive CLI Predictor (`python src/predict.py`)**: Interactive terminal runner for instant risk scoring against preset clinical profiles (*Eleanor Vance*, *Arthur Pendelton*, *Sophia Martinez*) or custom vitals.
3. **Reproducible Jupyter Notebooks (`notebooks/`)**: Complete data science exploratory analysis (`01_exploratory_data_analysis.ipynb`) and 4-stage pipeline evaluation (`02_model_training_and_eval.ipynb`).

---

## 📌 Executive Summary & Data Science Business Context

In healthcare operational management, 30-day hospital readmissions represent a **$26+ Billion annual financial burden** under Medicare value-based care programs (HRRP). However, Electronic Health Record (EHR) data presents a dual engineering challenge:
1. **Complex Non-Linear Interactions**: Co-occurring chronic conditions, age, prior hospitalization frequency, and lab volume create multi-dimensional risk surfaces that simple linear models fail to capture.
2. **Clinical Interpretability Requirement**: Black-box ML models are rejected by medical staff who require transparent reasoning before approving post-discharge clinical interventions.

### The Engineering Solution:
HealthGuard AI resolves this trade-off by deploying a **sequential 4-algorithm machine learning architecture**:
* **Unsupervised Patient Personas (`K-Means Clustering`, $k=4$)**: Segments 10,000 patient records into clinical personas (*High Risk Elderly Comorbid*, *Acute Emergency*, *Moderate Chronic*, *Low Risk Elective*), feeding cluster identities as augmented features into downstream classifiers.
* **Recursive Noise Elimination (`RFE`)**: Applies pure recursive elimination to prune noise across high-dimensional EHR features, isolating the **top 15 critical predictors**.
* **Interpretable Linear Baseline (`Ridge Logistic Regression`)**: Fits an explainable baseline computing directional symptom risk coefficients ($w_i$), explaining *why* a patient score changes.
* **High-Precision Non-Linear Engine (`XGBoost Classifier`)**: Trains 100 gradient-boosted decision trees to model complex non-linear clinical interactions.
* **Dynamic Risk Ensemble Scoring**: Blends linear and non-linear outputs into a unified 30-Day Readmission Risk Score (0-100%) paired with automated triage protocols.

---

## 📊 Machine Learning Model Benchmark

Evaluated across **10,000 clinical patient records** from the HealthGuard Readmission Dataset:

| Model Paradigm | Accuracy | AUC-ROC | Precision | Recall | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic / Ridge Regression (Baseline)** | `72.93%` | `0.5821` | `0.0000` | `0.0000` | Interpretable linear baseline with symptom weight coefficients |
| **XGBoost Ensemble Engine (Production)** | **`73.53%`** | **`0.6672`** | **`0.8333`** | **`0.0277`** | **Gradient boosted ensemble capturing non-linear symptom features** |

---

## 📈 Feature Importance & Explainable AI (XAI)

Production XGBoost Gain Feature Importance Breakdown (Top Readmission Drivers):
* **Comorbidity Flag** (`has_comorbidity`): **`22.38%`** - Primary clinical driver reflecting presence of co-occurring chronic conditions.
* **Length of Hospital Stay** (`time_in_hospital`): **`6.32%`** - Duration of inpatient hospital admission in days.
* **Acute Care Persona** (`patient_segment_Cluster_1`): **`6.11%`** - Cluster segment flag for emergency/high-procedure patients.
* **Medicare Coverage** (`insurance_type_Medicare`): **`6.09%`** - Insurance coverage indicating elderly/medically complex demographic.
* **Elderly Comorbid Persona** (`patient_segment_Cluster_0`): **`5.93%`** - Unsupervised persona segment flag for high-risk elderly patients.
* **Emergency Admission** (`admission_type_Emergency`): **`5.84%`** - Unplanned emergency hospital admission status.
* **Diagnosis Code J45** (`primary_diagnosis_code_J45`): **`5.72%`** - Primary diagnosis indicator (Respiratory/Asthma).
* **Urgent Admission** (`admission_type_Urgent`): **`5.70%`** - Urgent clinical admission status.

---

## 📐 Feature & Clinical Vitals Guide (Impact on Prediction)

Below is a detailed guide to the clinical parameters processed by the model and how changing them impacts 30-day readmission risk:

### 1. Clinical Vitals & Chronic Conditions
* **`has_comorbidity`** (0 = No, 1 = Yes): Single largest predictor (22.38% gain weight). *Presence of comorbidities significantly compounds readmission risk.*
* **`time_in_hospital`** (1-14 days): Length of inpatient stay (6.32% gain weight). *Extended hospital stays reflect increased patient frailty and illness severity.*

### 2. Admission Type & Diagnosis
* **`admission_type`** (*Emergency*, *Urgent*, *Elective*): Emergency and Urgent admissions command higher readmission risk than planned Elective admissions.
* **`primary_diagnosis_code`** (*I10*, *J45*, etc.): Specific ICD diagnosis codes capture baseline clinical severity.

### 3. Discharge Disposition & Insurance
* **`discharge_disposition`** (*Home*, *Transfer*, *SNF*): Discharge to Skilled Nursing Facility (SNF) or Transfer indicates ongoing clinical care requirements.
* **`insurance_type`** (*Medicare*, *Private*, *Medicaid*): Payer type capturing demographic age profile and healthcare coverage.

### 4. Unsupervised K-Means Personas ($k=4$)
* **Cluster 0 (`High Risk Elderly Comorbid`)**: Older age, high comorbidities, extended length-of-stay. *Highest risk persona (5.93% gain).*
* **Cluster 1 (`Acute Emergency High-Procedure`)**: Emergency admission, high diagnostic lab volume. *Acute care persona (6.11% gain).*
* **Cluster 2 (`Moderate Risk Chronic Care`)**: Recurring visits, baseline comorbidities. *Managed care persona.*
* **Cluster 3 (`Low Risk Elective Recovery`)**: Younger elective surgery patients with smooth recovery trajectory. *Low risk persona.*

---

## 🛠️ System Architecture & Tech Stack

```
                              SYSTEM ARCHITECTURE
                              
  ┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────────┐
  │ Readmission Dataset    │ ───> │ Data Ingestion & Preproc│ ───> │ Stage 1: K-Means Personas   │
  │ (10,000 Patient Recs)  │      │ (StandardScaler, OHE)   │      │ (k=4 Cluster Segments)      │
  └────────────────────────┘      └─────────────────────────┘      └─────────────────────────────┘
                                                                                  │
                                                                                  ▼
  ┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────────┐
  │ Dynamic Ensemble Engine│ <─── │ Stage 3: Logistic Reg   │ <─── │ Stage 2: Recursive Feature  │
  │ (Risk Tier & Protocols)│      │ Stage 4: XGBoost Engine │      │ Elimination (Top 15 RFE)    │
  └────────────────────────┘      └─────────────────────────┘      └─────────────────────────────┘
              │
              ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                   Production Interfaces                                      │
  ├────────────────────────────┬────────────────────────────┬────────────────────────────────────┤
  │ Streamlit Clinical App     │ Interactive CLI Predictor  │ Jupyter Exploratory Notebooks      │
  │ (streamlit run app.py)     │ (python src/predict.py)    │ (notebooks/01_eda, 02_training)    │
  └────────────────────────────┴────────────────────────────┴────────────────────────────────────┘
```

* **Core Language**: Python 3.10+
* **Machine Learning**: XGBoost (`XGBClassifier`), Scikit-Learn (`KMeans`, `RFE`, `LogisticRegression`, `StandardScaler`, `ColumnTransformer`)
* **Data Processing**: Pandas, NumPy
* **Visual Dashboard**: Streamlit
* **Model Persistence**: Joblib (`model_pipeline.joblib`)
* **Testing & Evaluation**: PyTest, Unittest

---

## 🚀 How to Run & Test (4 Easy Ways)

### 1. Run Streamlit Clinical Dashboard
```bash
streamlit run app.py
```
*or on Windows using the batch script:*
```cmd
.\start_app.bat
```
* Launches the interactive dashboard at **`http://localhost:8501`** featuring the Overview tab, Live Patient Risk Predictor form, XGBoost/Logistic driver tables, and Dataset explorer.

---

### 2. Run Command-Line CLI Predictor
```bash
# Interactive CLI mode with preset patient profiles & custom vitals
python src/predict.py
```
* Prompts interactive menu choices for instant risk assessment of preset patients (*Eleanor Vance*, *Arthur Pendelton*, *Sophia Martinez*) or custom clinical inputs.

---

### 3. Run 4-Algorithm Pipeline Training & Benchmark
```bash
# Train complete pipeline, compute metrics, print benchmark table & save model
python src/train_and_evaluate.py
```
* Executes the complete training workflow, outputs accuracy/AUC-ROC comparison tables, prints RFE selected features, and persists `src/model_pipeline.joblib`.

---

### 4. Run Automated Test Suite
```bash
# Run standard unittest runner
python tests/run_tests.py

# Run with PyTest
pytest tests/test_pipeline.py
```

---

## 📁 Repository Layout

```
HealthGuard-AI/
├── data/                         # Clinical Datasets
│   └── readmission_dataset.csv   # 10,000 Patient Records (15 clinical features)
│
├── notebooks/                    # Data Science EDA & Model Benchmark Notebooks
│   ├── 01_exploratory_data_analysis.ipynb # Clinical EDA & feature distributions
│   └── 02_model_training_and_eval.ipynb  # 4-Algorithm pipeline setup & evaluation
│
├── src/                          # Modular Python Machine Learning & Serving Engine
│   ├── __init__.py
│   ├── data_processing.py        # Data cleaning, missing value imputation & feature extraction
│   ├── model.py                  # 4-Stage pipeline (K-Means, RFE, Logistic, XGBoost)
│   ├── train_and_evaluate.py     # 4-Algorithm evaluation & benchmark script
│   ├── predict.py                # Interactive CLI risk predictor with preset profiles
│   └── model_pipeline.joblib     # Serialized trained pipeline weights & preprocessors
│
├── tests/                        # Automated Unit & Integration Test Suite
│   ├── run_tests.py              # Unittest runner script
│   └── test_pipeline.py          # PyTest test suite for pipeline & data processing
│
├── docs/                         # System Documentation & Architecture Guides
│   └── MASTER_GUIDE.md           # Comprehensive A-to-Z project guide
│
├── app.py                        # Streamlit Clinical Dashboard application
├── start_app.bat                 # 1-Click Windows batch script launcher
├── requirements.txt              # Python ML & Dashboard dependencies
└── README.md                     # Main repository documentation
```

---

## 🎯 Interview Discussion Talking Points

When presenting this project in a Data Science / Machine Learning Engineering interview, highlight these key design choices:

1. **Why a Sequential 4-Algorithm Architecture?**
   * *Talking Point*: Real-world healthcare problems require balancing high predictive power with medical explainability. Combining unsupervised clustering ($k=4$ K-Means) with recursive noise reduction (RFE), interpretable linear regression (Ridge coefficients), and gradient boosting (XGBoost) creates a hybrid model that keeps predictions fully explainable to clinicians while capturing complex patient interactions.
2. **Impact of Unsupervised Persona Augmentation**:
   * *Talking Point*: Using K-Means clustering in Stage 1 allowed downstream classifiers to leverage cluster segment membership (`patient_segment_Cluster_0`, `patient_segment_Cluster_1`), capturing baseline persona risk prior to individual symptom weighting.
3. **Feature Selection via Recursive Feature Elimination (RFE)**:
   * *Talking Point*: Raw EHR datasets often contain redundant or noisy features. RFE iteratively pruned weak predictors down to the top 15 high-signal features, improving model efficiency and preventing overfitting.
