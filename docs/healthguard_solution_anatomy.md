# HealthGuard AI: 4-Algorithm ML Healthcare Readmission Pipeline Architecture & Solution Anatomy

## 📌 Business Problem & Clinical Context

Hospital readmissions within 30 days of discharge represent significant financial penalties for healthcare providers under the **Hospital Readmissions Reduction Program (HRRP)** and indicate potential gaps in post-discharge care. 

The **HealthGuard AI Readmission Dataset** contains 10,000 anonymized patient electronic health records (EHR) spanning patient demographics, prior admission counts, length of stay, comorbidities, lab procedure volume, medication counts, primary diagnosis codes, and insurance categories.

### The Clinical Data Challenge:
Clinical EHR datasets are high-dimensional and noisy. Simple linear models miss non-linear comorbidity synergies, while unconstrained complex models act as uninterpretable black boxes that clinicians hesitate to trust.

### Engineering & ML Architecture Solution:
HealthGuard AI resolves this by engineering a sequential **4-Algorithm Machine Learning Pipeline**:

```
                       HEALTHGUARD AI PIPELINE ARCHITECTURE
                                
       [ Patient Clinical EHR Dataset (10,000 Patient Records, 15 Features) ]
                                        │
                                        ▼
      ┌──────────────────────────────────────────────────────────────────┐
      │ Stage 1: Unsupervised K-Means Patient Segmentation (k=4)         │
      │ Discovers persona clusters -> Appends "Patient Segment" Feature  │
      └─────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼
      ┌──────────────────────────────────────────────────────────────────┐
      │ Stage 2: Recursive Feature Elimination (RFE)                     │
      │ Pure recursive pruning -> Selects top 15 critical predictors     │
      └─────────────────────────────────┬────────────────────────────────┘
                                        │
                       ┌────────────────┴────────────────┐
                       ▼                                 ▼
      ┌─────────────────────────────────┐  ┌─────────────────────────────┐
      │ Stage 3: Ridge Logistic Regr.   │  │ Stage 4: XGBoost Engine     │
      │ Interpretable baseline &        │  │ Non-linear feature interactions│
      │ linear symptom risk weights     │  │ & gain feature importances  │
      └────────────────┬────────────────┘  └──────────────┬──────────────┘
                       │                                  │
                       └────────────────┬─────────────────┘
                                        ▼
      ┌──────────────────────────────────────────────────────────────────┐
      │ Dynamic Weighted Risk Score Ensemble & Clinical Intervention Plan│
      │ (Critical, High, Moderate, Low Risk Tiers)                       │
      └──────────────────────────────────────────────────────────────────┘
```

---

## 🔬 The 4 Pipeline Algorithms Breakdown

### 1. Stage 1: K-Means Clustering (Unsupervised Learning)
* **Objective**: Groups patients into 4 distinct persona clusters based on standardized clinical attributes.
* **Persona Profiles**:
  * `Cluster 0`: High Risk Elderly Comorbid (Elevated prior admissions & chronic conditions)
  * `Cluster 1`: Acute Emergency High-Procedure (Emergency admission, high lab work & meds)
  * `Cluster 2`: Moderate Risk Chronic Care (Urgent/outpatient visits with baseline stability)
  * `Cluster 3`: Low Risk Elective Recovery (Younger elective surgery recovery pathway)
* **ML Value**: Appends a categorical `"patient_segment"` feature that informs downstream classifiers.

### 2. Stage 2: Recursive Feature Elimination - RFE (Pure Recursion)
* **Objective**: Recursively trains estimators, ranks feature coefficients, prunes the lowest-weight variable, and recurses until the top 15 predictors remain.
* **ML Value**: Eliminates clinical noise, reduces over-fitting, and optimizes real-time prediction latency (<15ms).

### 3. Stage 3: Logistic / Ridge Regression (Baseline Supervised Learning)
* **Objective**: Fits linear decision boundaries over RFE features to output baseline readmission probabilities.
* **ML Value**: Provides clear, linear coefficient risk weights so clinicians understand *which symptoms* increase or decrease readmission risk.

### 4. Stage 4: XGBoost Classifier (Advanced Ensemble Learning)
* **Objective**: Trains a gradient-boosted decision tree ensemble over RFE features.
* **ML Value**: Captures non-linear feature interactions (e.g., comorbidity compounding with extended stay duration) and outputs relative Gain Feature Importances.

---

## 📊 Benchmark & Empirical Performance Metrics

Evaluated on 10,000 patient records:

| Model / Pipeline Stage | Accuracy | AUC-ROC Score | Precision | Recall | Primary Strengths |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | `72.93%` | `0.5821` | Baseline | Baseline | Linear symptom explainability & coefficient risk weights |
| **XGBoost Ensemble Engine** | **`73.53%`** | **`0.6672`** | `0.8333` | Elevated | High precision, non-linear interaction modeling |

---

## 🚀 Execution & Testing Instructions

### 1. Run Automated Test Suite:
```bash
python tests/run_tests.py
```

### 2. Run Data Science Benchmark Script:
```bash
python src/train_and_evaluate.py
```

### 3. Launch Interactive Terminal Risk Predictor:
```bash
python src/predict.py
```

### 4. Train & Serialize Production ML Model:
```bash
python src/model.py
```

### 5. Launch FastAPI Microservice Server:
```bash
python src/api.py
```
* Interactive API Documentation live at: `http://127.0.0.1:8000/docs`
