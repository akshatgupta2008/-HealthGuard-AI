# HealthGuard AI - 4-Algorithm ML Healthcare Readmission Pipeline 🏥🤖

> **An end-to-end Machine Learning Pipeline & Clinical Decision Support System** combining Unsupervised Patient Segmentation (`K-Means`), Pure Recursive Feature Elimination (`RFE`), Interpretable Baseline Modeling (`Logistic/Ridge Regression`), and High-Precision Non-Linear Predictive Engines (`XGBoost`).

---

## 🌟 Overview & Architecture

Medical readmission datasets contain hundreds of complex, noisy variables (EHR history, lab results, billing flags). **HealthGuard AI** implements a sequential 4-algorithm ML pipeline where the output of each stage enriches the next:

```
[ Raw Patient Data (10,000 Records) ]
                 │
                 ▼
┌──────────────────────────────────────────┐
│  Stage 1: K-Means Clustering (k=4)       │  <-- Unsupervised Patient Segmentation
│  Appends "Patient Segment ID" Feature    │
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

## 💡 The 4-Algorithm Breakdown

1. **K-Means Clustering (Unsupervised Learning)**
   - **Role**: Groups patients into 4 distinct persona clusters based on medical history, vitals, and demographics (`High Risk Elderly Comorbid`, `Acute Emergency High-Procedure`, `Moderate Risk Chronic Care`, `Low Risk Elective Recovery`).
   - **Why**: Creates a new `"Patient Segment"` categorical feature that enhances downstream predictive accuracy.

2. **Recursive Feature Elimination - RFE (Pure Recursion)**
   - **Role**: Recursively fits base estimators, ranks feature importance, prunes the single weakest variable, and recurses until the top 15 critical predictors remain.
   - **Why**: Drastically reduces noise, prevents overfitting, and optimizes model inference speed.

3. **Logistic / Ridge Regression (Baseline Supervised Learning)**
   - **Role**: Establishes baseline readmission probability and provides linear symptom risk weights (coefficients).
   - **Why**: Gives clinicians clear explainability on *why* predictions were generated.

4. **XGBoost (Advanced Ensemble Learning)**
   - **Role**: High-performance gradient boosted decision trees capturing complex non-linear feature interactions.
   - **Why**: Blended with Logistic Regression via a dynamic weighted ensemble slider.

---

## 🛠️ Tech Stack & Microservices

- **ML Microservice Engine**: Python 3.14, FastAPI, Uvicorn, scikit-learn, XGBoost, Pandas, NumPy, Joblib.
- **API Gateway Backend**: Node.js, Express, Axios, CORS.
- **Clinical Dashboard Frontend**: React 19, Vite, Tailwind CSS v4, Lucide Icons, Recharts.

---

## 🚀 Quick Start Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Dependencies
```bash
# Install root orchestrator dependencies
npm install

# Install Node backend dependencies
npm install --prefix backend

# Install React frontend dependencies
npm install --prefix frontend

# Install Python ML dependencies
python -m pip install -r ml_engine/requirements.txt
```

### 3. Launch All Services (One Command)
```bash
npm start
```
*or on Windows, run:*
```cmd
.\start_app.bat
```

Open your browser at **`http://localhost:5173/`** to access the dashboard!

---

## 📊 Dataset & Metrics

- **Dataset**: `readmission_dataset.csv` (10,000 Patient Records).
- **Baseline Readmission Rate**: 27.07%
- **Logistic Regression Baseline**: Accuracy: 72.93% | AUC-ROC: 0.5821
- **XGBoost Ensemble Engine**: Accuracy: 73.53% | AUC-ROC: 0.6672

---

## 📝 License
Distributed under the MIT License.
