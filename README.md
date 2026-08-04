# HealthGuard AI - 4-Algorithm ML Healthcare Readmission Pipeline 🏥🤖

> **An end-to-end machine learning pipeline and clinical decision support system** that combines unsupervised patient segmentation, recursive feature selection, interpretable baseline modeling, and high-precision nonlinear prediction.

---

## 🌟 Overview

Medical readmission datasets often contain noisy and highly correlated variables. HealthGuard AI addresses this by applying a sequential 4-stage pipeline where each step improves the next:

```text
[ Raw Patient Data ]
        │
        ▼
[ K-Means Clustering ]
        │
        ▼
[ Recursive Feature Elimination ]
        │
        ┌───────────────┬───────────────┐
        ▼               ▼
[ Logistic/Ridge ] [ XGBoost ]
        └───────────────┬───────────────┘
                        ▼
           [ Ensemble Readmission Score ]
```

This workflow creates a more explainable and accurate readmission risk system for clinicians and researchers.

---

## 💡 The 4-Algorithm Breakdown

1. **K-Means Clustering (Unsupervised Learning)**
   - Groups patients into four distinct segments based on clinical and demographic patterns.
   - Adds a new patient-segment feature that improves downstream modeling.

2. **Recursive Feature Elimination (RFE)**
   - Iteratively removes weak predictors and retains the most informative features.
   - Helps reduce noise and improves training efficiency.

3. **Logistic / Ridge Regression (Baseline Supervised Learning)**
   - Provides an interpretable baseline probability for readmission risk.
   - Produces transparent feature-weight insights for clinical explainability.

4. **XGBoost (Advanced Ensemble Learning)**
   - Captures nonlinear interactions and complex decision boundaries.
   - Combines with the baseline model for stronger predictive performance.

---

## 🛠️ Tech Stack

- **ML Engine**: Python, scikit-learn, XGBoost, Pandas, NumPy, Joblib
- **Backend API**: Node.js, Express, Axios, CORS
- **Frontend Dashboard**: React, Vite, Tailwind CSS, Lucide Icons, Recharts

---

## ✅ Prerequisites

Before running the project, make sure you have:

- **Node.js 18+**
- **Python 3.10+**
- **npm** installed with Node
- A local terminal environment with access to `python` and `npm`

---

## 📁 Project Structure

```text
HealthGuard-AI/
├── backend/           # Express API server
├── frontend/          # React + Vite dashboard
├── ml_engine/         # Training pipeline and model artifacts
├── HealthGuard_Readmission_Data/  # Dataset files
├── package.json       # Root scripts to launch the app
├── start_app.bat      # Windows launcher
└── README.md          # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/akshatgupta2008/-HealthGuard-AI
cd .\-HealthGuard-AI
```

### 2. Install dependencies

```bash
npm install
npm install --prefix backend
npm install --prefix frontend
python -m pip install -r ml_engine/requirements.txt
```

### 3. Start the full application

```bash
npm start
```

On Windows, you can also run:

```cmd
.\start_app.bat
```

The dashboard should open at **http://localhost:5173/**.

---

## 🧠 Train the ML Model

To retrain the pipeline and regenerate the model artifact:

```bash
python ml_engine/train_and_save.py
```

This script reads the dataset from the data folder and saves the trained model to the ML engine directory.

---

## 📊 Dataset & Metrics

- **Dataset**: `readmission_dataset.csv` with 10,000 patient records
- **Baseline Readmission Rate**: 27.07%
- **Logistic Regression Baseline**: Accuracy 72.93% | AUC-ROC 0.5821
- **XGBoost Ensemble Engine**: Accuracy 73.53% | AUC-ROC 0.6672

---

## 🔧 Troubleshooting

- If `npm install` fails, make sure you are using a recent Node.js version.
- If Python dependencies fail to install, upgrade `pip` first:

```bash
python -m pip install --upgrade pip
```

- If the app does not open in the browser, confirm that the frontend dev server started successfully and that port `5173` is free.

---

## 📝 License

Distributed under the MIT License.
