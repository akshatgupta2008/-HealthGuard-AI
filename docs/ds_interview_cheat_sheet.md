# HealthGuard AI: Data Science & Machine Learning Interview Cheat Sheet

This guide provides exact talking points, technical deep dives, mathematical justifications, and ready-to-use resume bullet points to present this project confidently in **Data Science, Machine Learning, and Healthcare Analytics Interviews**.

---

## 1. 🎯 Elevator Pitches

### 60-Second Version:
> *"I built HealthGuard AI, an end-to-end 4-algorithm Machine Learning pipeline designed to predict 30-day hospital readmission risk on a dataset of 10,000 patient records. To tackle high-dimensional EHR noise and model interpretability, I designed a 4-stage architecture: First, K-Means clustering performs unsupervised patient segmentation to create persona features. Second, Recursive Feature Elimination (RFE) prunes noisy predictors down to the top 15 features. Third, Logistic Regression establishes baseline risk probabilities and symptom coefficient weights for clinician explainability. Fourth, XGBoost captures non-linear clinical interactions. The models are blended via a dynamic weighted ensemble slider and served through a FastAPI microservice, React 19 dashboard, and Node.js backend."*

### 3-Minute Version:
> *"In clinical predictive analytics, black-box ML models are often met with skepticism by healthcare providers because clinicians need to know why a patient is flagged as high risk. To address both predictive performance and clinical explainability, I designed HealthGuard AI around a sequential 4-algorithm pipeline.*
>
> *Stage 1 uses K-Means clustering to discover 4 core patient personas—such as High-Risk Elderly Comorbid vs. Acute Emergency High-Procedure patients—and appends a cluster segment feature to the dataset.*
> *Stage 2 applies Recursive Feature Elimination (RFE) to prune noisy variables down to the top 15 critical predictors, improving inference efficiency.*
> *Stage 3 trains a Ridge Logistic Regression baseline to derive linear symptom risk weights for clinician explainability.*
> *Stage 4 fits an XGBoost gradient boosted decision tree ensemble to capture non-linear feature interactions, achieving an AUC-ROC of 0.6672.*
>
> *Finally, I deployed the model using FastAPI, joblib serialization, and containerized microservices integrated with an interactive clinical dashboard."*

---

## 2. 💡 Key Technical Q&A for Technical Interviews

### Q1: Why run K-Means Clustering BEFORE supervised training?
* **Answer**: Unsupervised clustering discovers hidden patient sub-cohorts (e.g., elderly comorbid vs. acute procedure-driven admissions). By appending the K-Means cluster assignment as a feature before RFE and supervised modeling, we allow downstream classifiers to leverage persona-level baseline risk offsets.

### Q2: Why use RFE instead of simple correlation filtering?
* **Answer**: Correlation filtering only evaluates individual linear pairwise relationships with the target. RFE evaluates feature importance *in the presence of other features*, iteratively pruning the weakest feature and re-fitting the estimator until an optimal subset is isolated.

### Q3: Why blend Logistic Regression with XGBoost in an ensemble?
* **Answer**: Logistic Regression provides linear interpretability (clinicians can see exact log-odds coefficient weights for symptoms), while XGBoost captures non-linear feature interactions and high-dimensional splits. Blending them gives healthcare providers both explainability and enhanced predictive AUC-ROC.

---

## 3. 📊 Resume Bullet Points (Copy & Paste Ready)

* **Architected an end-to-end 4-algorithm Machine Learning readmission pipeline** on 10,000 patient records using **K-Means**, **RFE**, **Logistic Regression**, and **XGBoost**.
* **Engineered unsupervised patient segmentation** with K-Means clustering ($k=4$), isolating key patient personas (`Elderly Comorbid`, `Acute Emergency`) to enrich downstream predictive feature matrices.
* **Implemented Recursive Feature Elimination (RFE)** to prune noisy variables down to 15 critical clinical predictors, reducing model inference latency to **<15ms**.
* **Developed interpretable symptom risk weighting** with Ridge Logistic Regression, providing linear log-odds coefficient explainability for clinical decision support.
* **Deployed real-time prediction microservice** using **Python FastAPI**, **Joblib**, **Express Node.js**, and **React 19**, enabling dynamic weighted risk scoring and automated unit testing (`pytest`).

---

## 4. 🛠️ Key Pipeline Specifications Summary

| Metric / Parameter | Value / Implementation Details |
| :--- | :--- |
| **Dataset Size** | 10,000 Patient Records (`data/readmission_dataset.csv`) |
| **Baseline Readmission Rate** | 27.07% (2,707 readmitted patients) |
| **Cluster Algorithm** | K-Means ($k=4$, `n_init=10`, `StandardScaler`) |
| **Feature Selection** | Pure Recursive Feature Elimination (RFE, top 15 features) |
| **Baseline Model** | Ridge Logistic Regression (`max_iter=1000`) |
| **Ensemble Engine** | XGBoost Classifier (`n_estimators=100`, `learning_rate=0.08`, `max_depth=5`) |
| **Serving Stack** | Python 3.14, FastAPI, Pydantic, Express Node.js, React 19 |
