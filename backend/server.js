const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_ML_URL = process.env.PYTHON_ML_URL || 'http://127.0.0.1:8000';

app.use(cors());
app.use(express.json());

// In-memory audit log for recent patient evaluations
const predictionHistory = [];

// Curated Sample Patient Presets
const SAMPLE_PATIENTS = [
  {
    id: "P-1001",
    name: "Eleanor Vance (High Risk)",
    age: 78,
    gender: "Female",
    admission_type: "Emergency",
    primary_diagnosis_code: "E11",
    diagnosis_name: "Type 2 Diabetes Mellitus with Complications",
    num_prior_admissions: 4,
    time_in_hospital: 9,
    num_lab_procedures: 88,
    num_medications: 35,
    has_comorbidity: 1,
    discharge_disposition: "Rehabilitation",
    insurance_type: "Medicare",
    hospital_id: 5,
    clinical_note: "Polypharmacy, frequent prior emergency visits, elevated HbA1c and renal parameters."
  },
  {
    id: "P-1002",
    name: "Marcus Brody (Low Risk)",
    age: 44,
    gender: "Male",
    admission_type: "Elective",
    primary_diagnosis_code: "G47",
    diagnosis_name: "Sleep Apnea / Routine Airway Surgery",
    num_prior_admissions: 0,
    time_in_hospital: 2,
    num_lab_procedures: 14,
    num_medications: 4,
    has_comorbidity: 0,
    discharge_disposition: "Home",
    insurance_type: "Private",
    hospital_id: 2,
    clinical_note: "Uncomplicated elective recovery, strong familial support, baseline normal lab values."
  },
  {
    id: "P-1003",
    name: "Sofia Rodriguez (Moderate Risk)",
    age: 63,
    gender: "Female",
    admission_type: "Urgent",
    primary_diagnosis_code: "J45",
    diagnosis_name: "Asthma Exacerbation / Chronic Airway Obstruction",
    num_prior_admissions: 2,
    time_in_hospital: 5,
    num_lab_procedures: 46,
    num_medications: 18,
    has_comorbidity: 1,
    discharge_disposition: "Home",
    insurance_type: "Medicaid",
    hospital_id: 3,
    clinical_note: "Moderate respiratory stress, secondary comorbidity, outpatient pulmonology follow-up needed."
  },
  {
    id: "P-1004",
    name: "David Kim (Acute Emergency)",
    age: 71,
    gender: "Male",
    admission_type: "Emergency",
    primary_diagnosis_code: "I10",
    diagnosis_name: "Essential Hypertension & Cardiac Strain",
    num_prior_admissions: 3,
    time_in_hospital: 7,
    num_lab_procedures: 75,
    num_medications: 28,
    has_comorbidity: 1,
    discharge_disposition: "Transfer",
    insurance_type: "Medicare",
    hospital_id: 1,
    clinical_note: "Transferred post-stabilization, requires aggressive blood pressure management & telemetry monitoring."
  }
];

// Fallback Javascript ML Pipeline Engine (Runs if Python service is initializing)
function executeFallbackPipeline(patientInput, weightLogistic = 0.35, weightXgb = 0.65) {
  const { age, time_in_hospital, num_prior_admissions, num_lab_procedures, num_medications, has_comorbidity, admission_type } = patientInput;
  
  // 1. K-Means Persona Scoring
  let clusterId = 3;
  let persona = {
    name: "Low Risk Elective Recovery",
    badge: "Low Risk Persona",
    description: "Younger/mid-age elective stay with minimal prior readmissions and smooth discharge trajectory.",
    risk_level: "Low"
  };
  
  if (age >= 70 && num_prior_admissions >= 3) {
    clusterId = 0;
    persona = {
      name: "High Risk Elderly Comorbid",
      badge: "Critical Risk Persona",
      description: "Older patient demographic with elevated prior admissions, multiple chronic comorbidities, and extended hospital stay.",
      risk_level: "High"
    };
  } else if (admission_type === 'Emergency' && num_lab_procedures >= 60) {
    clusterId = 1;
    persona = {
      name: "Acute Emergency High-Procedure",
      badge: "Acute Care Persona",
      description: "Emergency admission with heavy diagnostic lab work, high medication count, and acute clinical interventions.",
      risk_level: "High"
    };
  } else if (num_prior_admissions >= 1 || has_comorbidity === 1) {
    clusterId = 2;
    persona = {
      name: "Moderate Risk Chronic Care",
      badge: "Managed Care Persona",
      description: "Middle-to-older age patients with recurring urgent visits, stable vital parameters, and baseline comorbidity.",
      risk_level: "Moderate"
    };
  }

  // 2. RFE Feature Contributions
  const rfeFeatures = [
    'age', 'num_prior_admissions', 'time_in_hospital', 'num_lab_procedures', 
    'num_medications', 'has_comorbidity', 'admission_type_Emergency', 
    'primary_diagnosis_code_E11', 'discharge_disposition_Rehabilitation', 
    'insurance_type_Medicare', `patient_segment_Cluster_${clusterId}`
  ];

  // 3. Logistic Regression Simulation
  let logit = -3.2 
    + (age * 0.035) 
    + (num_prior_admissions * 0.42) 
    + (time_in_hospital * 0.12) 
    + (num_lab_procedures * 0.015) 
    + (num_medications * 0.028) 
    + (has_comorbidity * 0.55) 
    + (admission_type === 'Emergency' ? 0.45 : 0)
    + (clusterId === 0 ? 0.65 : clusterId === 1 ? 0.40 : 0);
    
  const logisticProb = 1 / (1 + Math.exp(-logit));

  // 4. XGBoost Non-Linear Simulation
  const nonLinearFactor = Math.pow(num_prior_admissions, 1.3) * 0.08 + (age > 70 && has_comorbidity ? 0.15 : 0);
  const xgbProb = Math.min(0.98, Math.max(0.02, logisticProb * 0.85 + nonLinearFactor));

  // Ensemble Score
  const totalW = weightLogistic + weightXgb;
  const wLog = weightLogistic / totalW;
  const wXgb = weightXgb / totalW;
  const ensembleScore = wLog * logisticProb + wXgb * xgbProb;

  let riskTier = "Low Risk";
  let badgeColor = "#10b981";
  let recommendation = "Routine Recovery Pathway: Patient demonstrates favorable clinical profile with low expected readmission probability.";

  if (ensembleScore >= 0.70) {
    riskTier = "Critical Risk";
    badgeColor = "#ef4444";
    recommendation = "Immediate Clinical Intervention: Schedule mandatory outpatient follow-up within 48-72 hours, assign care manager, and optimize discharge regimen.";
  } else if (ensembleScore >= 0.45) {
    riskTier = "High Risk";
    badgeColor = "#f97316";
    recommendation = "Enhanced Monitoring: Recommend medication reconciliation call within 5 days and telehealth check-in at 14 days.";
  } else if (ensembleScore >= 0.25) {
    riskTier = "Moderate Risk";
    badgeColor = "#eab308";
    recommendation = "Standard Post-Discharge Care: Provide standard discharge instructions and standard primary care follow-up.";
  }

  return {
    patient_input: patientInput,
    engine_mode: "fallback_js",
    pipeline_stages: {
      stage1_kmeans: {
        cluster_id: clusterId,
        persona: persona,
        cluster_centroids: { avg_age: age, avg_hospital_days: time_in_hospital, count: 2500 }
      },
      stage2_rfe: {
        total_features_evaluated: 24,
        selected_features_count: rfeFeatures.length,
        selected_features: rfeFeatures
      },
      stage3_logistic_regression: {
        readmission_probability: parseFloat(logisticProb.toFixed(4)),
        percentage: parseFloat((logisticProb * 100).toFixed(1)),
        top_feature_contributions: [
          { feature: "num_prior_admissions", value: num_prior_admissions, coefficient: 0.42, contribution: +(num_prior_admissions * 0.42).toFixed(4), impact: "Increases Risk" },
          { feature: "has_comorbidity", value: has_comorbidity, coefficient: 0.55, contribution: +(has_comorbidity * 0.55).toFixed(4), impact: "Increases Risk" },
          { feature: `patient_segment_Cluster_${clusterId}`, value: 1, coefficient: 0.65, contribution: 0.65, impact: "Increases Risk" },
          { feature: "age", value: age, coefficient: 0.035, contribution: +(age * 0.035).toFixed(4), impact: "Increases Risk" },
          { feature: "time_in_hospital", value: time_in_hospital, coefficient: 0.12, contribution: +(time_in_hospital * 0.12).toFixed(4), impact: "Increases Risk" }
        ]
      },
      stage4_xgboost: {
        readmission_probability: parseFloat(xgbProb.toFixed(4)),
        percentage: parseFloat((xgbProb * 100).toFixed(1)),
        top_importance_drivers: [
          { feature: "num_prior_admissions", importance: 0.285 },
          { feature: "patient_segment_Cluster_0", importance: 0.210 },
          { feature: "has_comorbidity", importance: 0.165 },
          { feature: "num_lab_procedures", importance: 0.120 },
          { feature: "time_in_hospital", importance: 0.095 }
        ]
      }
    },
    ensemble_result: {
      ensemble_score: parseFloat(ensembleScore.toFixed(4)),
      readmission_risk_percentage: parseFloat((ensembleScore * 100).toFixed(1)),
      risk_tier: riskTier,
      badge_color: badgeColor,
      recommendation: recommendation,
      weights_used: {
        logistic_weight: parseFloat(wLog.toFixed(2)),
        xgboost_weight: parseFloat(wXgb.toFixed(2))
      }
    }
  };
}

// Routes
app.get('/api/health', async (req, res) => {
  try {
    const pyHealth = await axios.get(`${PYTHON_ML_URL}/health`, { timeout: 1500 });
    res.json({
      status: "online",
      gateway: "Node.js Express Gateway",
      ml_microservice: pyHealth.data
    });
  } catch (err) {
    res.json({
      status: "degraded",
      gateway: "Node.js Express Gateway",
      ml_microservice: { status: "offline_or_booting", fallback_active: true }
    });
  }
});

app.get('/api/samples', (req, res) => {
  res.json({ samples: SAMPLE_PATIENTS });
});

app.get('/api/metrics', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_ML_URL}/metrics`, { timeout: 3000 });
    res.json(response.data);
  } catch (err) {
    // Return baseline mock metrics if python service is offline
    res.json({
      metrics: {
        logistic_regression: { accuracy: 0.784, auc_roc: 0.832, precision: 0.741, recall: 0.765 },
        xgboost: { accuracy: 0.865, auc_roc: 0.914, precision: 0.838, recall: 0.852 },
        total_samples: 10000,
        readmission_rate_overall: 0.324,
        rfe_selected_count: 15
      },
      fallback_active: true
    });
  }
});

app.get('/api/pipeline-details', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_ML_URL}/pipeline-details`, { timeout: 3000 });
    res.json(response.data);
  } catch (err) {
    res.json({
      pipeline_description: {
        algorithm1: { name: "K-Means Clustering", type: "Unsupervised Learning", n_clusters: 4 },
        algorithm2: { name: "Recursive Feature Elimination (RFE)", type: "Recursive Selection", target_features: 15 },
        algorithm3: { name: "Logistic / Ridge Regression", type: "Baseline Supervised Learning" },
        algorithm4: { name: "XGBoost Classifier", type: "Advanced Ensemble Engine" }
      }
    });
  }
});

app.post('/api/assess', async (req, res) => {
  const patientData = req.body;
  try {
    const response = await axios.post(`${PYTHON_ML_URL}/predict`, patientData, { timeout: 4000 });
    const result = { ...response.data, engine_mode: "python_fastapi", timestamp: new Date().toISOString() };
    predictionHistory.unshift(result);
    if (predictionHistory.length > 50) predictionHistory.pop();
    res.json(result);
  } catch (err) {
    console.log("[Express] Python ML microservice offline/busy, running fallback JS engine...");
    const fallbackResult = executeFallbackPipeline(patientData);
    fallbackResult.timestamp = new Date().toISOString();
    predictionHistory.unshift(fallbackResult);
    if (predictionHistory.length > 50) predictionHistory.pop();
    res.json(fallbackResult);
  }
});

app.get('/api/history', (req, res) => {
  res.json({ history: predictionHistory });
});

app.listen(PORT, () => {
  console.log(`[Express Gateway] Server running on http://localhost:${PORT}`);
});
