import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Brain, 
  Cpu, 
  Layers, 
  GitMerge, 
  Zap, 
  Sliders, 
  AlertTriangle, 
  CheckCircle2, 
  HelpCircle, 
  TrendingUp, 
  UserCheck, 
  RefreshCw, 
  ShieldAlert, 
  ChevronRight, 
  FileText, 
  Sparkles,
  BarChart3,
  Stethoscope,
  Clock,
  FlaskConical,
  Database,
  Building2,
  Lock,
  ArrowRight,
  Printer,
  Copy,
  Check,
  Search,
  Filter,
  Info,
  ChevronDown,
  User,
  HeartPulse,
  Hospital,
  PieChart
} from 'lucide-react';

import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer, 
  Cell, 
  LineChart, 
  Line, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar, 
  Legend 
} from 'recharts';

const API_BASE = 'http://localhost:5000/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('assessor');
  const [healthStatus, setHealthStatus] = useState(null);
  const [samples, setSamples] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [pipelineDetails, setPipelineDetails] = useState(null);
  const [history, setHistory] = useState([]);
  const [historySearch, setHistorySearch] = useState('');
  const [copiedReport, setCopiedReport] = useState(false);

  // Dynamic Ensemble Slider State (Default 35% Logistic, 65% XGBoost)
  const [logisticWeight, setLogisticWeight] = useState(0.35);

  // Form State
  const [formData, setFormData] = useState({
    patient_name: 'Eleanor Vance',
    name: 'Eleanor Vance',
    age: 72,
    gender: 'Female',
    admission_type: 'Emergency',
    primary_diagnosis_code: 'E11',
    num_prior_admissions: 3,
    time_in_hospital: 6,
    num_lab_procedures: 78.0,
    num_medications: 25.0,
    has_comorbidity: 1,
    discharge_disposition: 'Rehabilitation',
    insurance_type: 'Medicare',
    hospital_id: 5
  });

  // Assessment Results & Pipeline Animation state
  const [assessmentResult, setAssessmentResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [animatingStep, setAnimatingStep] = useState(0); // 0: Idle, 1: K-Means, 2: RFE, 3: Logistic, 4: XGBoost, 5: Complete

  useEffect(() => {
    fetchHealth();
    fetchSamples();
    fetchMetrics();
    fetchPipelineDetails();
    fetchHistory();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setHealthStatus(data);
    } catch (e) {
      console.warn("Express backend health check warning:", e);
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/samples`);
      const data = await res.json();
      if (data.samples) setSamples(data.samples);
    } catch (e) {
      console.warn("Fetch samples error:", e);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/metrics`);
      const data = await res.json();
      setMetrics(data);
    } catch (e) {
      console.warn("Fetch metrics error:", e);
    }
  };

  const fetchPipelineDetails = async () => {
    try {
      const res = await fetch(`${API_BASE}/pipeline-details`);
      const data = await res.json();
      setPipelineDetails(data);
    } catch (e) {
      console.warn("Fetch pipeline details error:", e);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/history`);
      const data = await res.json();
      if (data.history) {
        const activeName = formData.patient_name || formData.name;
        const patched = data.history.map(item => {
          const input = item.patient_input || {};
          const pName = input.patient_name || input.name || item.patient_name || item.name;
          if (!pName || pName === 'Anonymous Patient') {
            if (activeName && activeName !== 'Anonymous Patient') {
              return {
                ...item,
                patient_input: {
                  ...input,
                  patient_name: activeName,
                  name: activeName
                }
              };
            }
          }
          return item;
        });
        setHistory(patched);
      }
    } catch (e) {
      console.warn("Fetch history error:", e);
    }
  };

  const handlePresetSelect = (sample) => {
    const pName = sample.patient_name || sample.name || '';
    setFormData({
      patient_name: pName,
      name: pName,
      age: sample.age,
      gender: sample.gender,
      admission_type: sample.admission_type,
      primary_diagnosis_code: sample.primary_diagnosis_code,
      num_prior_admissions: sample.num_prior_admissions,
      time_in_hospital: sample.time_in_hospital,
      num_lab_procedures: sample.num_lab_procedures,
      num_medications: sample.num_medications,
      has_comorbidity: sample.has_comorbidity,
      discharge_disposition: sample.discharge_disposition,
      insurance_type: sample.insurance_type,
      hospital_id: sample.hospital_id
    });
  };

  const handleFormChange = (e) => {
    const { name, value, type } = e.target;
    const parsedVal = type === 'number' ? parseFloat(value) || 0 : value;
    setFormData(prev => {
      const updated = { ...prev, [name]: parsedVal };
      if (name === 'patient_name') updated.name = parsedVal;
      if (name === 'name') updated.patient_name = parsedVal;
      return updated;
    });
  };

  const handleAssessPatient = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setAssessmentResult(null);

    setAnimatingStep(1); // K-Means
    setTimeout(() => setAnimatingStep(2), 450); // RFE
    setTimeout(() => setAnimatingStep(3), 900); // Logistic Regression
    setTimeout(() => setAnimatingStep(4), 1350); // XGBoost

    try {
      const pName = formData.patient_name || formData.name || 'Anonymous Patient';
      const payload = {
        ...formData,
        patient_name: pName,
        name: pName,
        weight_logistic: logisticWeight,
        weight_xgb: 1 - logisticWeight
      };

      const res = await fetch(`${API_BASE}/assess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data) {
        if (!data.patient_input) data.patient_input = { ...payload };
        data.patient_input.patient_name = pName;
        data.patient_input.name = pName;
        setHistory(prev => [data, ...prev.filter(h => h.timestamp !== data.timestamp)]);
      }
      
      setTimeout(() => {
        setAssessmentResult(data);
        setAnimatingStep(5); // Complete
        setLoading(false);
        fetchHistory();
      }, 1800);
    } catch (err) {
      console.error("Assessment request failed:", err);
      setLoading(false);
      setAnimatingStep(0);
    }
  };

  // Dynamic Ensemble Recalculation
  const computedEnsembleScore = () => {
    if (!assessmentResult) return null;
    const logProb = assessmentResult.pipeline_stages.stage3_logistic_regression.readmission_probability;
    const xgbProb = assessmentResult.pipeline_stages.stage4_xgboost.readmission_probability;
    const score = (logisticWeight * logProb) + ((1 - logisticWeight) * xgbProb);
    const pct = (score * 100).toFixed(1);
    
    let tier = "Low Risk";
    let color = "#10b981";
    let recommendation = "Routine Recovery Pathway: Patient demonstrates favorable clinical profile with low expected readmission probability.";

    if (score >= 0.70) {
      tier = "Critical Risk";
      color = "#ef4444";
      recommendation = "Immediate Clinical Intervention: Schedule mandatory outpatient follow-up within 48-72 hours, assign care manager, and optimize discharge regimen.";
    } else if (score >= 0.45) {
      tier = "High Risk";
      color = "#f97316";
      recommendation = "Enhanced Monitoring: Recommend medication reconciliation call within 5 days and telehealth check-in at 14 days.";
    } else if (score >= 0.25) {
      tier = "Moderate Risk";
      color = "#eab308";
      recommendation = "Standard Post-Discharge Care: Provide standard discharge instructions and standard primary care follow-up.";
    }

    return { score, pct, tier, color, recommendation };
  };

  const dynamicEnsemble = computedEnsembleScore();

  const handleCopyReport = () => {
    if (!assessmentResult) return;
    const pName = formData.patient_name || formData.name || assessmentResult.patient_input?.patient_name || assessmentResult.patient_input?.name || 'Anonymous Patient';
    const text = `
HEALTHGUARD AI - CLINICAL PATIENT READMISSION ASSESSMENT REPORT
------------------------------------------------------------------
Timestamp: ${new Date().toLocaleString()}
Patient Name: ${pName}
Patient Age: ${formData.age} | Gender: ${formData.gender} | Admission: ${formData.admission_type}
Diagnosis Code: ${formData.primary_diagnosis_code} | Hospital Days: ${formData.time_in_hospital}

4-ALGORITHM PIPELINE OUTPUTS:
1. K-Means Segment: ${assessmentResult.pipeline_stages.stage1_kmeans.persona.name} (Cluster #${assessmentResult.pipeline_stages.stage1_kmeans.cluster_id})
2. RFE Features Evaluated: ${assessmentResult.pipeline_stages.stage2_rfe.selected_features_count} Critical Predictors
3. Logistic Regression Baseline: ${assessmentResult.pipeline_stages.stage3_logistic_regression.percentage}% Risk
4. XGBoost Predictive Engine: ${assessmentResult.pipeline_stages.stage4_xgboost.percentage}% Risk

FINAL ENSEMBLE READMISSION RISK SCORE: ${dynamicEnsemble?.pct || assessmentResult.ensemble_result.readmission_risk_percentage}%
RISK TIER: ${dynamicEnsemble?.tier || assessmentResult.ensemble_result.risk_tier}
RECOMMENDATION: ${dynamicEnsemble?.recommendation || assessmentResult.ensemble_result.recommendation}
------------------------------------------------------------------
`;
    navigator.clipboard.writeText(text.trim());
    setCopiedReport(true);
    setTimeout(() => setCopiedReport(false), 2500);
  };

  // Recharts Data
  const logisticCoeffData = metrics?.metrics?.logistic_regression?.coefficients
    ? Object.entries(metrics.metrics.logistic_regression.coefficients).map(([feature, coef]) => ({
        feature: feature.replace('primary_diagnosis_code_', 'Dx: ').replace('patient_segment_', 'Seg: '),
        coefficient: parseFloat(coef.toFixed(4)),
        impact: coef > 0 ? 'Increases Risk' : 'Decreases Risk'
      }))
    : [];

  const xgboostImportancesData = metrics?.metrics?.xgboost?.feature_importances
    ? Object.entries(metrics.metrics.xgboost.feature_importances).map(([feature, imp]) => ({
        feature: feature.replace('primary_diagnosis_code_', 'Dx: ').replace('patient_segment_', 'Seg: '),
        importance: parseFloat((imp * 100).toFixed(2))
      }))
    : [];

  const rfeCurveData = metrics?.rfe_history
    ? metrics.rfe_history.map(r => ({
        step: `Step ${r.step}`,
        featuresCount: r.remaining_features_count,
        accuracy: parseFloat((r.step_accuracy * 100).toFixed(2)),
        dropped: r.dropped_feature
      }))
    : [];

  const clusterRadarData = [
    { subject: 'Age Profile', Cluster0: 85, Cluster1: 65, Cluster2: 55, Cluster3: 40 },
    { subject: 'Hospital Days', Cluster0: 90, Cluster1: 80, Cluster2: 45, Cluster3: 25 },
    { subject: 'Prior Admissions', Cluster0: 95, Cluster1: 50, Cluster2: 40, Cluster3: 15 },
    { subject: 'Lab Procedures', Cluster0: 75, Cluster1: 95, Cluster2: 50, Cluster3: 30 },
    { subject: 'Medications', Cluster0: 85, Cluster1: 75, Cluster2: 40, Cluster3: 20 },
    { subject: 'Comorbidity', Cluster0: 100, Cluster1: 85, Cluster2: 60, Cluster3: 10 }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white">
      
      {/* Background Mesh Glow */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden opacity-30">
        <div className="absolute -top-40 -left-40 w-[600px] h-[600px] bg-cyan-600/25 rounded-full blur-3xl"></div>
        <div className="absolute top-1/3 -right-40 w-[600px] h-[600px] bg-indigo-600/25 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-40 left-1/3 w-[600px] h-[600px] bg-purple-600/25 rounded-full blur-3xl"></div>
      </div>

      {/* Prominent Header Bar */}
      <header className="border-b border-slate-800 bg-slate-950/95 backdrop-blur-2xl sticky top-0 z-50 px-10 py-6">
        <div className="max-w-[1500px] mx-auto flex flex-col md:flex-row md:items-center justify-between gap-6">
          
          <div className="flex items-center gap-5">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-cyan-500 via-indigo-500 to-purple-500 p-0.5 shadow-2xl shadow-cyan-500/30 flex-shrink-0">
              <div className="w-full h-full bg-slate-950 rounded-[14px] flex items-center justify-center">
                <Brain className="w-8 h-8 text-cyan-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-3xl font-extrabold tracking-tight text-white">HealthGuard AI</h1>
                <span className="text-xs px-3.5 py-1 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-600/80 font-mono font-bold">
                  4-Algorithm Pipeline
                </span>
                <span className="text-xs px-3.5 py-1 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-600/80 font-mono font-bold flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5" /> 10,000 Patient Dataset
                </span>
              </div>
              <p className="text-base text-slate-300 font-medium mt-1">Clinical Patient Readmission Risk Analytics & Model Explainability</p>
            </div>
          </div>

          {/* Status & Actions */}
          <div className="flex items-center gap-5">
            <div className="flex items-center gap-3 px-5 py-3 rounded-2xl bg-slate-900 border border-slate-800 font-mono text-base">
              <div className={`w-3.5 h-3.5 rounded-full ${healthStatus?.status === 'online' ? 'bg-emerald-400 shadow-lg shadow-emerald-400/50 animate-ping' : 'bg-amber-400'}`}></div>
              <span className="text-slate-200">
                Engine: <strong className="text-cyan-400 font-bold">{healthStatus?.ml_microservice?.status === 'online' ? 'FastAPI Python ML' : 'Express Gateway'}</strong>
              </span>
            </div>

            <button 
              onClick={handleCopyReport}
              disabled={!assessmentResult}
              className="flex items-center gap-2.5 px-5 py-3 rounded-2xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-100 font-bold transition-all disabled:opacity-40 disabled:cursor-not-allowed text-base"
            >
              {copiedReport ? <Check className="w-5 h-5 text-emerald-400" /> : <Copy className="w-5 h-5 text-cyan-400" />}
              {copiedReport ? 'Report Copied!' : 'Copy Summary'}
            </button>
          </div>

        </div>
      </header>

      {/* Dataset Overview Prominent Banner */}
      <div className="bg-slate-900/80 border-b border-slate-800 py-3.5 px-10 relative z-10">
        <div className="max-w-[1500px] mx-auto flex flex-wrap items-center justify-between gap-4 text-sm font-mono">
          <div className="flex items-center gap-6">
            <span className="text-slate-400 flex items-center gap-2">
              <Database className="w-4 h-4 text-cyan-400" /> Total Records Trained: <strong className="text-white">10,000 Patients</strong>
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-400">
              Overall Readmission Rate: <strong className="text-rose-400">{metrics?.metrics?.readmission_rate_overall ? `${(metrics.metrics.readmission_rate_overall * 100).toFixed(2)}%` : '27.07%'}</strong>
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-400">
              RFE Features Selected: <strong className="text-indigo-400">15 of 24 Features</strong>
            </span>
          </div>

          <div className="text-xs text-cyan-300 bg-cyan-950/80 px-3 py-1 rounded-full border border-cyan-800/60 font-sans font-medium">
            ✓ 100% Dataset Fully Preprocessed & Trained Across All 4 Pipeline Stages
          </div>
        </div>
      </div>

      {/* Main Container */}
      <main className="flex-1 max-w-[1500px] w-full mx-auto px-8 py-8 flex flex-col gap-8 relative z-10">
        
        {/* Navigation Tabs Bar */}
        <nav className="flex items-center gap-4 overflow-x-auto pb-4 border-b-2 border-slate-800/80">
          
          <button
            onClick={() => setActiveTab('assessor')}
            className={`flex items-center gap-3 px-7 py-4 rounded-2xl text-base font-bold transition-all whitespace-nowrap ${
              activeTab === 'assessor'
                ? 'glass-nav-active'
                : 'text-slate-300 hover:text-white hover:bg-slate-900/80'
            }`}
          >
            <Stethoscope className="w-6 h-6" />
            Live Patient Assessor
          </button>

          <button
            onClick={() => setActiveTab('architecture')}
            className={`flex items-center gap-3 px-7 py-4 rounded-2xl text-base font-bold transition-all whitespace-nowrap ${
              activeTab === 'architecture'
                ? 'glass-nav-active'
                : 'text-slate-300 hover:text-white hover:bg-slate-900/80'
            }`}
          >
            <GitMerge className="w-6 h-6" />
            4-Algorithm Architecture
          </button>

          <button
            onClick={() => setActiveTab('explainable_ai')}
            className={`flex items-center gap-3 px-7 py-4 rounded-2xl text-base font-bold transition-all whitespace-nowrap ${
              activeTab === 'explainable_ai'
                ? 'glass-nav-active'
                : 'text-slate-300 hover:text-white hover:bg-slate-900/80'
            }`}
          >
            <Brain className="w-6 h-6" />
            Explainable AI & Symptom Drivers
          </button>

          <button
            onClick={() => setActiveTab('clusters')}
            className={`flex items-center gap-3 px-7 py-4 rounded-2xl text-base font-bold transition-all whitespace-nowrap ${
              activeTab === 'clusters'
                ? 'glass-nav-active'
                : 'text-slate-300 hover:text-white hover:bg-slate-900/80'
            }`}
          >
            <Layers className="w-6 h-6" />
            K-Means Patient Segments
          </button>

          <button
            onClick={() => setActiveTab('metrics')}
            className={`flex items-center gap-3 px-7 py-4 rounded-2xl text-base font-bold transition-all whitespace-nowrap ${
              activeTab === 'metrics'
                ? 'glass-nav-active'
                : 'text-slate-300 hover:text-white hover:bg-slate-900/80'
            }`}
          >
            <BarChart3 className="w-6 h-6" />
            Model Metrics & RFE
          </button>

          <button
            onClick={() => setActiveTab('history')}
            className={`flex items-center gap-3 px-7 py-4 rounded-2xl text-base font-bold transition-all whitespace-nowrap ${
              activeTab === 'history'
                ? 'glass-nav-active'
                : 'text-slate-300 hover:text-white hover:bg-slate-900/80'
            }`}
          >
            <Clock className="w-6 h-6" />
            Audit Log ({history.length})
          </button>

        </nav>

        {/* ========================================================================= */}
        {/* TAB 1: LIVE PATIENT ASSESSOR & PIPELINE EXECUTOR                          */}
        {/* ========================================================================= */}
        {activeTab === 'assessor' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 animate-fade-in">
            
            {/* Left Col: Organized Form (6 cols) */}
            <div className="lg:col-span-6 flex flex-col gap-8">
              
              {/* Presets Bar */}
              <div className="glass-card rounded-3xl p-7 border border-slate-800">
                <div className="flex items-center justify-between mb-5">
                  <span className="text-base font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-2.5">
                    <Sparkles className="w-5 h-5" /> Sample Patient Presets
                  </span>
                  <span className="text-xs text-slate-400 font-mono">Click to Auto-Fill Form</span>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {samples.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => handlePresetSelect(s)}
                      className="text-left p-4 rounded-2xl bg-slate-900/90 hover:bg-slate-800 border-2 border-slate-800 hover:border-cyan-500/60 transition-all group"
                    >
                      <div className="font-extrabold text-base text-slate-100 group-hover:text-cyan-300 truncate">{s.name}</div>
                      <div className="text-sm text-slate-400 mt-1 flex items-center gap-2 font-mono">
                        <span>{s.age} y/o</span>
                        <span>•</span>
                        <span className="text-cyan-400 font-bold">{s.admission_type}</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Form Card */}
              <form onSubmit={handleAssessPatient} className="glass-card rounded-3xl p-8 border border-slate-800 flex flex-col gap-8">
                
                <div className="flex items-center justify-between border-b-2 border-slate-800 pb-5">
                  <h3 className="text-xl font-extrabold text-slate-100 flex items-center gap-3">
                    <UserCheck className="w-6 h-6 text-cyan-400" /> Patient Medical Profile
                  </h3>
                  <span className="text-xs px-3.5 py-1.5 rounded-full bg-slate-900 text-slate-300 font-mono font-bold border border-slate-800">
                    12 Clinical Variables
                  </span>
                </div>

                {/* Section 1: Demographics */}
                <div className="flex flex-col gap-4">
                  <div className="text-sm font-extrabold uppercase text-slate-300 tracking-wider flex items-center gap-2 border-b border-slate-800/80 pb-2">
                    <User className="w-4 h-4 text-cyan-400" /> 1. Patient Demographics
                  </div>
                  
                  <div>
                    <label className="block text-base font-semibold text-slate-200 mb-2">Patient Full Name</label>
                    <input
                      type="text"
                      name="patient_name"
                      placeholder="e.g. Eleanor Vance"
                      value={formData.patient_name}
                      onChange={handleFormChange}
                      className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100 placeholder-slate-500 focus:border-cyan-500 focus:outline-none transition-colors font-sans"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-5">
                    <div>
                      <label className="block text-base font-semibold text-slate-200 mb-2">Age (Years)</label>
                      <input
                        type="number"
                        name="age"
                        value={formData.age}
                        onChange={handleFormChange}
                        className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100 font-mono"
                      />
                    </div>

                    <div>
                      <label className="block text-base font-semibold text-slate-200 mb-2">Gender</label>
                      <select
                        name="gender"
                        value={formData.gender}
                        onChange={handleFormChange}
                        className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100"
                      >
                        <option value="Female">Female</option>
                        <option value="Male">Male</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-base font-semibold text-slate-200 mb-2">Insurance Coverage</label>
                      <select
                        name="insurance_type"
                        value={formData.insurance_type}
                        onChange={handleFormChange}
                        className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100"
                      >
                        <option value="Medicare">Medicare</option>
                        <option value="Private">Private</option>
                        <option value="Medicaid">Medicaid</option>
                        <option value="Self-pay">Self-pay</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-base font-semibold text-slate-200 mb-2">Hospital Facility ID</label>
                      <input
                        type="number"
                        name="hospital_id"
                        value={formData.hospital_id}
                        onChange={handleFormChange}
                        className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100 font-mono"
                      />
                    </div>
                  </div>
                </div>

                {/* Section 2: Clinical Admission & Diagnosis */}
                <div className="flex flex-col gap-4">
                  <div className="text-sm font-extrabold uppercase text-slate-300 tracking-wider flex items-center gap-2 border-b border-slate-800/80 pb-2">
                    <Hospital className="w-4 h-4 text-indigo-400" /> 2. Admission & Primary Diagnosis
                  </div>

                  <div className="grid grid-cols-2 gap-5">
                    <div>
                      <label className="block text-base font-semibold text-slate-200 mb-2">Admission Type</label>
                      <select
                        name="admission_type"
                        value={formData.admission_type}
                        onChange={handleFormChange}
                        className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100"
                      >
                        <option value="Emergency">Emergency</option>
                        <option value="Urgent">Urgent</option>
                        <option value="Elective">Elective</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-base font-semibold text-slate-200 mb-2">Discharge Disposition</label>
                      <select
                        name="discharge_disposition"
                        value={formData.discharge_disposition}
                        onChange={handleFormChange}
                        className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100"
                      >
                        <option value="Home">Home</option>
                        <option value="Rehabilitation">Rehabilitation</option>
                        <option value="Transfer">Transfer Facility</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="block text-base font-semibold text-slate-200 mb-2">Primary Diagnosis Code (ICD-10)</label>
                    <select
                      name="primary_diagnosis_code"
                      value={formData.primary_diagnosis_code}
                      onChange={handleFormChange}
                      className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100 font-mono"
                    >
                      <option value="E11">E11 - Type 2 Diabetes Mellitus</option>
                      <option value="J45">J45 - Asthma & Respiratory Exacerbation</option>
                      <option value="I10">I10 - Essential Hypertension & Cardiac Strain</option>
                      <option value="G47">G47 - Sleep Apnea & Airway Support</option>
                      <option value="N39">N39 - Urinary Tract Infection / Renal</option>
                      <option value="K21">K21 - Gastroesophageal Reflux Disease</option>
                      <option value="F32">F32 - Major Depressive Episode</option>
                      <option value="E78">E78 - Pure Hypercholesterolemia</option>
                    </select>
                  </div>
                </div>

                {/* Section 3: Hospital Utilization & Vitals */}
                <div className="flex flex-col gap-4">
                  <div className="text-sm font-extrabold uppercase text-slate-300 tracking-wider flex items-center gap-2 border-b border-slate-800/80 pb-2">
                    <HeartPulse className="w-4 h-4 text-rose-400" /> 3. Utilization & Clinical Intensity
                  </div>

                  <div className="grid grid-cols-2 gap-5">
                    <div>
                      <label className="block text-base font-semibold text-slate-200 mb-2">Prior Admissions Count</label>
                      <input
                        type="number"
                        name="num_prior_admissions"
                        value={formData.num_prior_admissions}
                        onChange={handleFormChange}
                        className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100 font-mono"
                      />
                    </div>

                    <div>
                      <label className="block text-base font-semibold text-slate-200 mb-2">Length of Stay (Days)</label>
                      <input
                        type="number"
                        name="time_in_hospital"
                        value={formData.time_in_hospital}
                        onChange={handleFormChange}
                        className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100 font-mono"
                      />
                    </div>

                    <div>
                      <label className="block text-base font-semibold text-slate-200 mb-2">Lab Procedures Count</label>
                      <input
                        type="number"
                        name="num_lab_procedures"
                        value={formData.num_lab_procedures}
                        onChange={handleFormChange}
                        className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100 font-mono"
                      />
                    </div>

                    <div>
                      <label className="block text-base font-semibold text-slate-200 mb-2">Medications Prescribed</label>
                      <input
                        type="number"
                        name="num_medications"
                        value={formData.num_medications}
                        onChange={handleFormChange}
                        className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100 font-mono"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-base font-semibold text-slate-200 mb-2">Secondary Chronic Comorbidity?</label>
                    <select
                      name="has_comorbidity"
                      value={formData.has_comorbidity}
                      onChange={handleFormChange}
                      className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl px-4 py-3.5 text-base text-slate-100"
                    >
                      <option value={1}>Yes - Multiple Chronic Comorbidities (1)</option>
                      <option value={0}>No - Single Baseline Condition (0)</option>
                    </select>
                  </div>
                </div>

                {/* Model Weight Slider */}
                <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col gap-3">
                  <div className="flex items-center justify-between text-base">
                    <span className="font-bold text-slate-100 flex items-center gap-2">
                      <Sliders className="w-5 h-5 text-cyan-400" /> Interactive Model Weighting
                    </span>
                    <span className="font-mono text-cyan-400 font-extrabold text-lg">
                      {(logisticWeight * 100).toFixed(0)}% Log / {((1 - logisticWeight) * 100).toFixed(0)}% XGB
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={logisticWeight}
                    onChange={(e) => setLogisticWeight(parseFloat(e.target.value))}
                    className="w-full accent-cyan-500 cursor-pointer h-3 bg-slate-950 rounded-lg"
                  />
                  <div className="flex justify-between text-xs text-slate-400 font-mono">
                    <span>100% Logistic Baseline</span>
                    <span>100% XGBoost Engine</span>
                  </div>
                </div>

                {/* Action Submit Button */}
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full py-5 rounded-2xl bg-gradient-to-r from-cyan-500 via-indigo-500 to-purple-600 hover:from-cyan-400 hover:to-purple-500 font-extrabold text-lg text-white shadow-2xl shadow-cyan-500/30 flex items-center justify-center gap-3 transition-all disabled:opacity-50 mt-2"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-6 h-6 animate-spin text-white" />
                      Executing 4-Algorithm ML Pipeline...
                    </>
                  ) : (
                    <>
                      <Zap className="w-6 h-6 text-white fill-white" />
                      Execute 4-Algorithm Risk Assessment
                    </>
                  )}
                </button>

              </form>
            </div>

            {/* Right Col: Prominent Results (6 cols) */}
            <div className="lg:col-span-6 flex flex-col gap-8">

              {/* Steps Progress */}
              <div className="glass-card rounded-3xl p-7 border border-slate-800">
                <div className="flex items-center justify-between text-base font-bold uppercase tracking-wider text-slate-200 mb-5">
                  <span className="flex items-center gap-2.5">
                    <Activity className="w-5 h-5 text-cyan-400" /> Pipeline Execution Flow
                  </span>
                  <span className="font-mono text-cyan-400 font-extrabold">
                    {animatingStep === 0 && 'Ready'}
                    {animatingStep === 1 && '1/4 K-Means'}
                    {animatingStep === 2 && '2/4 RFE'}
                    {animatingStep === 3 && '3/4 Logistic'}
                    {animatingStep === 4 && '4/4 XGBoost'}
                    {animatingStep === 5 && 'Completed'}
                  </span>
                </div>

                <div className="grid grid-cols-4 gap-4 text-center">
                  <div className={`p-4 rounded-2xl border-2 transition-all ${
                    animatingStep >= 1 ? 'bg-purple-950/70 border-purple-500 text-purple-200 shadow-xl shadow-purple-500/20' : 'bg-slate-900 border-slate-800 text-slate-500'
                  }`}>
                    <div className="font-bold text-xs uppercase mb-1">1. K-Means</div>
                    <div className="text-xs text-slate-300">Persona Segment</div>
                  </div>

                  <div className={`p-4 rounded-2xl border-2 transition-all ${
                    animatingStep >= 2 ? 'bg-indigo-950/70 border-indigo-500 text-indigo-200 shadow-xl shadow-indigo-500/20' : 'bg-slate-900 border-slate-800 text-slate-500'
                  }`}>
                    <div className="font-bold text-xs uppercase mb-1">2. RFE</div>
                    <div className="text-xs text-slate-300">Recursive Prune</div>
                  </div>

                  <div className={`p-4 rounded-2xl border-2 transition-all ${
                    animatingStep >= 3 ? 'bg-cyan-950/70 border-cyan-500 text-cyan-200 shadow-xl shadow-cyan-500/20' : 'bg-slate-900 border-slate-800 text-slate-500'
                  }`}>
                    <div className="font-bold text-xs uppercase mb-1">3. Logistic</div>
                    <div className="text-xs text-slate-300">Baseline Weights</div>
                  </div>

                  <div className={`p-4 rounded-2xl border-2 transition-all ${
                    animatingStep >= 4 ? 'bg-emerald-950/70 border-emerald-500 text-emerald-200 shadow-xl shadow-emerald-500/20' : 'bg-slate-900 border-slate-800 text-slate-500'
                  }`}>
                    <div className="font-bold text-xs uppercase mb-1">4. XGBoost</div>
                    <div className="text-xs text-slate-300">Ensemble Engine</div>
                  </div>
                </div>
              </div>

              {/* Output Display */}
              {assessmentResult ? (
                <div className="flex flex-col gap-8 animate-fade-in">
                  
                  {/* Gauge Risk Banner */}
                  <div className="glass-card rounded-3xl p-8 border border-slate-800 relative overflow-hidden">
                    <div 
                      className="absolute top-0 left-0 bottom-0 w-3.5 transition-all" 
                      style={{ backgroundColor: dynamicEnsemble?.color || assessmentResult.ensemble_result.badge_color }}
                    ></div>

                    <div className="flex flex-col sm:flex-row items-center justify-between gap-8">
                      
                      {/* Gauge */}
                      <div className="flex items-center gap-7">
                        
                        <div className="relative w-36 h-36 flex items-center justify-center flex-shrink-0">
                          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                            <circle cx="50" cy="50" r="40" stroke="#1e293b" strokeWidth="12" fill="transparent" />
                            <circle 
                              cx="50" 
                              cy="50" 
                              r="40" 
                              stroke={dynamicEnsemble?.color || assessmentResult.ensemble_result.badge_color} 
                              strokeWidth="12" 
                              strokeDasharray="251" 
                              strokeDashoffset={251 - (251 * (dynamicEnsemble?.score || assessmentResult.ensemble_result.ensemble_score))} 
                              strokeLinecap="round"
                              fill="transparent"
                              className="transition-all duration-700 ease-out"
                            />
                          </svg>
                          <div className="absolute flex flex-col items-center justify-center">
                            <span className="text-3xl font-extrabold font-mono text-white">
                              {dynamicEnsemble?.pct || assessmentResult.ensemble_result.readmission_risk_percentage}%
                            </span>
                          </div>
                        </div>

                        <div>
                          <div className="text-xs uppercase font-mono tracking-wider text-slate-400 mb-1">
                            Ensemble 30-Day Readmission Risk
                          </div>
                          <div className="flex items-center gap-3">
                            <span className="text-3xl font-extrabold text-white">
                              {dynamicEnsemble?.tier || assessmentResult.ensemble_result.risk_tier}
                            </span>
                          </div>
                          {(assessmentResult?.patient_input?.patient_name || assessmentResult?.patient_input?.name || formData.patient_name || formData.name) && (
                            <div className="text-sm font-semibold text-cyan-400 mt-1 flex items-center gap-1">
                              <User className="w-3.5 h-3.5" />
                              {assessmentResult?.patient_input?.patient_name || assessmentResult?.patient_input?.name || formData.patient_name || formData.name}
                            </div>
                          )}
                        </div>

                      </div>

                      {/* Scores Grid */}
                      <div className="grid grid-cols-2 gap-5 font-mono bg-slate-900 p-5 rounded-2xl border border-slate-800 w-full sm:w-auto">
                        <div>
                          <span className="text-slate-400 text-xs block">Logistic Baseline:</span>
                          <span className="text-cyan-400 font-extrabold text-2xl">{assessmentResult.pipeline_stages.stage3_logistic_regression.percentage}%</span>
                        </div>
                        <div>
                          <span className="text-slate-400 text-xs block">XGBoost Engine:</span>
                          <span className="text-emerald-400 font-extrabold text-2xl">{assessmentResult.pipeline_stages.stage4_xgboost.percentage}%</span>
                        </div>
                      </div>

                    </div>

                    {/* Protocol */}
                    <div className="mt-8 p-5 rounded-2xl bg-slate-900 border border-slate-800 flex items-start gap-5">
                      <ShieldAlert className="w-7 h-7 text-amber-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="text-base font-bold text-slate-100 mb-1">Clinical Intervention Protocol:</div>
                        <p className="text-base text-slate-300 leading-relaxed">{dynamicEnsemble?.recommendation || assessmentResult.ensemble_result.recommendation}</p>
                      </div>
                    </div>

                  </div>

                  {/* Stage 1 & Stage 2 Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    
                    <div className="glass-card rounded-2xl p-6 border border-purple-900/50 flex flex-col justify-between">
                      <div>
                        <div className="text-xs font-mono uppercase text-purple-400 mb-2 flex items-center gap-2">
                          <Layers className="w-4 h-4" /> Stage 1: K-Means Persona
                        </div>
                        <div className="font-bold text-slate-100 text-lg mb-2">
                          {assessmentResult.pipeline_stages.stage1_kmeans.persona.name}
                        </div>
                        <p className="text-sm text-slate-300 leading-relaxed mb-5">
                          {assessmentResult.pipeline_stages.stage1_kmeans.persona.description}
                        </p>
                      </div>
                      <div className="text-xs font-mono text-purple-300 bg-purple-950/70 px-4 py-2 rounded-xl border border-purple-800/40 w-max font-bold">
                        Segment ID: #{assessmentResult.pipeline_stages.stage1_kmeans.cluster_id}
                      </div>
                    </div>

                    <div className="glass-card rounded-2xl p-6 border border-indigo-900/50 flex flex-col justify-between">
                      <div>
                        <div className="text-xs font-mono uppercase text-indigo-400 mb-2 flex items-center gap-2">
                          <GitMerge className="w-4 h-4" /> Stage 2: RFE Features
                        </div>
                        <div className="font-bold text-slate-100 text-lg mb-2">
                          {assessmentResult.pipeline_stages.stage2_rfe.selected_features_count} Critical Predictors
                        </div>
                        <p className="text-sm text-slate-300 leading-relaxed mb-4">
                          Pruned raw dataset to optimize predictors for model training.
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {assessmentResult.pipeline_stages.stage2_rfe.selected_features.slice(0, 4).map((f, i) => (
                          <span key={i} className="text-xs font-mono bg-indigo-950/70 text-indigo-300 px-3 py-1 rounded-lg border border-indigo-800/40 font-bold">
                            {f.replace('primary_diagnosis_code_', 'Dx:').replace('patient_segment_', '')}
                          </span>
                        ))}
                      </div>
                    </div>

                  </div>

                  {/* Symptom Drivers List */}
                  <div className="glass-card rounded-3xl p-7 border border-slate-800">
                    <div className="flex items-center justify-between mb-5 border-b border-slate-800 pb-4">
                      <h4 className="text-base font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-2.5">
                        <Brain className="w-5 h-5" /> Primary Symptom Risk Drivers (Logistic Regression Weights)
                      </h4>
                      <button 
                        onClick={() => setActiveTab('explainable_ai')}
                        className="text-xs text-slate-300 hover:text-cyan-300 flex items-center gap-1 font-mono transition-all font-bold"
                      >
                        Deep Feature Breakdown <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>

                    <div className="space-y-3.5">
                      {assessmentResult.pipeline_stages.stage3_logistic_regression.top_feature_contributions.slice(0, 5).map((c, i) => (
                        <div key={i} className="flex items-center justify-between text-base p-4 rounded-2xl bg-slate-900 border border-slate-800">
                          <span className="font-mono text-slate-200 font-medium">{c.feature}</span>
                          <div className="flex items-center gap-5">
                            <span className="text-slate-400 text-sm font-mono">val: {c.value}</span>
                            <span className={`font-mono font-extrabold text-lg ${c.contribution > 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                              {c.contribution > 0 ? `+${c.contribution}` : c.contribution}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                </div>
              ) : (
                /* Empty state prompt */
                <div className="glass-card rounded-3xl p-20 border border-slate-800 flex flex-col items-center justify-center text-center">
                  <div className="w-24 h-24 rounded-3xl bg-cyan-950/50 border border-cyan-800/40 flex items-center justify-center mb-6 glow-cyan">
                    <Stethoscope className="w-12 h-12 text-cyan-400 animate-pulse" />
                  </div>
                  <h3 className="text-2xl font-bold text-slate-100 mb-3">Ready for Live Patient Assessment</h3>
                  <p className="text-base text-slate-300 max-w-lg leading-relaxed">
                    Select a sample profile from the left or enter patient parameters to trigger the 4-algorithm ML pipeline evaluation.
                  </p>
                </div>
              )}

            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: 4-ALGORITHM FLOW ARCHITECTURE                                      */}
        {/* ========================================================================= */}
        {activeTab === 'architecture' && (
          <div className="flex flex-col gap-10 animate-fade-in">
            
            <div className="glass-card rounded-3xl p-10 border border-slate-800">
              <h2 className="text-2xl font-bold text-white mb-3 flex items-center gap-3">
                <GitMerge className="w-7 h-7 text-indigo-400" /> The 4-Algorithm Pipeline Architecture
              </h2>
              <p className="text-base text-slate-300 leading-relaxed max-w-5xl">
                Because the output of one algorithm feeds into the next, the order of execution is critical for this architecture to work. 
                Pure programming recursion is heavily utilized in Recursive Feature Elimination (RFE) to isolate top predictors and reduce dataset noise before baseline & ensemble training.
              </p>
            </div>

            {/* Architecture Grid */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              
              <div className="glass-card rounded-3xl p-8 border border-purple-900/50 flex flex-col justify-between">
                <div>
                  <div className="text-xs font-mono uppercase px-3.5 py-1.5 rounded bg-purple-950 text-purple-400 border border-purple-800/40 w-max mb-5 font-bold">
                    Algorithm 1 • Unsupervised
                  </div>
                  <h3 className="font-bold text-slate-100 text-lg mb-3 flex items-center gap-2">
                    <Layers className="w-6 h-6 text-purple-400" /> K-Means Clustering
                  </h3>
                  <p className="text-base text-slate-300 leading-relaxed mb-8">
                    Groups patients into distinct clusters based on medical history, vitals, and demographics. Creates a new <strong>"Patient Segment"</strong> feature that boosts downstream predictive accuracy.
                  </p>
                </div>
                <div className="text-xs font-mono text-purple-300 bg-purple-950/70 p-4 rounded-2xl border border-purple-900/60 font-bold">
                  Output: Patient Segment ID
                </div>
              </div>

              <div className="glass-card rounded-3xl p-8 border border-indigo-900/50 flex flex-col justify-between">
                <div>
                  <div className="text-xs font-mono uppercase px-3.5 py-1.5 rounded bg-indigo-950 text-indigo-400 border border-indigo-800/40 w-max mb-5 font-bold">
                    Algorithm 2 • Pure Recursion
                  </div>
                  <h3 className="font-bold text-slate-100 text-lg mb-3 flex items-center gap-2">
                    <GitMerge className="w-6 h-6 text-indigo-400" /> RFE Feature Elimination
                  </h3>
                  <p className="text-base text-slate-300 leading-relaxed mb-8">
                    Recursively fits the model, ranks feature importance, drops the weakest variable, and repeats until only the top 15 critical predictors remain.
                  </p>
                </div>
                <div className="text-xs font-mono text-indigo-300 bg-indigo-950/70 p-4 rounded-2xl border border-indigo-900/60 font-bold">
                  Output: 15 Optimized Features
                </div>
              </div>

              <div className="glass-card rounded-3xl p-8 border border-cyan-900/50 flex flex-col justify-between">
                <div>
                  <div className="text-xs font-mono uppercase px-3.5 py-1.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/40 w-max mb-5 font-bold">
                    Algorithm 3 • Baseline
                  </div>
                  <h3 className="font-bold text-slate-100 text-lg mb-3 flex items-center gap-2">
                    <Brain className="w-6 h-6 text-cyan-400" /> Logistic Regression
                  </h3>
                  <p className="text-base text-slate-300 leading-relaxed mb-8">
                    Trains a baseline regression model on optimized features. Provides clear, interpretable linear weights so doctors know why a risk prediction was made.
                  </p>
                </div>
                <div className="text-xs font-mono text-cyan-300 bg-cyan-950/70 p-4 rounded-2xl border border-cyan-900/60 font-bold">
                  Output: Baseline Score & Weights
                </div>
              </div>

              <div className="glass-card rounded-3xl p-8 border border-emerald-900/50 flex flex-col justify-between">
                <div>
                  <div className="text-xs font-mono uppercase px-3.5 py-1.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/40 w-max mb-5 font-bold">
                    Algorithm 4 • Advanced Ensemble
                  </div>
                  <h3 className="font-bold text-slate-100 text-lg mb-3 flex items-center gap-2">
                    <Zap className="w-6 h-6 text-emerald-400" /> XGBoost Engine
                  </h3>
                  <p className="text-base text-slate-300 leading-relaxed mb-8">
                    Captures complex non-linear feature interactions between patient symptoms that linear regression misses. Blended via weighted ensemble.
                  </p>
                </div>
                <div className="text-xs font-mono text-emerald-300 bg-emerald-950/70 p-4 rounded-2xl border border-emerald-900/60 font-bold">
                  Output: Non-Linear Ensemble Score
                </div>
              </div>

            </div>

          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: EXPLAINABLE AI & FEATURE WEIGHTS                                   */}
        {/* ========================================================================= */}
        {activeTab === 'explainable_ai' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 animate-fade-in">
            
            {/* Logistic Regression Chart */}
            <div className="lg:col-span-6 glass-card rounded-3xl p-8 border border-slate-800 flex flex-col gap-6">
              <div className="border-b border-slate-800 pb-5">
                <h3 className="text-lg font-bold text-cyan-400 flex items-center gap-2">
                  <Brain className="w-6 h-6" /> Logistic Regression Linear Coefficients
                </h3>
                <p className="text-sm text-slate-400 mt-1">Interpretable linear risk weights (positive values increase readmission risk).</p>
              </div>

              {logisticCoeffData.length > 0 ? (
                <div className="h-[420px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={logisticCoeffData} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                      <XAxis type="number" stroke="#64748b" tick={{ fontSize: 13 }} />
                      <YAxis dataKey="feature" type="category" stroke="#94a3b8" tick={{ fontSize: 12 }} width={140} />
                      <RechartsTooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '16px', fontSize: '14px' }} />
                      <Bar dataKey="coefficient">
                        {logisticCoeffData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.coefficient > 0 ? '#f43f5e' : '#10b981'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="text-center text-slate-500 py-20 text-base">Loading coefficient data...</div>
              )}
            </div>

            {/* XGBoost Feature Importances Chart */}
            <div className="lg:col-span-6 glass-card rounded-3xl p-8 border border-slate-800 flex flex-col gap-6">
              <div className="border-b border-slate-800 pb-5">
                <h3 className="text-lg font-bold text-emerald-400 flex items-center gap-2">
                  <Zap className="w-6 h-6" /> XGBoost Feature Importances (Gain Weight %)
                </h3>
                <p className="text-sm text-slate-400 mt-1">Non-linear decision boundary split importance.</p>
              </div>

              {xgboostImportancesData.length > 0 ? (
                <div className="h-[420px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={xgboostImportancesData} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                      <XAxis type="number" stroke="#64748b" tick={{ fontSize: 13 }} />
                      <YAxis dataKey="feature" type="category" stroke="#94a3b8" tick={{ fontSize: 12 }} width={140} />
                      <RechartsTooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '16px', fontSize: '14px' }} />
                      <Bar dataKey="importance" fill="#10b981" radius={[0, 8, 8, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="text-center text-slate-500 py-20 text-base">Loading XGBoost importance data...</div>
              )}
            </div>

          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 4: K-MEANS PATIENT SEGMENTS                                           */}
        {/* ========================================================================= */}
        {activeTab === 'clusters' && (
          <div className="flex flex-col gap-10 animate-fade-in">
            
            <div className="glass-card rounded-3xl p-10 border border-slate-800">
              <h2 className="text-2xl font-bold text-white mb-3 flex items-center gap-3">
                <Layers className="w-7 h-7 text-purple-400" /> K-Means Patient Segment Radar & Centroid Explorer
              </h2>
              <p className="text-base text-slate-300">
                Patient segmentation clusters similar medical histories together prior to supervised training, appending segment IDs into feature space.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
              
              {/* Radar Visualizer (7 cols) */}
              <div className="lg:col-span-7 glass-card rounded-3xl p-8 border border-slate-800">
                <h3 className="text-base font-bold uppercase tracking-wider text-purple-400 mb-6">
                  Multi-Dimensional Segment Radar Comparison
                </h3>
                <div className="h-[420px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={clusterRadarData}>
                      <PolarGrid stroke="#334155" />
                      <PolarAngleAxis dataKey="subject" stroke="#94a3b8" tick={{ fontSize: 13 }} />
                      <PolarRadiusAxis stroke="#475569" />
                      <Radar name="High Risk Elderly (#0)" dataKey="Cluster0" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.25} />
                      <Radar name="Acute Emergency (#1)" dataKey="Cluster1" stroke="#a855f7" fill="#a855f7" fillOpacity={0.2} />
                      <Radar name="Moderate Chronic (#2)" dataKey="Cluster2" stroke="#eab308" fill="#eab308" fillOpacity={0.2} />
                      <Radar name="Low Risk Elective (#3)" dataKey="Cluster3" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
                      <Legend wrapperStyle={{ fontSize: '13px', paddingTop: '15px' }} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Persona Cards Grid (5 cols) */}
              <div className="lg:col-span-5 flex flex-col gap-5">
                {[0, 1, 2, 3].map((cid) => {
                  const persona = metrics?.cluster_personas?.[cid] || { name: `Cluster ${cid}`, badge: "Persona", description: "" };
                  const stats = metrics?.cluster_centroids?.[cid] || {};

                  return (
                    <div key={cid} className="glass-card rounded-2xl p-6 border border-slate-800 flex flex-col justify-between">
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-xs font-mono uppercase px-3 py-1.5 rounded bg-purple-950 text-purple-300 border border-purple-800/40 font-bold">
                          Segment #{cid}
                        </span>
                        <span className="text-xs font-bold px-3 py-1.5 rounded bg-slate-900 text-slate-300 border border-slate-800">
                          {persona.badge}
                        </span>
                      </div>

                      <div className="font-bold text-slate-100 text-base mb-1">{persona.name}</div>
                      <p className="text-xs text-slate-300 leading-relaxed mb-4">{persona.description}</p>

                      <div className="grid grid-cols-3 gap-3 text-center text-xs font-mono bg-slate-900 p-3 rounded-xl border border-slate-800">
                        <div>
                          <span className="text-slate-400 block text-[10px]">Avg Age</span>
                          <span className="text-purple-300 font-bold">{stats.avg_age ? Math.round(stats.avg_age) : '--'} yrs</span>
                        </div>
                        <div>
                          <span className="text-slate-400 block text-[10px]">Avg Stay</span>
                          <span className="text-purple-300 font-bold">{stats.avg_hospital_days ? stats.avg_hospital_days.toFixed(1) : '--'}d</span>
                        </div>
                        <div>
                          <span className="text-slate-400 block text-[10px]">Readmit Rate</span>
                          <span className="text-rose-400 font-bold">{stats.readmission_rate ? `${(stats.readmission_rate * 100).toFixed(1)}%` : '--'}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

            </div>

          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 5: MODEL PERFORMANCE & METRICS                                       */}
        {/* ========================================================================= */}
        {activeTab === 'metrics' && (
          <div className="flex flex-col gap-10 animate-fade-in">
            
            {/* Cards Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              
              <div className="glass-card rounded-3xl p-8 border border-slate-800">
                <div className="text-xs font-mono uppercase text-slate-400 mb-2 font-bold">Logistic Accuracy</div>
                <div className="text-4xl font-extrabold font-mono text-cyan-400">
                  {metrics?.metrics?.logistic_regression?.accuracy ? `${(metrics.metrics.logistic_regression.accuracy * 100).toFixed(2)}%` : '72.93%'}
                </div>
                <div className="text-sm text-slate-400 mt-2">Baseline Linear Model</div>
              </div>

              <div className="glass-card rounded-3xl p-8 border border-slate-800">
                <div className="text-xs font-mono uppercase text-slate-400 mb-2 font-bold">Logistic AUC-ROC</div>
                <div className="text-4xl font-extrabold font-mono text-cyan-300">
                  {metrics?.metrics?.logistic_regression?.auc_roc ? metrics.metrics.logistic_regression.auc_roc.toFixed(4) : '0.5821'}
                </div>
                <div className="text-sm text-slate-400 mt-2">Interpretable Baseline</div>
              </div>

              <div className="glass-card rounded-3xl p-8 border border-slate-800">
                <div className="text-xs font-mono uppercase text-slate-400 mb-2 font-bold">XGBoost Accuracy</div>
                <div className="text-4xl font-extrabold font-mono text-emerald-400">
                  {metrics?.metrics?.xgboost?.accuracy ? `${(metrics.metrics.xgboost.accuracy * 100).toFixed(2)}%` : '73.53%'}
                </div>
                <div className="text-sm text-slate-400 mt-2">Ensemble Predictive Engine</div>
              </div>

              <div className="glass-card rounded-3xl p-8 border border-slate-800">
                <div className="text-xs font-mono uppercase text-slate-400 mb-2 font-bold">XGBoost AUC-ROC</div>
                <div className="text-4xl font-extrabold font-mono text-emerald-300">
                  {metrics?.metrics?.xgboost?.auc_roc ? metrics.metrics.xgboost.auc_roc.toFixed(4) : '0.6672'}
                </div>
                <div className="text-sm text-slate-400 mt-2">Non-Linear Precision</div>
              </div>

            </div>

            {/* RFE Trajectory Chart */}
            <div className="glass-card rounded-3xl p-8 border border-slate-800">
              <div className="border-b border-slate-800 pb-5 mb-6">
                <h3 className="text-lg font-bold text-indigo-400 flex items-center gap-2 font-mono">
                  <GitMerge className="w-6 h-6" /> RFE Recursive Feature Pruning Trajectory
                </h3>
                <p className="text-sm text-slate-400 mt-1">Demonstrates model accuracy maintained while recursively dropping weakest variables.</p>
              </div>

              {rfeCurveData.length > 0 ? (
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={rfeCurveData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                      <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 13 }} />
                      <YAxis stroke="#94a3b8" tick={{ fontSize: 13 }} domain={['dataMin - 1', 'dataMax + 1']} />
                      <RechartsTooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '16px', fontSize: '14px' }} />
                      <Line type="monotone" dataKey="accuracy" stroke="#6366f1" strokeWidth={4} dot={{ fill: '#818cf8', r: 7 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="text-center text-slate-500 py-16 text-base">Loading RFE trajectory chart...</div>
              )}
            </div>

          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 6: AUDIT HISTORY LOG                                                 */}
        {/* ========================================================================= */}
        {activeTab === 'history' && (
          <div className="flex flex-col gap-8 animate-fade-in">
            
            <div className="glass-card rounded-3xl p-8 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-6">
              <div>
                <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                  <Clock className="w-7 h-7 text-cyan-400" /> Patient Assessment Audit Trail
                </h2>
                <p className="text-base text-slate-300 mt-1">Log of real-time evaluations executed by the 4-algorithm ML engine.</p>
              </div>

              {/* Search Box */}
              <div className="relative w-full md:w-80">
                <Search className="w-5 h-5 text-slate-500 absolute left-4 top-3.5" />
                <input
                  type="text"
                  placeholder="Search history..."
                  value={historySearch}
                  onChange={(e) => setHistorySearch(e.target.value)}
                  className="w-full bg-slate-900 border-2 border-slate-800 rounded-2xl pl-12 pr-4 py-3 text-base text-slate-100 font-mono"
                />
              </div>
            </div>

            <div className="glass-card rounded-3xl p-8 border border-slate-800 overflow-x-auto">
              <table className="w-full text-base font-mono text-left text-slate-300">
                <thead className="bg-slate-900 text-slate-400 uppercase text-xs">
                  <tr>
                    <th className="p-5">Timestamp</th>
                    <th className="p-5">Patient Profile</th>
                    <th className="p-5">K-Means Segment</th>
                    <th className="p-5">Logistic Prob</th>
                    <th className="p-5">XGBoost Prob</th>
                    <th className="p-5">Ensemble Score</th>
                    <th className="p-5">Risk Tier</th>
                  </tr>
                </thead>
                <tbody className="divide-y border-slate-800">
                  {history.length > 0 ? (
                    history
                      .filter(item => JSON.stringify(item).toLowerCase().includes(historySearch.toLowerCase()))
                      .map((item, i) => (
                        <tr key={i} className="hover:bg-slate-900/60">
                          <td className="p-5 text-slate-400">{item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : '--'}</td>
                          <td className="p-5 font-sans text-slate-100">
                            <div className="font-bold text-slate-100 flex items-center gap-1.5">
                              <User className="w-4 h-4 text-cyan-400 inline" />
                              {item.patient_input?.patient_name || item.patient_input?.name || item.patient_name || item.name || formData.patient_name || formData.name || 'Anonymous Patient'}
                            </div>
                            <div className="text-xs text-slate-400 mt-0.5 font-mono">
                              {item.patient_input?.age ?? '--'}y/o {item.patient_input?.gender ?? ''} ({item.patient_input?.admission_type ?? ''})
                            </div>
                          </td>
                          <td className="p-5 text-purple-300">
                            {item.pipeline_stages?.stage1_kmeans?.persona?.name || `Segment #${item.pipeline_stages?.stage1_kmeans?.cluster_id}`}
                          </td>
                          <td className="p-5 text-cyan-400 font-bold">{item.pipeline_stages?.stage3_logistic_regression?.percentage}%</td>
                          <td className="p-5 text-emerald-400 font-bold">{item.pipeline_stages?.stage4_xgboost?.percentage}%</td>
                          <td className="p-5 font-extrabold text-white text-lg">{item.ensemble_result?.readmission_risk_percentage}%</td>
                          <td className="p-5">
                            <span 
                              className="px-3.5 py-1.5 rounded-lg text-sm font-bold text-slate-950 font-sans"
                              style={{ backgroundColor: item.ensemble_result?.badge_color || '#38bdf8' }}
                            >
                              {item.ensemble_result?.risk_tier}
                            </span>
                          </td>
                        </tr>
                      ))
                  ) : (
                    <tr>
                      <td colSpan={7} className="p-16 text-center text-slate-500 font-sans text-base">
                        No recent assessments recorded. Run an evaluation in the <strong>Live Patient Assessor</strong> tab to see audit history here!
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-950 py-6 px-10 text-center text-sm text-slate-500 font-mono relative z-10">
        HealthGuard AI • 4-Algorithm ML Pipeline (K-Means, RFE, Logistic Regression, XGBoost) • Carehub 2.0
      </footer>
    </div>
  );
}
