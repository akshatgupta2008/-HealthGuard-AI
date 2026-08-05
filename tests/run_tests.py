"""
HealthGuard AI: Automated Data Science Test Suite Runner
---------------------------------------------------------
Executes unit & integration tests for ML Pipeline package (`src.model`).
"""

import os
import sys
import unittest
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import FourAlgorithmPipeline, CLUSTER_PERSONAS, DEFAULT_DATASET_PATH

class TestHealthGuardPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEFAULT_DATASET_PATH.exists():
            raise unittest.SkipTest(f"Dataset not found at {DEFAULT_DATASET_PATH}")
        cls.df = pd.read_csv(DEFAULT_DATASET_PATH).head(500)
        cls.pipeline = FourAlgorithmPipeline(dataset_path=str(DEFAULT_DATASET_PATH), n_clusters=4, n_rfe_features=15)
        cls.pipeline.train(cls.df)

    def test_01_data_ingestion_and_preprocessing(self):
        from src.data_processing import load_and_preprocess_raw_data
        X_raw, y = load_and_preprocess_raw_data(self.df)
        self.assertIsNotNone(X_raw)
        self.assertIsNotNone(y)
        self.assertEqual(len(X_raw), len(y))
        self.assertNotIn('readmitted_within_30days', X_raw.columns)

    def test_02_four_algorithm_pipeline_training(self):
        self.assertTrue(self.pipeline.is_trained)
        self.assertIsNotNone(self.pipeline.kmeans_model)
        self.assertIsNotNone(self.pipeline.rfe_selector)
        self.assertIsNotNone(self.pipeline.logistic_model)
        self.assertIsNotNone(self.pipeline.xgboost_model)
        self.assertEqual(len(self.pipeline.rfe_selected_feature_names), 15)
        self.assertIn("logistic_regression", self.pipeline.metrics)
        self.assertIn("xgboost", self.pipeline.metrics)

    def test_03_single_patient_prediction(self):
        patient_sample = {
            "patient_name": "Test Patient",
            "age": 75,
            "gender": "Female",
            "admission_type": "Emergency",
            "primary_diagnosis_code": "I10",
            "num_prior_admissions": 4,
            "time_in_hospital": 7,
            "num_lab_procedures": 50.0,
            "num_medications": 18.0,
            "has_comorbidity": 1,
            "discharge_disposition": "Transfer",
            "insurance_type": "Medicare",
            "hospital_id": 101
        }
        res = self.pipeline.predict_patient(patient_sample)
        self.assertIn("pipeline_stages", res)
        self.assertIn("ensemble_result", res)
        self.assertIn(res["ensemble_result"]["risk_tier"], ["Critical Risk", "High Risk", "Moderate Risk", "Low Risk"])
        self.assertTrue(0.0 <= res["ensemble_result"]["ensemble_score"] <= 1.0)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHealthGuardPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
