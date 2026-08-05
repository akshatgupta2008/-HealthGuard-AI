"""
HealthGuard AI: Automated PyTest Suite for ML Engine & FastAPI Microservice
----------------------------------------------------------------------------
"""

import os
import sys
import pytest
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import FourAlgorithmPipeline, CLUSTER_PERSONAS, DEFAULT_DATASET_PATH
from src.data_processing import load_and_preprocess_raw_data
from src.api import app

@pytest.fixture(scope="session")
def sample_dataset():
    if not DEFAULT_DATASET_PATH.exists():
        pytest.skip(f"Dataset not found at {DEFAULT_DATASET_PATH}")
    df = pd.read_csv(DEFAULT_DATASET_PATH)
    return df.head(500)

@pytest.fixture(scope="session")
def trained_pipeline(sample_dataset):
    pipeline = FourAlgorithmPipeline(dataset_path=str(DEFAULT_DATASET_PATH), n_clusters=4, n_rfe_features=15)
    pipeline.train(sample_dataset)
    return pipeline

def test_data_ingestion_and_preprocessing(sample_dataset):
    X_raw, y = load_and_preprocess_raw_data(sample_dataset)
    assert X_raw is not None
    assert y is not None
    assert len(X_raw) == len(y)
    assert 'readmitted_within_30days' not in X_raw.columns

def test_four_algorithm_pipeline_training(trained_pipeline):
    assert trained_pipeline.is_trained is True
    assert trained_pipeline.kmeans_model is not None
    assert trained_pipeline.rfe_selector is not None
    assert trained_pipeline.logistic_model is not None
    assert trained_pipeline.xgboost_model is not None
    assert len(trained_pipeline.rfe_selected_feature_names) == 15
    assert "logistic_regression" in trained_pipeline.metrics
    assert "xgboost" in trained_pipeline.metrics
