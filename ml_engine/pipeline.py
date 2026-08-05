"""
HealthGuard AI: ML Engine Module Proxy (Redirecting to src.model)
------------------------------------------------------------------
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import FourAlgorithmPipeline, CLUSTER_PERSONAS
from src.data_processing import load_and_preprocess_raw_data

__all__ = ["FourAlgorithmPipeline", "CLUSTER_PERSONAS", "load_and_preprocess_raw_data"]
