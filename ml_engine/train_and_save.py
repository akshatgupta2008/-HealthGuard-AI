"""
HealthGuard AI: ML Engine Training Script Proxy (Redirecting to src.train_and_evaluate)
----------------------------------------------------------------------------------------
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.train_and_evaluate import run_ml_benchmark

if __name__ == '__main__':
    run_ml_benchmark()
