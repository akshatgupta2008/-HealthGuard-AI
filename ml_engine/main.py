"""
HealthGuard AI: ML Engine FastAPI Proxy (Redirecting to src.api)
----------------------------------------------------------------
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.api import app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
