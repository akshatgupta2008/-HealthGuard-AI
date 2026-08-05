"""Project-local Python startup tweaks.

This removes Google Cloud SDK paths that are injected through PYTHONPATH and
can shadow the packages installed in the project virtual environment.
"""

from __future__ import annotations

import os
import sys


def _remove_google_cloud_sdk_paths() -> None:
    blocked = {
        os.path.normcase(r"C:\Users\SWIFT LITE 14\AppData\Local\Google\google-cloud-sdk\lib"),
        os.path.normcase(r"C:\Users\SWIFT LITE 14\AppData\Local\Google\google-cloud-sdk\lib\third_party"),
    }

    sys.path[:] = [path for path in sys.path if os.path.normcase(path) not in blocked]


_remove_google_cloud_sdk_paths()