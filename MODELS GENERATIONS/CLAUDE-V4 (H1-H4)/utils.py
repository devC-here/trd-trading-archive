# utils.py
# Shared utilities for all deploy scripts.
# Handles file path resolution whether running:
#   • directly from source (python deploy_momentum_h1.py)
#   • frozen as a PyInstaller/Nuitka executable
#   • installed to a custom directory via --base-dir

import os
import sys

# Optional: override at runtime with env var  MODEL_DIR=/path/to/weights
_ENV_BASE = os.environ.get("MODEL_DIR", "").strip()

def resource_path(relative_path: str) -> str:
    """
    Resolve a relative filename to an absolute path.

    Resolution order:
      1. MODEL_DIR env var  (export MODEL_DIR=/path/to/weights)
      2. PyInstaller _MEIPASS temp dir  (frozen exe)
      3. Directory of the calling script / sys.argv[0]
      4. Current working directory  (fallback)
    """
    if _ENV_BASE:
        candidate = os.path.join(_ENV_BASE, relative_path)
        if os.path.exists(candidate):
            return candidate

    # PyInstaller bundles everything into _MEIPASS
    if hasattr(sys, "_MEIPASS"):
        candidate = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(candidate):
            return candidate

    # Same directory as the script being run
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    candidate  = os.path.join(script_dir, relative_path)
    if os.path.exists(candidate):
        return candidate

    # CWD fallback — lets you run from the weights directory
    return os.path.join(os.getcwd(), relative_path)
