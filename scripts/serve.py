"""Entry point to launch the PPA Inference API Server.

Usage:
    PYTHONPATH=src uv run python scripts/serve.py [--port 8000] [--checkpoint "Data Files/best.pt"]
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ppa.server import main

if __name__ == "__main__":
    main()
