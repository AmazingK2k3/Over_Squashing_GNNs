"""Central place for environment-specific paths.

Reads overrides from a local `.env` file (gitignored, see `.env.example`).
Falls back to paths relative to this file so the project works out of the
box on a fresh checkout with no `.env` present.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "DATA"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))
