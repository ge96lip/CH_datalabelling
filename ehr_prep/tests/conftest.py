# ehr_prep/tests/conftest.py
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve()
PKG_DIR = HERE.parents[1]          # .../CH_datalabelling/ehr_prep
ROOT_DIR = PKG_DIR.parent          # .../CH_datalabelling

# Make BOTH available:
#  - ROOT_DIR -> allows "import ehr_prep"
#  - PKG_DIR  -> allows "import parsers" used inside normalize.py
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PKG_DIR))