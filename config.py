# src/config.py
from pathlib import Path

RANDOM_STATE = 42
TARGET_COL = "Default"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Dataset2.csv"
MODEL_PATH = BASE_DIR / "models" / "loan_default_model.pkl"