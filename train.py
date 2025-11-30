# train.py
"""
Training entrypoint for CI/CD.
- Loads Dataset2.csv
- Drops ID
- Splits train/test
- Preprocesses + feature selection
- Trains ensemble
- Saves artifacts
"""

import pandas as pd
from pathlib import Path

from pipeline import (
    TARGET_COL,
    split_data,
    preprocess_train,
    train_ensemble,
    save_artifacts,
)


DATA_PATH = Path("data/Dataset2.csv")  # adjust if needed


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Drop first unnamed column and ID if present
    if df.columns[0].lower().startswith("unnamed"):
        df = df.drop(df.columns[0], axis=1)
    df = df.drop(columns=["ID"], errors="ignore")

    print("Full df shape:", df.shape)

    # Train / Test split
    X_train_raw, X_test_raw, y_train, y_test = split_data(df)

    # Preprocess + feature selection on full train (no SMOTE, as in notebook)
    print("\n===== STEP 5: Preprocess Full Train =====")
    X_train_final, y_train_final = preprocess_train(X_train_raw, y_train)

    # Train ensemble
    model = train_ensemble(X_train_final, y_train_final, X_test_raw, y_test)

    # Save
    save_artifacts(model)


if __name__ == "__main__":
    main()