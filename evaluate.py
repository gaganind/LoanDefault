# evaluate.py
"""
Evaluation helpers: use saved model to evaluate any dataset
that contains the 'Default' column.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)

from pipeline import load_artifacts, predict_proba_from_raw, TARGET_COL, BEST_THRESHOLD


def evaluate_on_dataset(df: pd.DataFrame):
    """
    df must include the target column TARGET_COL.
    """
    model = load_artifacts()

    df = df.copy()
    y_true = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    y_probs = predict_proba_from_raw(model, X)

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_probs = pd.Series(y_probs).reset_index(drop=True)

    y_pred = (y_probs >= BEST_THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_probs),
        "pr_auc": average_precision_score(y_true, y_probs),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred),
    }
    return metrics
