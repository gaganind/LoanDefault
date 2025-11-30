# pipeline.py
"""
End-to-end ML pipeline for Loan Default model,
ported from loan_default_v3.ipynb.

Includes:
- Feature engineering
- Preprocessing (impute, OHE, multicollinearity drop, scaling)
- Feature selection (ExtraTrees)
- Train / inference helpers
- Ensemble training (LR + CatBoost + XGB + LGBM)
"""

import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import joblib
from pathlib import Path


# 
# GLOBALS (from notebook)
# 

TARGET_COL = "Default"

numeric_cols = [
    "Client_Income", "Child_Count", "Credit_Amount", "Loan_Annuity",
    "Population_Region_Relative", "Age_Days", "Employed_Days",
    "Registration_Days", "ID_Days", "Own_House_Age",
    "Client_Family_Members", "Score_Source_1", "Score_Source_2",
    "Score_Source_3", "Social_Circle_Default", "Phone_Change",
    "Credit_Bureau"
]

categorical_cols = [
    "Car_Owned", "Bike_Owned", "Active_Loan", "House_Own",
    "Accompany_Client", "Client_Income_Type", "Client_Education",
    "Client_Marital_Status", "Client_Gender",
    "Loan_Contract_Type", "Client_Housing_Type",
    "Mobile_Tag", "Homephone_Tag", "Workphone_Working",
    "Client_Occupation", "Application_Process_Day",
    "Application_Process_Hour", "Client_Permanent_Match_Tag",
    "Client_Contact_Work_Tag", "Type_Organization",
    "Cleint_City_Rating"
]

# From your tuning cell:
BEST_THRESHOLD = 0.45

# Best hyperparams from Optuna (copied from notebook cell 17)
best_lr_params = {"C": 0.0021831740860139494}  # l1_ratio unused after you switch to l2

best_cat_params = {
    "iterations": 720,
    "depth": 8,
    "learning_rate": 0.14749085852037502,
    "l2_leaf_reg": 8.325394043482074,
    "random_strength": 10.078404243138106,
    "bagging_temperature": 0.40769575192903906,
}

best_xgb_params = {
    "n_estimators": 529,
    "learning_rate": 0.11035195585768404,
    "max_depth": 6,
    "min_child_weight": 11,
    "subsample": 0.6954631887061631,
    "colsample_bytree": 0.5874632325940672,
    "gamma": 0.18030890330611782,
    "reg_alpha": 1.5766604818203613,
    "reg_lambda": 12.5237655582215,
}

best_lgbm_params = {
    "n_estimators": 729,
    "learning_rate": 0.13891479514084934,
    "num_leaves": 44,
    "max_depth": 6,
    "min_child_samples": 77,
    "subsample": 0.5243253811425539,
    "colsample_bytree": 0.5047936560952209,
    "reg_alpha": 11.336592038772189,
    "reg_lambda": 15.635562072081797,
}

best_weights = (1, 8, 2, 3)  # (lr, cat, xgb, lgbm) from your ensemble scan

# Where to save model artifacts
MODEL_PATH = Path("models/loan_default_final_model.pkl")


# 
# 2. FEATURE ENGINEERING (from notebook)
# 

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()

    # 1. EXTERNAL SCORE AGGREGATIONS
    score_cols = ['Score_Source_1', 'Score_Source_2', 'Score_Source_3']
    existing_scores = [c for c in score_cols if c in df_new.columns]
    if existing_scores:
        df_new['NEW_Score_Mean'] = df_new[existing_scores].mean(axis=1)
        df_new['NEW_Score_Max'] = df_new[existing_scores].max(axis=1)
        df_new['NEW_Score_Min'] = df_new[existing_scores].min(axis=1)
        df_new['NEW_Score_Var'] = df_new[existing_scores].var(axis=1)

    # 2. FINANCIAL CAPACITY
    if 'Loan_Annuity' in df_new.columns and 'Credit_Amount' in df_new.columns:
        df_new['NEW_Payment_Rate'] = df_new['Loan_Annuity'] / (df_new['Credit_Amount'] + 1)

    # 3. STABILITY METRICS
    if 'Employed_Days' in df_new.columns and 'Age_Days' in df_new.columns:
        df_new['NEW_Employed_Ratio'] = df_new['Employed_Days'] / (df_new['Age_Days'] + 1)

    if 'ID_Days' in df_new.columns and 'Age_Days' in df_new.columns:
        df_new['NEW_ID_Change_Ratio'] = df_new['ID_Days'] / (df_new['Age_Days'] + 1)

    # 4. INTERACTION FEATURES
    if 'Client_Income' in df_new.columns and 'Age_Days' in df_new.columns:
        df_new['NEW_Income_per_Age'] = df_new['Client_Income'] / (df_new['Age_Days'] + 1)

    if 'Client_Income' in df_new.columns and 'Population_Region_Relative' in df_new.columns:
        df_new['NEW_Region_Adjusted_Income'] = (
            df_new['Client_Income'] * df_new['Population_Region_Relative']
        )

    return df_new


# 
# 3. BASIC UTILITIES: IMPUTE, ENCODE, SCALE, FEATURE SELECTION
# 

def smart_impute(df, numerical_cols, categorical_cols):
    df_clean = df.copy()
    impute_values = {}

    # numerical
    for col in numerical_cols:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            impute_values[col] = median_val

    # categorical
    for col in categorical_cols:
        if col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
                impute_values[col] = mode_val

    return df_clean, impute_values


def encode_categorical(df, categorical_cols):
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)


def remove_multicollinearity(df, threshold=0.95):
    df_corr = df.corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))

    to_drop = [
        column for column in upper.columns
        if any(upper[column] > threshold)
    ]

    if to_drop:
        print(f"Dropping {len(to_drop)} multicollinear features.")
    df_reduced = df.drop(columns=to_drop, errors="ignore")

    return df_reduced, to_drop


def feature_selection_train(X, y, n_estimators=200, random_state=42):
    """
    Supervised feature selection using ExtraTrees + SelectFromModel.
    Stores selected feature names in PREPROCESSOR["selected_features"].
    """
    global PREPROCESSOR
    X = X.copy()
    selector = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    selector.fit(X, y)

    sfm = SelectFromModel(selector, prefit=True, threshold=0.0)
    mask = sfm.get_support()
    selected_cols = X.columns[mask]

    print(f"Features reduced from {X.shape[1]} to {len(selected_cols)} using ExtraTrees feature importances.")
    PREPROCESSOR["selected_features"] = list(selected_cols)

    return X[selected_cols]


def apply_feature_selection(X):
    """Apply previously learned feature selection to any dataframe."""
    global PREPROCESSOR
    if PREPROCESSOR.get("selected_features") is None:
        return X
    cols = PREPROCESSOR["selected_features"]
    X_aligned = X.reindex(columns=cols, fill_value=0)
    return X_aligned


def clean_feature_names(df):
    df = df.copy()
    new_columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
    df.columns = new_columns
    return df


def fit_scaler(df, numeric_cols_after):
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    df_scaled = df.copy()
    if numeric_cols_after:
        df_scaled[numeric_cols_after] = pt.fit_transform(df_scaled[numeric_cols_after])
    return df_scaled, pt


def apply_scaler(df, numeric_cols_after, pt):
    df_scaled = df.copy()
    if numeric_cols_after:
        df_scaled[numeric_cols_after] = pt.transform(df_scaled[numeric_cols_after])
    return df_scaled


# 
# 4. PREPROCESSOR STATE (to save & reuse)
# 

PREPROCESSOR = {
    "numerical_medians": {},
    "categorical_modes": {},
    "power_transformer": None,
    "multicollinearity_drop_cols": [],
    "final_columns": None,
    "ohe_columns": None,
    "selected_features": None,
}


def full_preprocess_pipeline(df, numerical_cols, categorical_cols, mode="train"):
    """
    FULL PIPELINE FOR TRAIN / INFERENCE

    mode="train":
        - fits imputers, OHE structure, multicollinearity dropping, scaler
        - stores everything in PREPROCESSOR
        - returns processed DataFrame

    mode="infer":
        - uses saved PREPROCESSOR to transform new data
    """
    global PREPROCESSOR
    df = df.copy()

    # 0. Drop ID-like columns (if present)
    for col in ["ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 1. Add engineered features
    df_fe = add_advanced_features(df)

    # Identify final numeric cols (original + engineered)
    numeric_cols_after = [c for c in df_fe.columns if c in numeric_cols or c.startswith("NEW_")]

    # 2. Impute
    if mode == "train":
        df_imputed, impute_vals = smart_impute(df_fe, numeric_cols_after, categorical_cols)
        PREPROCESSOR["numerical_medians"] = {k: v for k, v in impute_vals.items() if k in numeric_cols_after}
        PREPROCESSOR["categorical_modes"] = {k: v for k, v in impute_vals.items() if k in categorical_cols}
    else:
        df_imputed = df_fe.copy()
        for col, val in PREPROCESSOR["numerical_medians"].items():
            if col in df_imputed.columns:
                df_imputed[col] = df_imputed[col].fillna(val)
        for col, val in PREPROCESSOR["categorical_modes"].items():
            if col in df_imputed.columns:
                df_imputed[col] = df_imputed[col].fillna(val)

    # 3. One-hot encoding
    if mode == "train":
        df_ohe = encode_categorical(df_imputed, categorical_cols)
        PREPROCESSOR["ohe_columns"] = df_ohe.columns.tolist()
    else:
        df_ohe = pd.get_dummies(df_imputed, columns=categorical_cols, drop_first=True, dtype=int)
        df_ohe = df_ohe.reindex(columns=PREPROCESSOR["ohe_columns"], fill_value=0)

    # 4. Drop multicollinear features
    if mode == "train":
        df_mc, dropped_cols = remove_multicollinearity(df_ohe)
        PREPROCESSOR["multicollinearity_drop_cols"] = dropped_cols
    else:
        df_mc = df_ohe.drop(columns=PREPROCESSOR["multicollinearity_drop_cols"], errors="ignore")

    # 5. Clean feature names
    df_mc = clean_feature_names(df_mc)

    # 6. Scaling / Yeo-Johnson
    numeric_cols_after = [c for c in df_mc.columns if c in numeric_cols_after]

    if mode == "train":
        df_scaled, pt = fit_scaler(df_mc, numeric_cols_after)
        df_scaled = df_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
        PREPROCESSOR["power_transformer"] = pt
        PREPROCESSOR["final_columns"] = df_scaled.columns.tolist()
        return df_scaled
    else:
        pt = PREPROCESSOR["power_transformer"]
        df_scaled = apply_scaler(df_mc, numeric_cols_after, pt)
        df_scaled = df_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
        df_final = df_scaled.reindex(columns=PREPROCESSOR["final_columns"], fill_value=0)
        return df_final


# 
# 5. DATA SPLITTING & TRAIN PREPROCESS
# 

def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train_raw, X_test_raw, y_train, y_test


def preprocess_train(X_train_raw, y_train):
    X_train_prep = full_preprocess_pipeline(
        X_train_raw,
        numerical_cols=numeric_cols,
        categorical_cols=categorical_cols,
        mode="train"
    )
    X_fs = feature_selection_train(X_train_prep, y_train)
    return X_fs, y_train


# 
# 6. ENSEMBLE TRAINING (LR + CAT + XGB + LGBM)
# 

def evaluate_single_model(model, name, X_test_prep, y_test):
    y_probs = model.predict_proba(X_test_prep)[:, 1]
    y_pred = (y_probs >= BEST_THRESHOLD).astype(int)

    print(f"\n=== {name} Evaluation on Test ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_probs))
    print("PR-AUC:", average_precision_score(y_test, y_probs))
    return {
        "name": name,
        "roc_auc": roc_auc_score(y_test, y_probs),
        "pr_auc": average_precision_score(y_test, y_probs),
    }


def train_ensemble(X_train_bal, y_train_bal, X_test_raw, y_test):
    """
    Uses tuned hyperparameters and conservative settings for each model,
    then trains a soft-voting ensemble with weights = best_weights.
    """
    print("\n===== STEP 3: Training Models + Individual Evaluations (REGULARIZED) =====")

    # Preprocess TEST
    X_test_prep = full_preprocess_pipeline(
        X_test_raw,
        numerical_cols=numeric_cols,
        categorical_cols=categorical_cols,
        mode="infer"
    )
    X_test_prep = apply_feature_selection(X_test_prep)

    # Internal validation
    from sklearn.model_selection import train_test_split as sk_split

    print("\nCreating internal validation split for early stopping...")
    X_tr, X_val, y_tr, y_val = sk_split(
        X_train_bal,
        y_train_bal,
        test_size=0.2,
        stratify=y_train_bal,
        random_state=42
    )
    pos_weight = float(np.sum(y_train_bal == 0)) / np.sum(y_train_bal == 1)
    print("Train (for model fit) shape:", X_tr.shape)
    print("Valid (for early stopping) shape:", X_val.shape)

    # A) Logistic Regression
    print("\nTraining Logistic Regression (regularized)...")
    lr_params = best_lr_params.copy()
    if "C" in lr_params:
        lr_params["C"] = min(lr_params["C"], 1.0)
    lr_params.setdefault("penalty", "l2")
    lr_params.setdefault("class_weight", "balanced")
    lr_params.setdefault("solver", "lbfgs")

    lr_final = LogisticRegression(
        **lr_params,
        max_iter=2000,
        n_jobs=-1
    )
    lr_final.fit(X_tr, y_tr)
    evaluate_single_model(lr_final, "Logistic Regression", X_test_prep, y_test)

    # B) CatBoost
    print("\nTraining CatBoost (with early stopping)...")
    cb_params = best_cat_params.copy()
    cb_params.update({
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": 42,
        "verbose": 0
    })
    cat_final = CatBoostClassifier(**cb_params)
    cat_final.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    evaluate_single_model(cat_final, "CatBoost", X_test_prep, y_test)

    # C) XGBoost
    print("\nTraining XGBoost (with early stopping)...")
    xgb_params = best_xgb_params.copy()
    if "max_depth" in xgb_params:
        xgb_params["max_depth"] = min(xgb_params["max_depth"], 6)
    if "n_estimators" in xgb_params:
        xgb_params["n_estimators"] = min(xgb_params["n_estimators"], 800)
    if "learning_rate" in xgb_params:
        xgb_params["learning_rate"] = min(xgb_params["learning_rate"], 0.1)

    xgb_final = XGBClassifier(
        **xgb_params,
        tree_method="hist",
        objective="binary:logistic",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=pos_weight
    )
    xgb_final.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="aucpr",
        early_stopping_rounds=50,
        verbose=False
    )
    evaluate_single_model(xgb_final, "XGBoost", X_test_prep, y_test)

    # D) LightGBM
    print("\nTraining LightGBM (with early stopping)...")
    lgbm_params = best_lgbm_params.copy()
    if "max_depth" in lgbm_params:
        lgbm_params["max_depth"] = min(lgbm_params["max_depth"], 6)
    if "n_estimators" in lgbm_params:
        lgbm_params["n_estimators"] = min(lgbm_params["n_estimators"], 800)
    if "learning_rate" in lgbm_params:
        lgbm_params["learning_rate"] = min(lgbm_params["learning_rate"], 0.1)

    lgbm_final = LGBMClassifier(
        **lgbm_params,
        random_state=42,
        n_jobs=-1,
        class_weight=None
    )
    lgbm_final.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[],
    )
    evaluate_single_model(lgbm_final, "LightGBM", X_test_prep, y_test)

    # SOFT VOTING ENSEMBLE
    print("\nTraining Soft Voting Ensemble with weights:", best_weights)
    final_ensemble = VotingClassifier(
        estimators=[
            ("lr", lr_final),
            ("cat", cat_final),
            ("xgb", xgb_final),
            ("lgbm", lgbm_final),
        ],
        voting="soft",
        weights=best_weights,
        n_jobs=-1,
    )
    final_ensemble.fit(X_train_bal, y_train_bal)

    print("\nEnsemble training complete.")
    return final_ensemble


# 
# 7. SAVING / LOADING / INFERENCE
# 

def save_artifacts(model):
    artifacts = {
        "model": model,
        "preprocessor": PREPROCESSOR,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "selected_features": PREPROCESSOR.get("selected_features"),
        "threshold": BEST_THRESHOLD,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, MODEL_PATH)
    print(f"\nModel saved successfully as: {MODEL_PATH}")


def load_artifacts():
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]

    global PREPROCESSOR, numeric_cols, categorical_cols, BEST_THRESHOLD
    PREPROCESSOR = payload["preprocessor"]
    numeric_cols = payload["numeric_cols"]
    categorical_cols = payload["categorical_cols"]
    BEST_THRESHOLD = payload["threshold"]

    return model


def predict_proba_from_raw(model, df_raw: pd.DataFrame):
    X_prep = full_preprocess_pipeline(
        df_raw,
        numerical_cols=numeric_cols,
        categorical_cols=categorical_cols,
        mode="infer"
    )
    X_prep = apply_feature_selection(X_prep)
    y_probs = model.predict_proba(X_prep)[:, 1]
    return y_probs
