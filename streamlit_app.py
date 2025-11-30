import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except:
    sns = None

from pipeline import load_artifacts, predict_proba_from_raw, TARGET_COL, BEST_THRESHOLD

st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("📊 Loan Default Prediction – Evaluation App")

st.write("""
Upload a CSV file containing the same schema as training data,
including the `Default` column, to evaluate the model.
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

model = load_artifacts()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### 🔍 Data Preview")
    st.dataframe(df.head())

    if st.button("Evaluate Model"):
        with st.spinner("Evaluating..."):

            y_true = df[TARGET_COL]
            X = df.drop(columns=[TARGET_COL])

            y_probs = predict_proba_from_raw(model, X)
            y_pred = (y_probs >= BEST_THRESHOLD).astype(int)

            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, average_precision_score, confusion_matrix,
                classification_report
            )

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_probs)
            pr_auc = average_precision_score(y_true, y_probs)
            cm = confusion_matrix(y_true, y_pred)

            st.success("Evaluation Complete!")

            st.subheader("📈 Metrics")
            st.write(f"**Accuracy:** {acc:.4f}")
            st.write(f"**Precision:** {prec:.4f}")
            st.write(f"**Recall:** {rec:.4f}")
            st.write(f"**F1 Score:** {f1:.4f}")
            st.write(f"**ROC-AUC:** {roc_auc:.4f}")
            st.write(f"**PR-AUC:** {pr_auc:.4f}")

            st.subheader("🔵 Confusion Matrix")
            fig, ax = plt.subplots()
            if sns:
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            else:
                ax.imshow(cm)
                ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            st.subheader("📄 Classification Report")
            st.text(classification_report(y_true, y_pred))

st.write("---")
st.caption("Model powered by Gradient Boosting Ensemble + Streamlit Cloud")