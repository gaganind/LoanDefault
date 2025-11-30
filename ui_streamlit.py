# ui_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# if you prefer, you can avoid seaborn and draw cm manually
try:
    import seaborn as sns
except ImportError:
    sns = None

API_URL = "http://localhost:8000/evaluate-file"  # adjust if API elsewhere

st.title("Loan Default Model – Evaluation UI")

st.write("""
Upload a CSV file containing the same schema as training data,
including the 'Default' column, and the app will compute:
- Confusion matrix
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    if st.button("Evaluate Model"):
        with st.spinner("Evaluating..."):
            files = {"file": ("uploaded.csv", uploaded_file.getvalue(), "text/csv")}
            resp = requests.post(API_URL, files=files)
            if resp.status_code != 200:
                st.error(f"Error from API: {resp.status_code} {resp.text}")
            else:
                data = resp.json()
                metrics = data["metrics"]

                st.subheader("Metrics")
                st.write(f"Accuracy: {metrics['accuracy']:.4f}")
                st.write(f"Precision: {metrics['precision']:.4f}")
                st.write(f"Recall: {metrics['recall']:.4f}")
                st.write(f"F1: {metrics['f1']:.4f}")
                st.write(f"ROC-AUC: {metrics['roc_auc']:.4f}")
                st.write(f"PR-AUC: {metrics['pr_auc']:.4f}")

                st.subheader("Confusion Matrix")
                cm = np.array(metrics["confusion_matrix"])
                fig, ax = plt.subplots()
                if sns:
                    sns.heatmap(
                        cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"],
                        ax=ax
                    )
                else:
                    ax.imshow(cm)
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, cm[i, j], ha="center", va="center")
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["Pred 0", "Pred 1"])
                    ax.set_yticks([0, 1])
                    ax.set_yticklabels(["True 0", "True 1"])
                st.pyplot(fig)

                st.subheader("Classification Report")
                st.text(metrics["classification_report"])
