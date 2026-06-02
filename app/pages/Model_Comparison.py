import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
)

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")

if "models" not in st.session_state:
    st.warning("Please visit the **Home** page first so models can be loaded.")
    st.stop()

models = st.session_state["models"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

st.subheader("Performance Metrics")

rows = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    rows.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
    })

df_metrics = pd.DataFrame(rows).set_index("Model")
st.dataframe(
    df_metrics.style
        .highlight_max(axis=0, color="#d4edda")
        .highlight_min(axis=0, color="#f8d7da")
        .format("{:.3f}"),
    use_container_width=True,
)
st.caption("Green = best value per metric. Red = lowest.")

st.divider()

col_roc, col_cm = st.columns([1, 1])

with col_roc:
    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        y_prob      = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Models")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_cm:
    st.subheader("Confusion Matrices")
    cm_cols = st.columns(len(models))
    for col, (name, model) in zip(cm_cols, models.items()):
        with col:
            y_pred   = model.predict(X_test)
            cm       = confusion_matrix(y_test, y_pred)
            fig2, ax2 = plt.subplots(figsize=(3, 2.8))
            ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(
                ax=ax2, colorbar=False, cmap="Blues"
            )
            ax2.set_title(name, fontsize=9)
            ax2.tick_params(labelsize=7)
            ax2.set_xlabel("Predicted", fontsize=8)
            ax2.set_ylabel("Actual", fontsize=8)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

st.divider()

st.subheader("Interpretation Guide")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
**Logistic Regression**
- Best AUC (0.835) and recall (0.79)
- Catches the most churners
- Best choice when missing a churner is costly
""")
with c2:
    st.markdown("""
**XGBoost**
- Best accuracy (0.763)
- Balanced precision/recall
- Best choice for SHAP explainability
""")
with c3:
    st.markdown("""
**Random Forest**
- Highest precision (0.55)
- Most conservative, flags fewer false positives
- Best when retention budget is limited
""")
