import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

ROOT       = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data" / "processed"

NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_month']
CAT_COLS = ['MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

st.set_page_config(page_title="SHAP Insights", layout="wide")
st.title("SHAP Global Insights")
st.markdown(
    "SHAP (SHapley Additive exPlanations) shows how much each feature contributed to "
    "predictions across the entire test set. Powered by **XGBoost + TreeExplainer**."
)

if "models" not in st.session_state:
    st.warning("Please visit the **Home** page first so models can be loaded.")
    st.stop()


@st.cache_data
def compute_global_shap():
    model        = joblib.load(MODELS_DIR / "xgb_model.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    X_test       = np.load(str(DATA_DIR / "X_test_processed.npy"))

    cat_names     = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(CAT_COLS)
    feature_names = NUM_COLS + list(cat_names)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    X_test_df   = pd.DataFrame(X_test, columns=feature_names)

    return shap_values, float(explainer.expected_value), X_test_df, feature_names


with st.spinner("Computing SHAP values on 1,407 test customers..."):
    shap_values, expected_value, X_test_df, feature_names = compute_global_shap()

st.divider()

st.subheader("1. Mean Absolute SHAP Values (Feature Importance)")
st.caption(
    "Average magnitude of each feature's impact on predictions. "
    "Higher = more influential on whether a customer churns."
)

mean_abs = np.abs(shap_values).mean(axis=0)
importance_df = (
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    .sort_values("mean_abs_shap", ascending=False)
    .head(15)
)

fig1, ax1 = plt.subplots(figsize=(8, 5))
bars = ax1.barh(
    importance_df["feature"][::-1],
    importance_df["mean_abs_shap"][::-1],
    color="#4e79a7",
)
ax1.set_xlabel("Mean |SHAP value|")
ax1.set_title("Top 15 Features by Mean Absolute SHAP")
ax1.grid(axis="x", alpha=0.3)
ax1.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
st.pyplot(fig1, use_container_width=True)
plt.close(fig1)

st.divider()

st.subheader("2. SHAP Summary Plot (Beeswarm)")
st.caption(
    "Each dot is one customer. **Red** = high feature value, **blue** = low. "
    "Position on x-axis shows impact on churn prediction. "
    "Features are ordered by importance (top = most impactful)."
)

plt.close("all")
shap.summary_plot(shap_values, X_test_df, max_display=15, show=False)
fig2 = plt.gcf()
fig2.set_size_inches(9, 6)
st.pyplot(fig2, bbox_inches="tight")
plt.close(fig2)

st.divider()

st.subheader("3. Dependence Plot")
st.caption(
    "Shows how one feature's value relates to its SHAP impact across all customers. "
    "Colour shows a second feature that interacts with it."
)

top_features = importance_df["feature"].tolist()
selected     = st.selectbox("Feature to explore", top_features, index=0)

plt.close("all")
shap.dependence_plot(selected, shap_values, X_test_df, show=False)
fig3 = plt.gcf()
fig3.set_size_inches(8, 5)
st.pyplot(fig3, bbox_inches="tight")
plt.close(fig3)

st.divider()

st.subheader("4. Highest-Risk Customer Breakdown")
st.caption(
    "Waterfall plot for the test customer the model is most confident will churn. "
    "Each bar shows one feature's contribution."
)

xgb_model = joblib.load(MODELS_DIR / "xgb_model.pkl")
X_test_arr = X_test_df.values
probs      = xgb_model.predict_proba(X_test_arr)[:, 1]
idx        = int(np.argmax(probs))

st.info(f"Customer index **#{idx}**, predicted churn probability: **{probs[idx]:.1%}**")

explanation = shap.Explanation(
    values        = shap_values[idx],
    base_values   = expected_value,
    data          = X_test_df.iloc[idx].values,
    feature_names = feature_names,
)

plt.close("all")
shap.plots.waterfall(explanation, max_display=14, show=False)
fig4 = plt.gcf()
st.pyplot(fig4, bbox_inches="tight")
plt.close(fig4)
