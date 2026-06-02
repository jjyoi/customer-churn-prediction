import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_month']
CAT_COLS = ['MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

CAT_OPTIONS = {
    'MultipleLines': ['No phone service', 'No', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No internet service', 'No', 'Yes'],
    'OnlineBackup': ['No internet service', 'No', 'Yes'],
    'DeviceProtection': ['No internet service', 'No', 'Yes'],
    'TechSupport': ['No internet service', 'No', 'Yes'],
    'StreamingTV': ['No internet service', 'No', 'Yes'],
    'StreamingMovies': ['No internet service', 'No', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],
}

PLAIN_ENGLISH = {
    'Contract_Month-to-month': 'Month-to-month contract (no long-term commitment)',
    'PaymentMethod_Electronic check': 'Electronic check payment, highest-churn method (~45%)',
    'InternetService_Fiber optic': 'Fiber optic internet, correlates with ~42% churn rate',
    'tenure': 'Low tenure, new customers churn at ~49%',
    'MonthlyCharges': 'High monthly charges, above average spend',
    'TechSupport_No': 'No tech support, unresolved issues drive churn',
    'OnlineSecurity_No': 'No online security add-on',
    'Contract_Two year': 'Two-year contract, very low churn risk',
    'Contract_One year': 'One-year contract, moderate loyalty signal',
}


@st.cache_resource
def load_models():
    return {
        "XGBoost": joblib.load(MODELS_DIR / "xgb_model.pkl"),
        "Random Forest": joblib.load(MODELS_DIR / "rf_model.pkl"),
        "Logistic Regression": joblib.load(MODELS_DIR / "lr_model.pkl"),
    }


@st.cache_resource
def load_preprocessor():
    return joblib.load(MODELS_DIR / "preprocessor.pkl")


@st.cache_data
def load_test_data():
    X = np.load(str(DATA_DIR / "X_test_processed.npy"))
    y = np.load(str(DATA_DIR / "y_test.npy"))
    return X, y


def get_feature_names(preprocessor):
    cat_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(CAT_COLS)
    return NUM_COLS + list(cat_names)


def build_input_df(tenure, monthly_charges, total_charges, cat_vals):
    row = {
        'gender': 0, 'SeniorCitizen': 0, 'Partner': 0, 'Dependents': 0,
        'PhoneService': 1, 'PaperlessBilling': 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'charges_per_month': total_charges / (tenure + 1),
    }
    row.update(cat_vals)
    return pd.DataFrame([row])


def shap_plain_english(shap_vals, feature_names, top_n=4):
    pairs = sorted(zip(shap_vals, feature_names), key=lambda x: x[0], reverse=True)
    drivers, protectors = [], []
    for val, name in pairs:
        label = PLAIN_ENGLISH.get(name, name.replace('_', ': '))
        if val > 0.05 and len(drivers) < top_n:
            drivers.append((val, label))
        elif val < -0.05 and len(protectors) < top_n:
            protectors.append((val, label))
    return drivers, protectors


st.set_page_config(page_title="Churn Predictor", layout="wide")

models = load_models()
preprocessor = load_preprocessor()
X_test, y_test = load_test_data()

st.session_state["models"] = models
st.session_state["X_test"] = X_test
st.session_state["y_test"] = y_test
st.session_state["preprocessor"] = preprocessor

st.title("Customer Churn Predictor")
st.markdown(
    "Enter a customer's account details to get their churn probability "
    "and a breakdown of the top factors driving the prediction."
)
st.divider()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", list(models.keys()))
    model      = models[model_name]
    st.divider()
    st.markdown("**Model performance (test set)**")
    st.markdown("""
| Model | AUC | Recall |
|---|---|---|
| Logistic Regression | 0.835 | 0.79 |
| XGBoost | 0.816 | 0.66 |
| Random Forest | 0.802 | 0.58 |
    """)
    st.caption("SHAP explanations available for XGBoost only.")

st.subheader("Customer Profile")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
    total_charges = round(monthly_charges * tenure, 2)
    st.caption(f"Estimated total charges: **${total_charges:,.0f}**")
    contract = st.selectbox("Contract",       CAT_OPTIONS['Contract'])
    payment_method = st.selectbox("Payment Method", CAT_OPTIONS['PaymentMethod'])

with c2:
    st.markdown("##### Internet Services")
    internet_service = st.selectbox("Internet Service", CAT_OPTIONS['InternetService'])
    online_security = st.selectbox("Online Security", CAT_OPTIONS['OnlineSecurity'])
    online_backup = st.selectbox("Online Backup", CAT_OPTIONS['OnlineBackup'])
    device_protection = st.selectbox("Device Protection", CAT_OPTIONS['DeviceProtection'])
    tech_support = st.selectbox("Tech Support", CAT_OPTIONS['TechSupport'])

with c3:
    st.markdown("##### Phone & Streaming")
    multiple_lines = st.selectbox("Multiple Lines", CAT_OPTIONS['MultipleLines'])
    streaming_tv = st.selectbox("Streaming TV", CAT_OPTIONS['StreamingTV'])
    streaming_movies = st.selectbox("Streaming Movies", CAT_OPTIONS['StreamingMovies'])

st.divider()
predict_btn = st.button("Predict Churn Risk", type="primary", use_container_width=True)

if predict_btn:
    cat_vals = {
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaymentMethod': payment_method,
    }

    df_input = build_input_df(tenure, monthly_charges, total_charges, cat_vals)
    X_input = preprocessor.transform(df_input)
    prob = float(model.predict_proba(X_input)[0, 1])
    pct = int(prob * 100)

    color = "#d32f2f" if pct >= 60 else "#f57c00" if pct >= 30 else "#2e7d32"
    label = "High Risk"   if pct >= 60 else "Medium Risk" if pct >= 30 else "Low Risk"

    r1, r2 = st.columns([1, 2])

    with r1:
        st.markdown(f"""
        <div style="text-align:center; padding:28px; border-radius:14px;
                    background:{color}18; border:2px solid {color}; margin-bottom:8px">
            <div style="font-size:60px; font-weight:700; color:{color}; line-height:1">{pct}%</div>
            <div style="font-size:20px; color:{color}; font-weight:600; margin-top:6px">{label}</div>
            <div style="font-size:13px; color:#888; margin-top:4px">churn probability</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Model: {model_name}")

    with r2:
        if model_name == "XGBoost":
            feature_names = get_feature_names(preprocessor)
            explainer     = shap.TreeExplainer(model)
            shap_vals_raw = explainer.shap_values(X_input)
            sv = shap_vals_raw[0] if shap_vals_raw.ndim == 2 else shap_vals_raw

            drivers, protectors = shap_plain_english(sv, feature_names)

            if drivers:
                st.markdown("**Why this customer is at risk:**")
                for val, txt in drivers:
                    st.markdown(f"- {txt} *(+{val:.2f})*")
            if protectors:
                st.markdown("**Factors reducing their risk:**")
                for val, txt in protectors:
                    st.markdown(f"- {txt} *({val:.2f})*")
            if not drivers and not protectors:
                st.markdown("No strong individual drivers detected.")
        else:
            st.markdown("**Key risk patterns for this customer:**")
            reasons = []
            if contract == "Month-to-month":
                reasons.append("Month-to-month contract (no long-term commitment)")
            if payment_method == "Electronic check":
                reasons.append("Electronic check payment, highest churn payment method")
            if internet_service == "Fiber optic":
                reasons.append("Fiber optic internet, correlates with ~42% churn rate")
            if tenure < 12:
                reasons.append(f"Low tenure ({tenure} mo), new customers churn ~49%")
            if monthly_charges > 80:
                reasons.append(f"High monthly charges (${monthly_charges:.0f}), above average")
            if not reasons:
                reasons.append("No major risk patterns detected.")
            for r in reasons:
                st.markdown(f"- {r}")

    if model_name == "XGBoost":
        st.subheader("SHAP Feature Explanation")
        st.caption(
            "Each bar shows how much a feature pushed this prediction toward (red) "
            "or away from (blue) churn. The base value is the average prediction across all customers."
        )

        explanation = shap.Explanation(
            values = sv,
            base_values = explainer.expected_value,
            data = X_input[0],
            feature_names = feature_names,
        )

        plt.close("all")
        shap.plots.waterfall(explanation, max_display=14, show=False)
        fig = plt.gcf()
        st.pyplot(fig, bbox_inches="tight")
        plt.close(fig)
