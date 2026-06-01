# customer churn prediction

predicting which telecom customers are likely to cancel their service, with model explainability via SHAP so predictions can be understood by nontechnical stakeholders.

cover full ML lifecycle: raw data to a deployed interactive dashboard

--

## what this project does

takes a telecom customer's account info (contract type, charges, tenure, services) and outputs:
- a churn probability score (0–100%)
- top factors driving that prediction (via SHAP)
- a plain english explanation of why model flagged them as high risk

--

## project structure

```
customer-churn-prediction/
├── data/
│   └── raw/               # download dataset here (see below)
├── notebooks/
│   ├── 01_eda.ipynb        # exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # preprocessing pipeline
│   ├── 03_modeling.ipynb   # model training + comparison (coming soon)
│   └── 04_shap.ipynb       # explainability analysis (coming soon)
├── models/                 # saved model + preprocessor artifacts
├── app/                    # streamlit dashboard (coming soon)
├── requirements.txt
└── README.md
```

--

## dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) via Kaggle

- 7,043 customers, 21 features
- target: `Churn` (yes/no) — ~26.6% positive rate
- download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and save to `data/raw/telco_churn.csv`

--

## what we found in EDA (phase 1)

- **class imbalance**: only 26.6% of customers churned, rest stayed
- **contract type** is the strongest signal; month to month customers churn at ~43% vs ~3% for two year contracts
- **fiber optic** customers churn at ~42% vs ~7% for no internet service
- **electronic check** payment method correlates with ~45% churn rate
- **new customers are highest risk**: 0–1yr tenure group churns at ~49%
- `tenure` has -0.35 correlation with churn --> means longer customers are more loyal
- `TotalCharges` and `tenure` are 0.83 correlated —-> multicollinearity to manage in modeling

--

## what we built in feature engineering (phase 2)

- fixed `TotalCharges` dtype (ships as string with 11 blank rows)
- encoded binary yes/no columns to 1/0
- one-hot encoded multicategory columns (contract, payment method, internet service, etc.)
- scaled all numeric features with `StandardScaler`
- engineered new feature: `charges_per_month = TotalCharges / (tenure + 1)`
- applied **SMOTE** to fix class imbalance --> training set goes from 73/26 → 50/50
- saved fitted preprocessor to `models/preprocessor.pkl` for reuse in the app

--

## stack

- **python 3.14**
- pandas, numpy, matplotlib, seaborn
- scikit-learn for preprocessing + modeling
- xgboost for gradient boosting model
- shap for model explainability
- imbalanced learn for SMOTE oversampling
- streamlit for interactive dashboard

--

## setup

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction
cd customer-churn-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

also download dataset from Kaggle and place it at `data/raw/telco_churn.csv`