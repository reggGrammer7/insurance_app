# actuarial_pricing_engine.py

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px

st.set_page_config(page_title="Actuarial Pricing Engine", layout="wide")

st.title("🏢 Professional Insurance Pricing Engine")
st.markdown("Frequency × Severity Modeling with Exposure & Risk Factors")

# ---------------------------------------------------
# DATA GENERATION (Policy Level)
# ---------------------------------------------------

@st.cache_data
def generate_insurance_data(n=3000):

    np.random.seed(42)

    age = np.random.randint(18, 80, n)
    vehicle_type = np.random.choice([0,1], n)  # 0=Sedan, 1=SUV
    region = np.random.choice([0,1,2], n)
    exposure = np.random.uniform(0.5, 1.0, n)

    # True frequency process
    lambda_true = np.exp(
        -3 +
        0.02 * age +
        0.4 * vehicle_type +
        0.3 * region
    ) * exposure

    claim_count = np.random.poisson(lambda_true)

    # Severity only for positive claims
    base_severity = np.exp(8 + 0.01 * age + 0.2 * vehicle_type)
    total_claim_amount = claim_count * np.random.gamma(2, base_severity/2)

    df = pd.DataFrame({
        "Age": age,
        "Vehicle_Type": vehicle_type,
        "Region": region,
        "Exposure": exposure,
        "Claim_Count": claim_count,
        "Total_Claim_Amount": total_claim_amount
    })

    return df


df = generate_insurance_data()

st.subheader("Sample Data Preview")
st.dataframe(df.head())

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

train, test = train_test_split(df, test_size=0.3, random_state=42)

# ---------------------------------------------------
# FREQUENCY MODEL (Poisson GLM with Offset)
# ---------------------------------------------------

st.header("📈 Frequency Modeling")

X_train_freq = sm.add_constant(train[["Age", "Vehicle_Type", "Region"]])
y_train_freq = train["Claim_Count"]
offset_train = np.log(train["Exposure"])

freq_model = sm.GLM(
    y_train_freq,
    X_train_freq,
    family=sm.families.Poisson(),
    offset=offset_train
).fit()

st.subheader("Frequency Model Summary")
st.text(freq_model.summary())

# Predict on test
X_test_freq = sm.add_constant(test[["Age", "Vehicle_Type", "Region"]])
offset_test = np.log(test["Exposure"])
test["Pred_Frequency"] = freq_model.predict(X_test_freq, offset=offset_test)

# ---------------------------------------------------
# SEVERITY MODEL (Gamma GLM)
# ---------------------------------------------------

st.header("💰 Severity Modeling")

train_sev = train[train["Claim_Count"] > 0].copy()
train_sev["Avg_Severity"] = train_sev["Total_Claim_Amount"] / train_sev["Claim_Count"]

X_train_sev = sm.add_constant(train_sev[["Age", "Vehicle_Type", "Region"]])
y_train_sev = train_sev["Avg_Severity"]

sev_model = sm.GLM(
    y_train_sev,
    X_train_sev,
    family=sm.families.Gamma(sm.families.links.log())
).fit()

st.subheader("Severity Model Summary")
st.text(sev_model.summary())

# Predict severity
X_test_sev = sm.add_constant(test[["Age", "Vehicle_Type", "Region"]])
test["Pred_Severity"] = sev_model.predict(X_test_sev)

# ---------------------------------------------------
# PURE PREMIUM CALCULATION
# ---------------------------------------------------

st.header("💵 Pure Premium Calculation")

test["Pred_Pure_Premium"] = test["Pred_Frequency"] * test["Pred_Severity"]

st.dataframe(test[[
    "Pred_Frequency",
    "Pred_Severity",
    "Pred_Pure_Premium"
]].head())

# ---------------------------------------------------
# VALIDATION
# ---------------------------------------------------

st.header("📊 Model Validation")

# Actual Loss Cost
test["Actual_Loss_Cost"] = test["Total_Claim_Amount"]

rmse = np.sqrt(mean_squared_error(
    test["Actual_Loss_Cost"],
    test["Pred_Pure_Premium"]
))

st.metric("Out-of-Sample RMSE (Pure Premium)", round(rmse,2))

# Plot comparison
fig = px.scatter(
    test,
    x="Actual_Loss_Cost",
    y="Pred_Pure_Premium",
    title="Actual vs Predicted Loss Cost"
)

st.plotly_chart(fig)
