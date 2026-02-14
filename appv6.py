# ======================================================
# INDUSTRY-STYLE ACTUARIAL INSURANCE PLATFORM
# Using freMTPL2 Dataset (Real Insurance Data)
# ======================================================

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import plotly.express as px
from sklearn.datasets import fetch_openml

st.set_page_config(layout="wide")
st.title("📊 Insurance Actuarial Analytics Platform (Real Data)")

# ======================================================
# LOAD REAL FREMTPL2 DATA
# ======================================================

@st.cache_data
def load_data():

    freq = fetch_openml(name="freMTPL2freq", version=1, as_frame=True)
    sev = fetch_openml(name="freMTPL2sev", version=1, as_frame=True)

    df_freq = freq.frame
    df_sev = sev.frame

    df = df_freq.merge(df_sev, how="left", on="IDpol")

    # Replace NA claim amounts with 0
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    return df

df = load_data()

# ======================================================
# EXPOSURE VALIDATION
# ======================================================

def validate_exposure(data):

    data = data[data["Exposure"] > 0]
    data = data[data["Exposure"] <= 1]  # annual policies
    return data

df = validate_exposure(df)

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================

module = st.sidebar.selectbox(
    "Select Module",
    [
        "Portfolio Dashboard",
        "Frequency Modeling",
        "Severity Modeling",
        "Pure Premium",
        "Deductible Modeling",
        "Machine Learning Pricing",
        "Risk Simulation"
    ]
)

# ======================================================
# MODULE 1: PORTFOLIO DASHBOARD
# ======================================================

if module == "Portfolio Dashboard":

    st.header("📈 Portfolio KPIs")

    total_exposure = df["Exposure"].sum()
    total_claims = df["ClaimNb"].sum()
    total_loss = df["ClaimAmount"].sum()

    frequency = total_claims / total_exposure
    severity = total_loss / total_claims if total_claims > 0 else 0
    pure_premium = total_loss / total_exposure

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Exposure", round(total_exposure,2))
    col2.metric("Claim Frequency", round(frequency,4))
    col3.metric("Pure Premium", round(pure_premium,2))

    fig = px.histogram(df[df["ClaimAmount"]>0], x="ClaimAmount", nbins=100)
    st.plotly_chart(fig)

# ======================================================
# MODULE 2: FREQUENCY MODELING
# ======================================================

elif module == "Frequency Modeling":

    st.header("🔢 Frequency GLM (Poisson with Exposure Offset)")

    formula = "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["Exposure"])
    ).fit()

    st.text(model.summary())

# ======================================================
# MODULE 3: SEVERITY MODELING
# ======================================================

elif module == "Severity Modeling":

    st.header("💰 Severity GLM (Gamma)")

    sev_df = df[df["ClaimAmount"] > 0]

    model = smf.glm(
        formula="ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=sev_df,
        family=sm.families.Gamma(sm.families.links.log())
    ).fit()

    st.text(model.summary())

# ======================================================
# MODULE 4: PURE PREMIUM MODELING
# ======================================================

elif module == "Pure Premium":

    st.header("📌 Frequency × Severity Pricing")

    freq_model = smf.glm(
        "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["Exposure"])
    ).fit()

    sev_df = df[df["ClaimAmount"] > 0]

    sev_model = smf.glm(
        "ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=sev_df,
        family=sm.families.Gamma(sm.families.links.log())
    ).fit()

    df["pred_freq"] = freq_model.predict(df)
    df["pred_sev"] = sev_model.predict(df)
    df["pred_pure_premium"] = df["pred_freq"] * df["pred_sev"]

    st.write(df[["pred_freq","pred_sev","pred_pure_premium"]].head())

# ======================================================
# MODULE 5: DEDUCTIBLE MODELING
# ======================================================

elif module == "Deductible Modeling":

    st.header("🧾 Deductible Impact Modeling")

    deductible = st.number_input("Select Deductible Amount", 0, 5000, 500)

    sev_df = df[df["ClaimAmount"] > 0].copy()

    # Apply deductible
    sev_df["AdjustedClaim"] = np.maximum(sev_df["ClaimAmount"] - deductible, 0)

    expected_before = sev_df["ClaimAmount"].mean()
    expected_after = sev_df["AdjustedClaim"].mean()

    st.write("Expected Severity Before Deductible:", round(expected_before,2))
    st.write("Expected Severity After Deductible:", round(expected_after,2))

    fig = px.histogram(sev_df, x="AdjustedClaim", nbins=100)
    st.plotly_chart(fig)

# ======================================================
# MODULE 6: MACHINE LEARNING PRICING
# ======================================================

elif module == "Machine Learning Pricing":

    st.header("🤖 ML Pure Premium Modeling")

    df["pure_premium"] = df["ClaimAmount"]

    features = ["VehPower","VehAge","DrivAge","BonusMalus"]
    X = df[features]
    y = df["pure_premium"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)

    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    preds_xgb = xgb.predict(X_test)

    st.write("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test,preds_rf)))
    st.write("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test,preds_xgb)))

# ======================================================
# MODULE 7: RISK SIMULATION
# ======================================================

elif module == "Risk Simulation":

    st.header("🎲 Monte Carlo Aggregate Loss Simulation")

    n_sim = st.slider("Number of Simulations", 1000, 20000, 5000)

    # Use empirical frequency and severity
    lambda_est = df["ClaimNb"].sum() / df["Exposure"].sum()
    mean_sev = df[df["ClaimAmount"]>0]["ClaimAmount"].mean()

    freq_sim = np.random.poisson(lambda_est, n_sim)
    sev_sim = np.random.gamma(shape=2, scale=mean_sev/2, size=n_sim)

    aggregate_loss = freq_sim * sev_sim

    VaR_95 = np.percentile(aggregate_loss, 95)
    VaR_99 = np.percentile(aggregate_loss, 99)

    st.write("VaR 95%:", round(VaR_95,2))
    st.write("VaR 99%:", round(VaR_99,2))

    fig = px.histogram(aggregate_loss, nbins=100)
    st.plotly_chart(fig)
