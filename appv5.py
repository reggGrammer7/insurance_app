# ================================
# INSURANCE ACTUARIAL ANALYTICS APP
# ================================

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📊 Insurance Actuarial Analytics Platform")

# =====================================================
# SYNTHETIC DATA GENERATION (Replace with real data)
# =====================================================

@st.cache_data
def generate_data(n=5000):
    np.random.seed(42)

    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "vehicle_power": np.random.randint(50, 300, n),
        "region": np.random.choice(["Urban", "Suburban", "Rural"], n),
        "exposure": np.random.uniform(0.3, 1.0, n),
    })

    # Frequency
    lambda_freq = 0.05 + 0.0005*df["vehicle_power"] + 0.001*(df["age"]<25)
    df["claim_count"] = np.random.poisson(lambda_freq * df["exposure"])

    # Severity
    df["claim_amount"] = np.where(
        df["claim_count"]>0,
        np.random.gamma(shape=2, scale=2000, size=n),
        0
    )

    df["pure_premium"] = df["claim_count"] * df["claim_amount"]
    df["premium"] = df["pure_premium"] * 1.3

    return df

df = generate_data()

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

module = st.sidebar.selectbox(
    "Select Module",
    [
        "Portfolio Dashboard",
        "Frequency Modeling",
        "Severity Modeling",
        "Pure Premium",
        "Reserving",
        "Machine Learning Pricing",
        "Risk Simulation"
    ]
)

# =====================================================
# MODULE 1: PORTFOLIO DASHBOARD
# =====================================================

if module == "Portfolio Dashboard":

    st.header("📈 Portfolio KPIs")

    total_policies = len(df)
    total_claims = df["claim_count"].sum()
    total_exposure = df["exposure"].sum()
    total_loss = df["pure_premium"].sum()
    total_premium = df["premium"].sum()

    loss_ratio = total_loss / total_premium
    frequency = total_claims / total_exposure
    severity = total_loss / total_claims if total_claims > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Policies", total_policies)
    col2.metric("Loss Ratio", round(loss_ratio,3))
    col3.metric("Frequency", round(frequency,3))

    fig = px.histogram(df, x="claim_amount", nbins=50)
    st.plotly_chart(fig)

# =====================================================
# MODULE 2: FREQUENCY MODELING
# =====================================================

elif module == "Frequency Modeling":

    st.header("🔢 Frequency GLM (Poisson)")

    formula = "claim_count ~ age + vehicle_power"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["exposure"])
    ).fit()

    st.text(model.summary())

# =====================================================
# MODULE 3: SEVERITY MODELING
# =====================================================

elif module == "Severity Modeling":

    st.header("💰 Severity GLM (Gamma)")

    sev_df = df[df["claim_amount"] > 0]

    model = smf.glm(
        formula="claim_amount ~ age + vehicle_power",
        data=sev_df,
        family=sm.families.Gamma(sm.families.links.log())
    ).fit()

    st.text(model.summary())

# =====================================================
# MODULE 4: PURE PREMIUM
# =====================================================

elif module == "Pure Premium":

    st.header("📌 Pure Premium Estimation")

    freq_model = smf.glm(
        "claim_count ~ age + vehicle_power",
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["exposure"])
    ).fit()

    sev_df = df[df["claim_amount"]>0]

    sev_model = smf.glm(
        "claim_amount ~ age + vehicle_power",
        data=sev_df,
        family=sm.families.Gamma(sm.families.links.log())
    ).fit()

    df["pred_freq"] = freq_model.predict(df)
    df["pred_sev"] = sev_model.predict(df)

    df["pred_pure_premium"] = df["pred_freq"] * df["pred_sev"]

    st.write(df[["pred_freq","pred_sev","pred_pure_premium"]].head())

# =====================================================
# MODULE 5: RESERVING (CHAIN LADDER)
# =====================================================

elif module == "Reserving":

    st.header("📊 Chain Ladder Reserving")

    triangle = pd.DataFrame({
        0: [1000,1200,1500,1300],
        1: [1500,1700,2000,None],
        2: [1800,2100,None,None],
        3: [2000,None,None,None]
    })

    st.write("Loss Development Triangle")
    st.write(triangle)

    # Simple development factors
    factors = []
    for i in range(3):
        num = triangle[i+1].sum(skipna=True)
        den = triangle[i].sum(skipna=True)
        factors.append(num/den)

    st.write("Development Factors:", factors)

# =====================================================
# MODULE 6: MACHINE LEARNING PRICING
# =====================================================

elif module == "Machine Learning Pricing":

    st.header("🤖 ML Pricing Models")

    X = df[["age","vehicle_power"]]
    y = df["pure_premium"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    rf = RandomForestRegressor()
    rf.fit(X_train,y_train)
    preds_rf = rf.predict(X_test)

    xgb = XGBRegressor()
    xgb.fit(X_train,y_train)
    preds_xgb = xgb.predict(X_test)

    st.write("RF RMSE:", np.sqrt(mean_squared_error(y_test,preds_rf)))
    st.write("XGB RMSE:", np.sqrt(mean_squared_error(y_test,preds_xgb)))

# =====================================================
# MODULE 7: RISK SIMULATION
# =====================================================

elif module == "Risk Simulation":

    st.header("🎲 Monte Carlo Risk Simulation")

    n_sim = st.slider("Number of Simulations",1000,20000,5000)

    freq = np.random.poisson(0.1,n_sim)
    sev = np.random.gamma(2,2000,n_sim)

    aggregate_loss = freq * sev

    VaR_95 = np.percentile(aggregate_loss,95)
    VaR_99 = np.percentile(aggregate_loss,99)

    st.write("VaR 95%:", round(VaR_95,2))
    st.write("VaR 99%:", round(VaR_99,2))

    fig = px.histogram(aggregate_loss, nbins=50)
    st.plotly_chart(fig)

