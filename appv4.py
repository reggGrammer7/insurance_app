import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sqlite3
import hashlib
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from scipy.optimize import minimize
import shap

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(page_title="Enterprise Actuarial ML Platform", layout="wide")
st.title("🏢 Enterprise Actuarial Pricing & ML Platform")

# ---------------------------------------------------
# DATABASE SETUP (SQLite - PostgreSQL ready structure)
# ---------------------------------------------------

conn = sqlite3.connect("actuarial_platform.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS users(
            username TEXT,
            password TEXT)""")
conn.commit()

# ---------------------------------------------------
# AUTH FUNCTIONS
# ---------------------------------------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    c.execute("INSERT INTO users VALUES (?,?)",
              (username, hash_password(password)))
    conn.commit()

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    return c.fetchone()

# ---------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------

st.sidebar.title("Authentication")

auth_mode = st.sidebar.radio("Select Mode", ["Login", "Register"])

if auth_mode == "Register":
    new_user = st.sidebar.text_input("Username")
    new_pass = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Create Account"):
        create_user(new_user, new_pass)
        st.sidebar.success("Account created.")

if auth_mode == "Login":
    user = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if login_user(user, password):
            st.session_state.user = user
            st.success("Logged in successfully.")
        else:
            st.error("Invalid credentials.")

if "user" not in st.session_state:
    st.stop()

# ---------------------------------------------------
# MENU
# ---------------------------------------------------

menu = st.sidebar.radio("Modules", [
    "Generate Portfolio",
    "GLM Pricing",
    "Machine Learning Pricing",
    "Pricing Optimization",
    "Capital Allocation",
    "Monte Carlo Risk Capital",
    "Reinsurance Modeling",
    "IFRS 17 Reserves",
    "Executive Dashboard"
])

# ---------------------------------------------------
# PORTFOLIO GENERATION
# ---------------------------------------------------

if menu == "Generate Portfolio":

    n = st.slider("Number of Policies", 500, 10000, 2000)

    if st.button("Generate Portfolio"):

        exposure = np.random.uniform(0.5, 1.5, n)
        region = np.random.choice(["Urban", "Rural"], n)
        vehicle = np.random.choice(["Sedan", "SUV", "Truck"], n)

        freq = stats.poisson.rvs(0.8, size=n)
        severity = stats.gamma.rvs(a=2, scale=2500, size=n)

        df = pd.DataFrame({
            "Exposure": exposure,
            "Region": region,
            "Vehicle": vehicle,
            "Claim_Count": freq,
            "Claim_Severity": severity
        })

        df["Total_Loss"] = df["Claim_Count"] * df["Claim_Severity"]

        st.session_state.data = df
        st.success("Portfolio Generated.")

    if "data" in st.session_state:
        st.dataframe(st.session_state.data.head())

# ---------------------------------------------------
# GLM PRICING
# ---------------------------------------------------

if menu == "GLM Pricing" and "data" in st.session_state:

    df = st.session_state.data.copy()
    df["log_exposure"] = np.log(df["Exposure"])

    model = smf.glm(
        "Claim_Count ~ Region + Vehicle",
        data=df,
        family=sm.families.Poisson(),
        offset=df["log_exposure"]
    ).fit()

    st.text(model.summary())

    df["Predicted_Frequency"] = model.predict(df)
    st.session_state.data = df
    st.session_state.glm_model = model

# ---------------------------------------------------
# MACHINE LEARNING PRICING
# ---------------------------------------------------

if menu == "Machine Learning Pricing" and "data" in st.session_state:

    st.header("🤖 ML Pure Premium Modeling")

    df = st.session_state.data.copy()
    df["Pure_Premium"] = df["Total_Loss"] / df["Exposure"]

    df_encoded = pd.get_dummies(df, columns=["Region", "Vehicle"], drop_first=True)

    X = df_encoded.drop(["Claim_Count", "Claim_Severity",
                         "Total_Loss", "Pure_Premium"], axis=1)

    y = df_encoded["Pure_Premium"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model_choice = st.selectbox(
        "Select ML Model",
        ["Random Forest", "Gradient Boosting", "XGBoost"]
    )

    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=200)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor()
    else:
        model = XGBRegressor(n_estimators=200, verbosity=0)

    if st.button("Train ML Model"):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        st.metric("RMSE", round(rmse, 2))

        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        st.subheader("Feature Importance")
        st.dataframe(importance)

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=np.abs(shap_values).mean(axis=0),
            y=X.columns,
            orientation='h'
        ))
        fig.update_layout(template="plotly_dark",
                          title="SHAP Feature Impact")
        st.plotly_chart(fig)

        df_encoded["ML_Pure_Premium"] = model.predict(X)

        st.session_state.data = df_encoded
        st.session_state.ml_model = model

        st.success("ML Model Trained Successfully")

# ---------------------------------------------------
# PRICING OPTIMIZATION
# ---------------------------------------------------

if menu == "Pricing Optimization" and "data" in st.session_state:

    df = st.session_state.data

    base_premium = df["Total_Loss"].mean()

    def objective(price):
        demand = np.exp(-0.0001 * price)
        profit = (price - base_premium) * demand
        return -profit

    result = minimize(objective, x0=[base_premium])
    optimal_price = result.x[0]

    st.metric("Base Premium", round(base_premium, 2))
    st.metric("Optimal Premium", round(optimal_price, 2))

# ---------------------------------------------------
# CAPITAL ALLOCATION
# ---------------------------------------------------

if menu == "Capital Allocation" and "data" in st.session_state:

    df = st.session_state.data

    total_loss = df["Total_Loss"].sum()
    capital = np.percentile(df["Total_Loss"], 99)

    df["Capital_Contribution"] = df["Total_Loss"] / total_loss * capital

    allocation = df.groupby("Region")["Capital_Contribution"].sum()

    st.subheader("Capital Allocation by Region")
    st.dataframe(allocation)

# ---------------------------------------------------
# MONTE CARLO ECONOMIC CAPITAL
# ---------------------------------------------------

if menu == "Monte Carlo Risk Capital" and "data" in st.session_state:

    simulations = 5000
    lambda_freq = st.session_state.data["Claim_Count"].mean()

    simulated = []

    for _ in range(simulations):
        freq = stats.poisson.rvs(lambda_freq)
        if freq > 0:
            sev = stats.gamma.rvs(2, scale=2500, size=freq)
            simulated.append(sev.sum())
        else:
            simulated.append(0)

    simulated = np.array(simulated)

    VaR = np.percentile(simulated, 99)
    st.metric("Economic Capital (99% VaR)", round(VaR, 2))

# ---------------------------------------------------
# REINSURANCE
# ---------------------------------------------------

if menu == "Reinsurance Modeling" and "data" in st.session_state:

    retention = st.number_input("Retention Level", value=10000.0)

    df = st.session_state.data

    df["Ceded"] = np.maximum(df["Total_Loss"] - retention, 0)
    df["Retained"] = df["Total_Loss"] - df["Ceded"]

    st.metric("Gross Loss", round(df["Total_Loss"].sum(), 2))
    st.metric("Ceded Loss", round(df["Ceded"].sum(), 2))
    st.metric("Net Loss", round(df["Retained"].sum(), 2))

# ---------------------------------------------------
# IFRS 17
# ---------------------------------------------------

if menu == "IFRS 17 Reserves" and "data" in st.session_state:

    df = st.session_state.data

    BEL = df["Total_Loss"].mean()
    RA = np.percentile(df["Total_Loss"], 75) - BEL
    CSM = BEL * 0.1

    st.metric("Best Estimate Liability (BEL)", round(BEL, 2))
    st.metric("Risk Adjustment", round(RA, 2))
    st.metric("Contractual Service Margin (CSM)", round(CSM, 2))

# ---------------------------------------------------
# EXECUTIVE DASHBOARD
# ---------------------------------------------------

if menu == "Executive Dashboard" and "data" in st.session_state:

    df = st.session_state.data

    total_loss = df["Total_Loss"].sum()
    premium = df["Exposure"].sum() * df["Total_Loss"].mean()
    loss_ratio = total_loss / premium

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Premium", round(premium, 2))
    col2.metric("Total Loss", round(total_loss, 2))
    col3.metric("Loss Ratio", round(loss_ratio, 3))
