import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from statsmodels.formula.api import glm, ols
from statsmodels.genmod.families import Poisson, NegativeBinomial, Gamma
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

st.set_page_config(layout="wide")
st.title("🏆 Elite Actuarial Insurance Platform")

# ==========================================================
# 1️⃣ DATA LAYER
# ==========================================================

@st.cache_data
def load_data():
    freq = fetch_openml(name="freMTPL2freq", version=1, as_frame=True)
    sev = fetch_openml(name="freMTPL2sev", version=1, as_frame=True)

    df = freq.frame.merge(sev.frame, how="left", on="IDpol")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    numeric_cols = ["ClaimNb", "Exposure", "ClaimAmount",
                    "VehPower", "VehAge", "DrivAge", "BonusMalus"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)].dropna()
    return df

df = load_data()
train_idx, test_idx = train_test_split(df.index, test_size=0.3, random_state=42)

train_df = df.loc[train_idx]
test_df = df.loc[test_idx]

# ==========================================================
# 2️⃣ FREQUENCY MODELING LAYER
# ==========================================================

@st.cache_resource
def train_frequency(train_df):

    poisson_model = glm(
        "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=train_df,
        family=Poisson(),
        offset=np.log(train_df["Exposure"])
    ).fit()

    nb_model = glm(
        "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=train_df,
        family=NegativeBinomial(),
        offset=np.log(train_df["Exposure"])
    ).fit()

    return poisson_model, nb_model

poisson_model, nb_model = train_frequency(train_df)

# Dispersion Test
dispersion_ratio = train_df["ClaimNb"].var() / train_df["ClaimNb"].mean()

# Model comparison via AIC
freq_aic = pd.DataFrame({
    "Model": ["Poisson", "NegBin"],
    "AIC": [poisson_model.aic, nb_model.aic]
}).sort_values("AIC")

best_freq = "Poisson" if poisson_model.aic < nb_model.aic else "NegBin"

# Predictions
def predict_frequency(model, data):
    return model.predict(data, offset=np.log(data["Exposure"]))

if best_freq == "Poisson":
    df["Pred_Count"] = predict_frequency(poisson_model, df)
else:
    df["Pred_Count"] = predict_frequency(nb_model, df)

df["Pred_Freq"] = df["Pred_Count"] / df["Exposure"]

# ==========================================================
# 3️⃣ SEVERITY MODELING LAYER
# ==========================================================

sev_df = df[df["ClaimAmount"] > 0]

@st.cache_resource
def train_severity(sev_df):

    gamma_model = glm(
        "ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=sev_df,
        family=Gamma()
    ).fit()

    log_model = ols(
        "np.log(ClaimAmount) ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=sev_df
    ).fit()

    return gamma_model, log_model

gamma_model, log_model = train_severity(sev_df)

# Pareto Tail
u = sev_df["ClaimAmount"].quantile(0.9)
tail = sev_df[sev_df["ClaimAmount"] >= u]["ClaimAmount"]
alpha = len(tail) / np.sum(np.log(tail/u))
pareto_mean = alpha*u/(alpha-1)

df["Pred_Sev_Gamma"] = gamma_model.predict(df)
df["Pred_Sev_Log"] = np.exp(log_model.predict(df) + 0.5*log_model.mse_resid)

# Hybrid
blend = 1/(1+np.exp(-(df["Pred_Sev_Gamma"]-u)/(0.1*u)))
df["Pred_Sev"] = (1-blend)*df["Pred_Sev_Gamma"] + blend*pareto_mean

# ==========================================================
# 4️⃣ PRICING & CALIBRATION LAYER
# ==========================================================

df["Pure_Premium"] = df["Pred_Count"] * df["Pred_Sev"]

# Gini
def gini(actual, pred):
    auc = roc_auc_score(actual > 0, pred)
    return 2*auc - 1

gini_score = gini(df["ClaimAmount"], df["Pure_Premium"])

# Calibration by decile
df["Decile"] = pd.qcut(df["Pure_Premium"], 10, labels=False)
calibration = df.groupby("Decile").agg(
    Observed_LR=("ClaimAmount", "sum"),
    Predicted_LR=("Pure_Premium", "sum")
)

# ==========================================================
# 5️⃣ TRUE COMPOUND SIMULATION & CAPITAL MODEL
# ==========================================================

def simulate_losses(n_sim=5000, seed=42):
    rng = np.random.default_rng(seed)
    lam = df["Pred_Count"].mean()

    losses = []

    for _ in range(n_sim):
        N = rng.poisson(lam)
        if N == 0:
            losses.append(0)
        else:
            severities = rng.gamma(shape=2, scale=df["Pred_Sev"].mean()/2, size=N)
            losses.append(severities.sum())

    return np.array(losses)

sim_losses = simulate_losses()

VaR_995 = np.percentile(sim_losses, 99.5)
Expected_Loss = sim_losses.mean()
SCR = VaR_995 - Expected_Loss

# Risk Margin
CoC = 0.06
Risk_Margin = CoC * SCR

# ==========================================================
# 6️⃣ STREAMLIT DISPLAY LAYER
# ==========================================================

tabs = st.tabs([
    "Model Diagnostics",
    "Calibration",
    "Capital Modeling",
    "Governance",
])

with tabs[0]:
    st.header("Frequency Model Selection")
    st.write("Dispersion Ratio:", round(dispersion_ratio,3))
    st.dataframe(freq_aic)
    st.write("Best Model:", best_freq)
    st.write("Gini Coefficient:", round(gini_score,4))

with tabs[1]:
    st.header("Calibration by Decile")
    st.dataframe(calibration)

with tabs[2]:
    st.header("Capital Metrics (Solvency Style)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Loss", f"{Expected_Loss:,.2f}")
    c2.metric("VaR 99.5%", f"{VaR_995:,.2f}")
    c3.metric("SCR", f"{SCR:,.2f}")
    st.metric("Risk Margin (6% CoC)", f"{Risk_Margin:,.2f}")

with tabs[3]:
    st.header("Model Governance")
    st.write("""
    Assumptions:
    - Poisson / NegBin for frequency
    - Gamma + Pareto tail for severity
    - Lognormal bias correction applied
    - 99.5% VaR used for capital
    - Cost of Capital = 6%
    """)

