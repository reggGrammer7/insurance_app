# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.discrete.count_model import (
    ZeroInflatedPoisson,
    ZeroInflatedNegativeBinomialP
)

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Auto Insurance Risk & Pricing App", layout="wide")

st.title("🚗 Auto Insurance Claim Frequency & Pricing App")

# ==============================
# DATA
# ==============================
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1500

    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "region": np.random.choice(["Northeast", "Midwest", "South", "West"], n),
        "policy_type": np.random.choice(["Basic", "Standard", "Premium"], n),
        "exposure": np.random.uniform(0.5, 1.0, n)
    })

    base_rate = 0.08
    lam = (
        base_rate
        + 0.02 * (df["age"] < 25)
        + 0.03 * (df["policy_type"] == "Basic")
        + 0.02 * (df["region"] == "South")
    )

    df["claims"] = np.random.poisson(lam * df["exposure"])
    return df


df = load_data()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Model Controls")

model_type = st.sidebar.selectbox(
    "Frequency Model",
    [
        "Poisson",
        "Negative Binomial",
        "Zero-Inflated Poisson",
        "Zero-Inflated Negative Binomial",
    ],
)

# ==============================
# DESIGN MATRIX
# ==============================
X = pd.get_dummies(
    df[["age", "region", "policy_type"]],
    drop_first=True
)

X.insert(0, "const", 1.0)

y = df["claims"]
exposure = df["exposure"]

# ==============================
# 🔥 NUCLEAR SANITIZATION 🔥
# ==============================
X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")
exposure = pd.to_numeric(exposure, errors="coerce")

mask = X.notnull().all(axis=1) & y.notnull() & exposure.notnull()

X = X.loc[mask]
y = y.loc[mask]
exposure = exposure.loc[mask]

# ⛔ CONVERT TO NUMPY (THIS IS THE FIX)
X_np = X.to_numpy(dtype=float)
y_np = y.to_numpy(dtype=float)
offset_np = np.log(exposure.to_numpy(dtype=float))

# ==============================
# MODEL FITTING (NUMPY ONLY)
# ==============================
if model_type == "Poisson":
    model = sm.GLM(
        y_np,
        X_np,
        family=sm.families.Poisson(),
        offset=offset_np
    )
    results = model.fit()

elif model_type == "Negative Binomial":
    model = sm.GLM(
        y_np,
        X_np,
        family=sm.families.NegativeBinomial(),
        offset=offset_np
    )
    results = model.fit()

elif model_type == "Zero-Inflated Poisson":
    model = ZeroInflatedPoisson(
        endog=y_np,
        exog=X_np,
        exog_infl=X_np,
        offset=offset_np
    )
    results = model.fit(method="lbfgs", maxiter=300, disp=False)

else:
    model = ZeroInflatedNegativeBinomialP(
        endog=y_np,
        exog=X_np,
        exog_infl=X_np,
        offset=offset_np
    )
    results = model.fit(method="lbfgs", maxiter=300, disp=False)

# ==============================
# OUTPUT
# ==============================
st.subheader("Model Coefficients")
st.dataframe(results.summary().tables[1])

# ==============================
# PREDICTIONS
# ==============================
if "Zero" in model_type:
    df = df.loc[X.index]
    df["predicted_claim_rate"] = results.predict(X_np, which="mean")
else:
    df = df.loc[X.index]
    df["predicted_claim_rate"] = results.predict(X_np)

# ==============================
# VISUALS
# ==============================
fig = px.box(
    df,
    x="policy_type",
    y="predicted_claim_rate",
    title="Predicted Claim Frequency by Policy Type"
)
st.plotly_chart(fig, use_container_width=True)