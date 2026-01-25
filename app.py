# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Auto Insurance Risk & Pricing App", layout="wide")

st.title("🚗 Auto Insurance Claim Frequency & Pricing App")
st.markdown("""
This app demonstrates **actuarial-style frequency modeling**,  
risk segmentation, and **pricing simulation** using real statistical models.
""")

# ------------------------------
# LOAD / GENERATE DATA
# ------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1500
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "region": np.random.choice(["Northeast", "Midwest", "South", "West"], n),
        "vehicle_type": np.random.choice(["Sedan", "SUV", "Truck"], n),
        "policy_type": np.random.choice(["Basic", "Standard", "Premium"], n),
        "exposure": np.random.uniform(0.5, 1.0, n)
    })

    base_rate = 0.08
    df["lambda"] = (
        base_rate
        + 0.002 * (df["age"] < 25)
        + 0.03 * (df["policy_type"] == "Basic")
        + 0.02 * (df["region"] == "South")
    )

    df["claims"] = np.random.poisson(df["lambda"] * df["exposure"])
    return df

df = load_data()

# ------------------------------
# SIDEBAR: MODEL SELECTION
# ------------------------------
st.sidebar.header("⚙️ Model Controls")

selected_vars = st.sidebar.multiselect(
    "Select model variables",
    ["age", "region", "vehicle_type", "policy_type"],
    default=["age", "region", "policy_type"]
)

model_type = st.sidebar.selectbox(
    "Select frequency model",
    ["Poisson", "Negative Binomial", "Zero-Inflated Poisson", "Zero-Inflated Negative Binomial"]
)

# ------------------------------
# DATA PREP
# ------------------------------
X = pd.get_dummies(df[selected_vars], drop_first=True)
X = sm.add_constant(X)
y = df["claims"]
exposure = df["exposure"]

# ------------------------------
# MODEL FITTING
# ------------------------------
if model_type == "Poisson":
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(exposure))
    results = model.fit()

elif model_type == "Negative Binomial":
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=np.log(exposure))
    results = model.fit()

elif model_type == "Zero-Inflated Poisson":
    model = ZeroInflatedPoisson(
        endog=y,
        exog=X,
        exog_infl=X,
        offset=np.log(exposure)
    )
    results = model.fit(method="bfgs", maxiter=100, disp=False)

else:
    model = ZeroInflatedNegativeBinomialP(
        endog=y,
        exog=X,
        exog_infl=X,
        offset=np.log(exposure)
    )
    results = model.fit(method="bfgs", maxiter=100, disp=False)

# ------------------------------
# MODEL OUTPUT
# ------------------------------
st.subheader("📊 Model Results")
st.dataframe(results.summary().tables[1])

st.info("""
**Why Zero-Inflated models?**  
Insurance data often has **many zero-claim policies**.  
ZIP/ZINB models separate:
- Claim occurrence probability  
- Claim frequency once claims occur
""")

# ------------------------------
# PREDICTED CLAIM RATES
# ------------------------------
df["predicted_claim_rate"] = results.predict(X)

# ------------------------------
# RISK BY SEGMENT
# ------------------------------
st.subheader("🔍 Risk Segmentation")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.box(df, x="policy_type", y="predicted_claim_rate",
                     title="Predicted Claim Rate by Policy Type")
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_region = px.box(df, x="region", y="predicted_claim_rate",
                        title="Predicted Claim Rate by Region")
    st.plotly_chart(fig_region, use_container_width=True)

# ------------------------------
# REGION-LEVEL RISK MAP (SIMULATED GEO)
# ------------------------------
st.subheader("🗺️ Regional Risk Map")

region_risk = df.groupby("region")["predicted_claim_rate"].mean().reset_index()

fig_map = px.bar(
    region_risk,
    x="region",
    y="predicted_claim_rate",
    title="Average Claim Frequency by Region",
    labels={"predicted_claim_rate": "Expected Claims per Policy"}
)

st.plotly_chart(fig_map, use_container_width=True)

# ------------------------------
# CLAIM RATE SIMULATOR
# ------------------------------
st.subheader("🎯 Claim Rate Simulator")

age_sim = st.slider("Driver Age", 18, 80, 35)
region_sim = st.selectbox("Region", df["region"].unique())
policy_sim = st.selectbox("Policy Type", df["policy_type"].unique())

sim_row = pd.DataFrame({
    "age": [age_sim],
    "region": [region_sim],
    "policy_type": [policy_sim]
})

sim_X = pd.get_dummies(sim_row, drop_first=True)
sim_X = sim_X.reindex(columns=X.columns, fill_value=0)
sim_X["const"] = 1

sim_claim_rate = results.predict(sim_X)[0]

st.success(f"📈 Expected annual claim frequency: **{sim_claim_rate:.3f}**")

# ------------------------------
# PRICING SIMULATION
# ------------------------------
st.subheader("💰 Pricing Simulation")

avg_claim_cost = st.number_input("Average Claim Cost ($)", 500, 20000, 3500)
loading = st.slider("Risk Loading (%)", 0.0, 50.0, 20.0)

pure_premium = sim_claim_rate * avg_claim_cost
loaded_premium = pure_premium * (1 + loading / 100)

st.metric("Pure Premium ($)", f"{pure_premium:,.2f}")
st.metric("Loaded Premium ($)", f"{loaded_premium:,.2f}")

st.caption("""
**Pure Premium** = Expected Frequency × Expected Severity  
**Loaded Premium** includes expenses, profit, and risk margin.
""")
