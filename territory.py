import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson, Gamma
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.genmod.families import Poisson, Gamma, NegativeBinomial
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from sklearn.datasets import fetch_openml
import shap

st.set_page_config(layout="wide")
st.title("📊 Advanced Actuarial Pricing Platform with Territorial Modeling")

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_data():
    freq = fetch_openml(name="freMTPL2freq", version=1, as_frame=True)
    sev = fetch_openml(name="freMTPL2sev", version=1, as_frame=True)

    df = freq.frame.merge(sev.frame, how="left", on="IDpol")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    numeric_cols = [
        "ClaimNb", "Exposure", "ClaimAmount",
        "VehPower", "VehAge", "DrivAge",
        "BonusMalus", "Density"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)]
    df = df.dropna()

    return df

df = load_data()

st.success(f"Dataset Loaded | Records: {len(df)}")

# =====================================================
# MODEL TRAINING
st.header("1. Model Training")

freq_formula = "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + np.log(Density)"
sev_formula  = "ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + np.log(Density)"

st.write("Training Poisson frequency model...")
freq_pois = glm(freq_formula, data=df, family=Poisson()).fit()

st.write("Training Negative Binomial frequency model...")
freq_nb = glm(freq_formula, data=df, family=NegativeBinomial()).fit()

st.write("Training Zero-Inflated Poisson frequency model...")
zip_exog = df[["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]]
zip_endog = df["ClaimNb"]
freq_zip = ZeroInflatedPoisson(zip_endog, zip_exog, exog_infl=zip_exog, inflation="logit").fit(disp=0)

st.write("Training Gamma severity model (ClaimNb > 0)...")
sev_data = df[df["ClaimNb"] > 0].copy()
sev = glm(sev_formula, data=sev_data, family=Gamma()).fit()

st.success("All models trained.")

# =====================================================
# MODEL COMPARISON (Poisson vs NB vs ZIP)
# =====================================================
st.header("2. Model Comparison (Frequency)")

col_aic1, col_aic2, col_aic3 = st.columns(3)
with col_aic1:
    st.metric("Poisson AIC", f"{freq_pois.aic:.1f}")
with col_aic2:
    st.metric("NegBin AIC", f"{freq_nb.aic:.1f}")
with col_aic3:
    st.metric("ZIP LogLik", f"{freq_zip.llf:.1f}")

st.write("Lower AIC (or higher log-likelihood) indicates better fit. Use this to justify Poisson vs NB vs ZIP.")

# =====================================================
# PORTFOLIO PREDICTIONS
# =====================================================
st.header("3. Portfolio-Level Predictions")

df["pred_freq_pois"] = freq_pois.predict(df)
df["pred_freq_nb"] = freq_nb.predict(df)

# For ZIP, need same exog structure
df_zip_exog = df[["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]]
df["pred_freq_zip"] = freq_zip.predict(df_zip_exog)

df["pred_sev"] = sev.predict(df)
df["pure_premium_pois"] = df["pred_freq_pois"] * df["pred_sev"]
df["pure_premium_nb"] = df["pred_freq_nb"] * df["pred_sev"]
df["pure_premium_zip"] = df["pred_freq_zip"] * df["pred_sev"]

st.write(df[["pred_freq_pois", "pred_freq_nb", "pred_freq_zip", "pred_sev", "pure_premium_pois"]].head())

# =====================================================
# TERRITORIAL HEATMAP (BY REGION)
# =====================================================
st.header("4. Territorial Premium View (by Region)")

region_summary = (
    df.groupby("Region")[["pure_premium_pois", "pure_premium_nb", "pure_premium_zip"]]
    .mean()
    .reset_index()
)

# =====================================================
# 1️⃣ FREQUENCY MODEL — BASE MODEL
# =====================================================

st.header("1️⃣ Frequency Modeling (Base Poisson GLM)")

base_freq_model = glm(
    formula="ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=df,
    family=Poisson(),
    offset=np.log(df["Exposure"])
).fit()

df["Pred_Freq_Base"] = base_freq_model.predict(df, offset=np.log(df["Exposure"]))

st.write("Base Model AIC:", round(base_freq_model.aic,2))

# =====================================================
# 2️⃣ TERRITORIAL FREQUENCY MODEL
# =====================================================

st.header("2️⃣ Frequency Modeling with Territory")

territorial_freq_model = glm(
    formula="ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + np.log(Density)",
    data=df,
    family=Poisson(),
    offset=np.log(df["Exposure"])
).fit()

df["Pred_Freq_Terr"] = territorial_freq_model.predict(df, offset=np.log(df["Exposure"]))

st.write("Territorial Model AIC:", round(territorial_freq_model.aic,2))

if territorial_freq_model.aic < base_freq_model.aic:
    st.success("Territorial variables improve model fit ✅")
else:
    st.warning("Territorial variables did not improve model fit ⚠️")

# =====================================================
# TERRITORIAL RELATIVITIES
# =====================================================

st.subheader("Territorial Risk Relativities")

territory_freq = (
    df.groupby("Region")["ClaimNb"].sum() /
    df.groupby("Region")["Exposure"].sum()
)

st.bar_chart(territory_freq)

# =====================================================
# 3️⃣ SEVERITY MODEL
# =====================================================

st.header("3️⃣ Severity Modeling (Gamma GLM)")

sev_df = df[df["ClaimAmount"] > 0].copy()

base_sev_model = glm(
    formula="ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=sev_df,
    family=Gamma()
).fit()

territorial_sev_model = glm(
    formula="ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region)",
    data=sev_df,
    family=Gamma()
).fit()

df["Pred_Sev"] = territorial_sev_model.predict(df)

st.write("Severity AIC (Base):", round(base_sev_model.aic,2))
st.write("Severity AIC (Territory):", round(territorial_sev_model.aic,2))
aic_diff = base_sev_model.aic - territorial_sev_model.aic

if aic_diff > 10:
    st.success("Strong evidence territorial factors improve severity modeling.")


# =====================================================
# 4️⃣ PURE PREMIUM CALCULATION
# =====================================================

st.header("4️⃣ Pure Premium Calculation")

df["Pure_Premium"] = df["Pred_Freq_Terr"] * df["Pred_Sev"]

st.write("Average Pure Premium:", df["Pure_Premium"])

fig = plt.figure(figsize=(10,5))
plt.hist(df["Pure_Premium"], bins=50)
plt.title("Pure Premium Distribution (Territorial Adjusted)")
st.pyplot(fig)

# =====================================================
# 5️⃣ MACHINE LEARNING BENCHMARK
# =====================================================

st.header("5️⃣ Machine Learning Severity Benchmark")

X = sev_df[["VehPower","VehAge","DrivAge","BonusMalus","Density"]]
y = sev_df["ClaimAmount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

st.write("Random Forest RMSE:", round(np.sqrt(mean_squared_error(y_test, rf_pred)),2))
st.write("XGBoost RMSE:", round(np.sqrt(mean_squared_error(y_test, xgb_pred)),2))

# =====================================================
# 6️⃣ SHAP EXPLAINABILITY
# =====================================================

st.header("6️⃣ SHAP Feature Importance")

explainer = shap.Explainer(rf)
shap_values = explainer(X_test[:200])

fig_shap = plt.figure(figsize=(10,5))
shap.plots.bar(shap_values, show=False)
st.pyplot(fig_shap)

# =====================================================
# 7️⃣ MONTE CARLO AGGREGATE LOSS SIMULATION
# =====================================================

st.header("7️⃣ Monte Carlo Aggregate Loss Simulation")

n_sim = st.slider("Number of Simulations", 1000, 10000, 5000)

sim_losses = []

for _ in range(n_sim):
    freq_sim = np.random.poisson(df["Pred_Freq_Terr"].mean())
    sev_sim = np.random.gamma(shape=2, scale=df["Pred_Sev"].mean()/2)
    sim_losses.append(freq_sim * sev_sim)

var95 = np.percentile(sim_losses,95)
var99 = np.percentile(sim_losses,99)

st.write("VaR 95%:", round(var95,2))
st.write("VaR 99%:", round(var99,2))

fig2 = plt.figure(figsize=(10,5))
plt.hist(sim_losses, bins=50)
plt.title("Simulated Aggregate Loss Distribution")
st.pyplot(fig2)
