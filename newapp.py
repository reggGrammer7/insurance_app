import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson, Gamma, NegativeBinomial
from statsmodels.discrete.count_model import ZeroInflatedPoisson

from sklearn.ensemble import RandomForestRegressor
from io import StringIO

st.set_page_config(layout="wide")
st.title("📊 Advanced Motor Insurance Pricing Platform")

# =====================================================
# DATA LOADING
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("freMTPL2freq.csv")  # adjust path
    df["Density_log"] = np.log(df["Density"])
    return df

df = load_data()

# =====================================================
# MODEL TRAINING
# =====================================================
st.header("1. Model Training")

freq_formula = "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + Density_log"
sev_formula  = "ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + Density_log"

st.write("Training Poisson frequency model...")
freq_pois = glm(freq_formula, data=df, family=Poisson()).fit()

st.write("Training Negative Binomial frequency model...")
freq_nb = glm(freq_formula, data=df, family=NegativeBinomial()).fit()

st.write("Training Zero-Inflated Poisson frequency model...")
zip_exog = df[["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density_log"]]
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
df_zip_exog = df[["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density_log"]]
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

st.write("Average pure premium by Region (Poisson-based):")
fig, ax = plt.subplots(figsize=(8, 4))
region_summary.plot(
    x="Region",
    y="pure_premium_pois",
    kind="bar",
    ax=ax,
    legend=False,
    color="orange",
)
ax.set_ylabel("Average Pure Premium")
ax.set_title("Average Pure Premium by Region (Poisson)")
st.pyplot(fig)

# =====================================================
# SHAP EXPLAINABILITY (Random Forest on Frequency)
# =====================================================
st.header("5. SHAP Explainability (Frequency Model)")

st.write("Training a Random Forest on ClaimNb for SHAP explanations (not for pricing, just interpretation).")

features = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density_log"]
X = df[features]
y = df["ClaimNb"]

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

st.write("Global feature importance (SHAP summary):")
fig_shap = plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig_shap)

# =====================================================
# NEW CUSTOMER PREDICTION + LOADINGS
# =====================================================
st.header("6. New Customer Pricing with Loadings")

col1, col2 = st.columns(2)
with col1:
    VehPower = st.number_input("Vehicle Power", 1, 20, 5)
    VehAge = st.number_input("Vehicle Age", 0, 30, 5)
    DrivAge = st.number_input("Driver Age", 18, 90, 35)
with col2:
    BonusMalus = st.number_input("Bonus-Malus", 50, 200, 100)
    Region = st.selectbox("Region", df["Region"].unique())
    Density = st.number_input("Density", 1, 100000, 3000)

st.subheader("Loadings")
col_l1, col_l2, col_l3 = st.columns(3)
with col_l1:
    expense_loading = st.number_input("Expense Loading (%)", 0.0, 100.0, 20.0)
with col_l2:
    profit_loading = st.number_input("Profit Loading (%)", 0.0, 100.0, 10.0)
with col_l3:
    reinsurance_loading = st.number_input("Reinsurance Loading (%)", 0.0, 100.0, 5.0)

if st.button("Compute Premium for New Customer"):
    new_data = pd.DataFrame({
        "VehPower": [VehPower],
        "VehAge": [VehAge],
        "DrivAge": [DrivAge],
        "BonusMalus": [BonusMalus],
        "Region": [Region],
        "Density": [Density],
        "Density_log": [np.log(Density)]
    })

    # choose Poisson as base pricing model (you can switch to NB/ZIP)
    base_freq = freq_pois.predict(new_data)[0]
    base_sev = sev.predict(new_data)[0]
    pure_premium = base_freq * base_sev

    total_loading_factor = (
        1
        + expense_loading / 100.0
        + profit_loading / 100.0
        + reinsurance_loading / 100.0
    )
    technical_premium = pure_premium * total_loading_factor

    st.subheader("Pricing Results")
    st.write(f"**Predicted Frequency (λ, Poisson):** {base_freq:.4f}")
    st.write(f"**Predicted Severity (μ):** {base_sev:.2f}")
    st.write(f"**Pure Premium:** {pure_premium:.2f}")
    st.write(f"**Total Loading Factor:** {total_loading_factor:.3f}")
    st.write(f"**Technical Premium (with loadings):** {technical_premium:.2f}")

# =====================================================
# DOWNLOADABLE PREMIUM TABLE
# =====================================================
st.header("7. Downloadable Premium Table")

download_cols = [
    "VehPower", "VehAge", "DrivAge", "BonusMalus", "Region", "Density",
    "pred_freq_pois", "pred_freq_nb", "pred_freq_zip",
    "pred_sev", "pure_premium_pois", "pure_premium_nb", "pure_premium_zip"
]
premium_table = df[download_cols].copy()

csv_buffer = StringIO()
premium_table.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download Premium Table (CSV)",
    data=csv_buffer.getvalue(),
    file_name="premium_table.csv",
    mime="text/csv",
)
