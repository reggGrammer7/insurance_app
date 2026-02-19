import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson, Gamma, NegativeBinomial
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.datasets import fetch_openml
import shap
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import TableStyle
from reportlab.lib.units import inch
from statsmodels.discrete.count_model import ZeroInflatedPoisson

st.set_page_config(layout="wide")
st.title("📊 End-to-End Actuarial Insurance Modeling Platform")

# =====================================================
# DATA LOADING (CORRECTED OPENML VERSION)
# =====================================================

@st.cache_data
def load_data():
    freq = fetch_openml(name="freMTPL2freq", version=1, as_frame=True)
    sev = fetch_openml(name="freMTPL2sev", version=1, as_frame=True)

    df = freq.frame.merge(sev.frame, how="left", on="IDpol")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    numeric_cols = [
        "ClaimNb", "Exposure", "ClaimAmount",
        "VehPower", "VehAge", "DrivAge", "BonusMalus"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)]
    df = df.dropna()

    return df

df = load_data()

st.success(f"Data loaded successfully. Records: {len(df)}")
st.dataframe(df.head())


# =====================================================
# CREATE TABS
# =====================================================

tabs = st.tabs([
    "📂 Data Overview",
    "📈 Pricing Models",
    "💰 Premium & Adjustments",
    "🤖 ML & Explainability",
    "📊 IBNR Reserving",
    "⚠️ Risk Simulation",
    "🛡 Reinsurance",
    "📄 Reporting"
])

# =====================================================
# TAB 1: DATA
# =====================================================

with tabs[0]:
    st.header("Data Overview")
    st.write("Number of records:", len(df))
    st.dataframe(df.head())



# =====================================================
# MODEL TRAINING
# =====================================================
st.header("1. Model Training")

freq_formula = "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region)"
sev_formula  = "ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region)"

st.write("Training Poisson frequency model...")
freq_pois = glm(freq_formula, data=df, family=Poisson()).fit()

st.write("Training Negative Binomial frequency model...")
freq_nb = glm(freq_formula, data=df, family=NegativeBinomial()).fit()

st.write("Training Zero-Inflated Poisson frequency model...")
zip_exog = df[["VehPower", "VehAge", "DrivAge", "BonusMalus"]]
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
    st.metric("ZIP AIC", f"{freq_zip.aic:.1f}")

st.write("Lower AIC (or higher log-likelihood) indicates better fit. Use this to justify Poisson vs NB vs ZIP.")


# =====================================================
# FREQUENCY MODEL (POISSON GLM)
# =====================================================

st.header("1️⃣ Frequency Modeling (Poisson GLM)")

freq_model = glm(
    formula="ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=df,
    family=Poisson(),
    offset=np.log(df["Exposure"])
).fit()

df["Pred_Freq"] = freq_model.predict(df, offset=np.log(df["Exposure"]))

st.write(" Predicted Frequency:", df["Pred_Freq"])

fig1 = plt.figure(figsize=(12,6))
plt.hist(df["Pred_Freq"], bins=50)
plt.title("Predicted Frequency Distribution")
st.pyplot(fig1)

# =====================================================
# SEVERITY MODEL (GAMMA GLM)
# =====================================================

st.header("2️⃣ Severity Modeling (Gamma GLM)")

sev_df = df[df["ClaimAmount"] > 0].copy()

sev_model = glm(
    formula="ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=sev_df,
    family=Gamma()
).fit()

df["Pred_Sev"] = sev_model.predict(df)

st.write("Predicted Severity:", df["Pred_Sev"])

fig2 = plt.figure(figsize=(12,6))
plt.hist(sev_df["ClaimAmount"], bins=50)
plt.title("Observed Severity Distribution")
st.pyplot(fig2)

# =====================================================
# TRUNCATED SEVERITY
# =====================================================

st.header("3️⃣ Truncated Severity")

threshold = st.slider("Truncation Threshold", 0, 2000, 500)
trunc_df = sev_df[sev_df["ClaimAmount"] > threshold]

st.write("Mean Before:", round(sev_df["ClaimAmount"].mean(),2))
st.write("Mean After:", round(trunc_df["ClaimAmount"].mean(),2))

# =====================================================
# DEDUCTIBLE MODELING
# =====================================================

st.header("4️⃣ Deductible Modeling")

deductible = st.slider("Deductible", 0, 5000, 500)
sev_df["After_Deductible"] = np.maximum(sev_df["ClaimAmount"] - deductible, 0)

st.write("Expected Severity Before:", round(sev_df["ClaimAmount"].mean(),2))
st.write("Expected Severity After:", round(sev_df["After_Deductible"].mean(),2))

# =====================================================
# INFLATION ADJUSTMENT
# =====================================================

st.header("5️⃣ Inflation Trend Adjustment")

inflation_rate = st.slider("Annual Inflation %", 0.0, 0.2, 0.05)
years = st.slider("Projection Years", 1, 5, 2)

infl_factor = (1 + inflation_rate) ** years
df["Inflated_Sev"] = df["Pred_Sev"] * infl_factor

st.write("Inflation Factor:", round(infl_factor,3))

# =====================================================
# PURE PREMIUM
# =====================================================

st.header("6️⃣ Pure Premium Calculation")

df["Pure_Premium"] = df["Pred_Freq"] * df["Inflated_Sev"]

st.write("Average Pure Premium:", round(df["Pure_Premium"].mean(),2))

fig3 = plt.figure(figsize=(12,6))
plt.hist(df["Pure_Premium"], bins=50)
plt.title("Pure Premium Distribution")
st.pyplot(fig3)


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
# MACHINE LEARNING BENCHMARK
# =====================================================

st.header("7️⃣ Machine Learning Benchmark")

ml_df = sev_df.copy()
X = ml_df[["VehPower", "VehAge", "DrivAge", "BonusMalus"]]
y = ml_df["ClaimAmount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

st.write("Random Forest RMSE:", round(rf_rmse,2))
st.write("XGBoost RMSE:", round(xgb_rmse,2))

# =====================================================
# SHAP EXPLAINABILITY
# =====================================================

st.header("8️⃣ SHAP Explainability")

explainer = shap.Explainer(rf)
shap_values = explainer(X_test[:200])

fig_shap = plt.figure(figsize=(12,6))
shap.plots.bar(shap_values, show=False)
st.pyplot(fig_shap)

# =====================================================
# IBNR RESERVING (CHAIN LADDER)
# =====================================================

st.header("9️⃣ IBNR Reserving")

triangle = np.array([
    [100,150,180],
    [120,160,0],
    [130,0,0]
])

development_factors = [
    triangle[0][1]/triangle[0][0],
    triangle[0][2]/triangle[0][1]
]

ultimate = []
for i,row in enumerate(triangle):
    last_val = max(row)
    factor = np.prod(development_factors[:len(development_factors)-i])
    ultimate.append(round(last_val * factor,2))

st.write("Estimated Ultimate Losses:", ultimate)

# =====================================================
# MONTE CARLO RISK SIMULATION
# =====================================================

st.header("🔟 Monte Carlo Risk Simulation")

n_sim = st.slider("Simulations", 1000, 10000, 5000)

sim_losses = []

for _ in range(n_sim):
    freq_sim = np.random.poisson(df["Pred_Freq"].mean())
    sev_sim = np.random.gamma(shape=2, scale=df["Pred_Sev"].mean()/2)
    sim_losses.append(freq_sim * sev_sim)

var95 = np.percentile(sim_losses,95)
var99 = np.percentile(sim_losses,99)

st.write("VaR 95%:", round(var95,2))
st.write("VaR 99%:", round(var99,2))

fig4 = plt.figure(figsize=(12,6))
plt.hist(sim_losses, bins=50)
plt.title("Simulated Aggregate Loss Distribution")
st.pyplot(fig4)

# =====================================================
# REINSURANCE MODELING
# =====================================================

st.header("1️⃣1️⃣ Reinsurance Modeling")

quota = st.slider("Quota Share %", 0.0,1.0,0.3)
retention = st.slider("Excess of Loss Retention", 1000,10000,5000)

total_loss = sum(sim_losses)
ceded_quota = total_loss * quota
retained_quota = total_loss - ceded_quota
excess_loss = sum([max(loss-retention,0) for loss in sim_losses])

st.write("Quota Share Ceded:", round(ceded_quota,2))
st.write("Quota Share Retained:", round(retained_quota,2))
st.write("Excess of Loss Ceded:", round(excess_loss,2))

# =====================================================
# PDF REPORT
# =====================================================

st.header("📄 Generate Actuarial Report")

if st.button("Generate PDF Report"):

    doc = SimpleDocTemplate("Actuarial_Report.pdf")
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Actuarial Modeling Report", styles['Heading1']))
    elements.append(Spacer(1, 0.3*inch))

    report_data = [
        ["Metric","Value"],
        ["Average Frequency", str(round(df["Pred_Freq"].mean(),4))],
        ["Average Severity", str(round(df["Pred_Sev"].mean(),2))],
        ["Pure Premium", str(round(df["Pure_Premium"].mean(),2))],
        ["VaR 95%", str(round(var95,2))],
        ["VaR 99%", str(round(var99,2))]
    ]

    table = Table(report_data)
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.grey),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ]))

    elements.append(table)
    doc.build(elements)

    st.success("PDF Generated Successfully!")
