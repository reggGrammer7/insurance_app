import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson, Gamma
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap
from scipy.stats import gamma
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Table
from reportlab.lib.units import inch
from reportlab.platypus import TableStyle

st.set_page_config(layout="wide")

st.title("📊 End-to-End Actuarial Insurance Modeling Platform")

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("freMTPL2.csv")  # Place dataset in same folder
    return df

df = load_data()

st.subheader("Raw Data Snapshot")
st.dataframe(df.head())

# =====================================================
# EXPOSURE VALIDATION
# =====================================================

df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)]
st.success(f"Valid Exposure Records: {len(df)}")

# =====================================================
# FREQUENCY MODEL
# =====================================================

st.header("1️⃣ Frequency Modeling (Poisson GLM)")

freq_model = glm(
    formula="ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=df,
    family=Poisson(),
    offset=np.log(df["Exposure"])
).fit()

st.text(freq_model.summary())

df["Pred_Freq"] = freq_model.predict(df, offset=np.log(df["Exposure"]))

# =====================================================
# SEVERITY MODEL
# =====================================================

st.header("2️⃣ Severity Modeling (Gamma GLM)")

sev_df = df[df["ClaimAmount"] > 0]

sev_model = glm(
    formula="ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=sev_df,
    family=Gamma()
).fit()

st.text(sev_model.summary())

sev_df["Pred_Sev"] = sev_model.predict(sev_df)

# =====================================================
# TRUNCATED SEVERITY
# =====================================================

st.header("3️⃣ Truncated Severity Modeling")

threshold = st.slider("Truncation Threshold", 0, 2000, 500)

trunc_df = sev_df[sev_df["ClaimAmount"] > threshold]
mean_before = sev_df["ClaimAmount"].mean()
mean_after = trunc_df["ClaimAmount"].mean()

st.write("Mean Severity Before Truncation:", round(mean_before,2))
st.write("Mean Severity After Truncation:", round(mean_after,2))

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

st.header("5️⃣ Inflation Adjustment")

inflation_rate = st.slider("Annual Inflation %", 0.0, 0.2, 0.05)
years = st.slider("Projection Years", 1, 5, 2)

infl_factor = (1 + inflation_rate) ** years
sev_df["Inflated_Claim"] = sev_df["ClaimAmount"] * infl_factor

st.write("Inflation Factor:", round(infl_factor,3))

# =====================================================
# PURE PREMIUM
# =====================================================

st.header("6️⃣ Pure Premium Calculation")

df["Pure_Premium"] = df["Pred_Freq"] * sev_model.predict(df)

st.dataframe(df[["Pred_Freq"]].head())

# =====================================================
# MACHINE LEARNING BENCHMARK
# =====================================================

st.header("7️⃣ Machine Learning Benchmark")

ml_df = sev_df.copy()

X = ml_df[["VehPower", "VehAge", "DrivAge", "BonusMalus"]]
y = ml_df["ClaimAmount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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

st.header("9️⃣ IBNR Reserving (Chain Ladder)")

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
    ultimate.append(last_val * factor)

st.write("Estimated Ultimate Losses:", ultimate)

# =====================================================
# MONTE CARLO RISK SIMULATION
# =====================================================

st.header("🔟 Monte Carlo Risk Simulation")

n_sim = st.slider("Number of Simulations", 1000, 10000, 5000)

sim_losses = []

for _ in range(n_sim):
    freq = np.random.poisson(df["Pred_Freq"].mean())
    sev = np.random.gamma(shape=2, scale=1000)
    sim_losses.append(freq * sev)

var95 = np.percentile(sim_losses,95)
var99 = np.percentile(sim_losses,99)

st.write("VaR 95%:", round(var95,2))
st.write("VaR 99%:", round(var99,2))

fig = plt.figure(figsize=(10,6))
plt.hist(sim_losses, bins=50)
plt.title("Simulated Loss Distribution")
st.pyplot(fig)

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
# PDF REPORT GENERATION
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
        ["Average Severity", str(round(sev_df["ClaimAmount"].mean(),2))],
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

