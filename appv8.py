import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson, Gamma
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import TableStyle
from reportlab.lib.units import inch

st.set_page_config(layout="wide")
st.title("📊 Actuarial Insurance Modeling Platform")

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
        "VehPower", "VehAge", "DrivAge", "BonusMalus"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)]
    df = df.dropna()

    return df

df = load_data()

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
# TAB 2: PRICING MODELS
# =====================================================

with tabs[1]:

    st.subheader("Frequency Model (Poisson)")

    freq_model = glm(
        formula="ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=df,
        family=Poisson(),
        offset=np.log(df["Exposure"])
    ).fit()

    df["Pred_Freq"] = freq_model.predict(df, offset=np.log(df["Exposure"]))

    st.write("Average Predicted Frequency:", round(df["Pred_Freq"].mean(),4))

    fig2 = plt.figure(figsize=(12,6))
    plt.hist(np.log1p(df["Pred_Freq"]), bins=50)
    plt.title("Predicted Frequency Distribution")
    st.pyplot(fig2)

    st.subheader("Severity Model (Gamma)")

    sev_df = df[df["ClaimAmount"] > 0].copy()

    sev_model = glm(
        formula="ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=sev_df,
        family=Gamma()
    ).fit()

    df["Pred_Sev"] = sev_model.predict(df)

    st.write("Average Predicted Severity:", round(df["Pred_Sev"].mean(),2))
    fig1 = plt.figure(figsize=(12,6))
    plt.hist(df["Pred_Sev"], bins=50)
    plt.title("Predicted Severity Distribution")
    st.pyplot(fig1)


# =====================================================
# TAB 3: PREMIUM & ADJUSTMENTS
# =====================================================

with tabs[2]:

    deductible = st.slider("Deductible", 0, 5000, 500)
    inflation_rate = st.slider("Inflation %", 0.0, 0.2, 0.05)

    df["Adj_Sev"] = np.maximum(df["Pred_Sev"] - deductible, 0)
    df["Adj_Sev"] *= (1 + inflation_rate)

    df["Pure_Premium"] = df["Pred_Freq"] * df["Adj_Sev"]

    st.write("Average Pure Premium:", round(df["Pure_Premium"].mean(),2))

    fig2 = plt.figure(figsize=(12,6))
    plt.hist(df["Pure_Premium"], bins=50)
    plt.title("Pure Premium Distribution")
    st.pyplot(fig2)

# =====================================================
# TAB 4: MACHINE LEARNING
# =====================================================

with tabs[3]:

    sev_df = df[df["ClaimAmount"] > 0].copy()

    X = sev_df[["VehPower", "VehAge", "DrivAge", "BonusMalus"]]
    y = sev_df["ClaimAmount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    st.write("Random Forest RMSE:", round(rf_rmse,2))

    explainer = shap.Explainer(rf)
    shap_values = explainer(X_test[:200])

    fig3 = plt.figure(figsize=(12,6))
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig3)

# =====================================================
# TAB 5: IBNR
# =====================================================

with tabs[4]:

    triangle = np.array([
        [100,150,180],
        [120,160,0],
        [130,0,0]
    ])

    dev_factors = [
        triangle[0][1]/triangle[0][0],
        triangle[0][2]/triangle[0][1]
    ]

    ultimate = []
    for i,row in enumerate(triangle):
        last_val = max(row)
        factor = np.prod(dev_factors[:len(dev_factors)-i])
        ultimate.append(round(last_val * factor,2))

    st.write("Estimated Ultimate Losses:", ultimate)

# =====================================================
# TAB 6: RISK SIMULATION
# =====================================================

with tabs[5]:

    n_sim = st.slider("Simulations", 1000, 10000, 5000)
    sim_losses = []

    for _ in range(n_sim):
        freq_sim = np.random.poisson(df["Pred_Freq"].mean())
        sev_sim = np.random.gamma(2, df["Pred_Sev"].mean()/2)
        sim_losses.append(freq_sim * sev_sim)

    var95 = np.percentile(sim_losses,95)
    var99 = np.percentile(sim_losses,99)

    st.write("VaR 95%:", round(var95,2))
    st.write("VaR 99%:", round(var99,2))

# =====================================================
# TAB 7: REINSURANCE
# =====================================================

with tabs[6]:

    quota = st.slider("Quota Share %", 0.0,1.0,0.3)
    retention = st.slider("Retention", 1000,10000,5000)

    total_loss = sum(sim_losses)
    st.write("Ceded (Quota):", round(total_loss * quota,2))
    st.write("Excess Loss Ceded:",
             round(sum([max(loss-retention,0) for loss in sim_losses]),2))

# =====================================================
# TAB 8: REPORTING
# =====================================================

with tabs[7]:

    if st.button("Generate PDF Report"):

        doc = SimpleDocTemplate("Actuarial_Report.pdf")
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Actuarial Modeling Summary", styles['Heading1']))
        elements.append(Spacer(1, 0.3*inch))

        report_data = [
            ["Metric","Value"],
            ["Avg Frequency", str(round(df["Pred_Freq"].mean(),4))],
            ["Avg Severity", str(round(df["Pred_Sev"].mean(),2))],
            ["Avg Premium", str(round(df["Pure_Premium"].mean(),2))]
        ]

        table = Table(report_data)
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('GRID',(0,0),(-1,-1),1,colors.black)
        ]))

        elements.append(table)
        doc.build(elements)

        st.success("PDF Generated Successfully!")
