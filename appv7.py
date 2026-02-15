# ===============================================================
# ENTERPRISE ACTUARIAL INSURANCE ANALYTICS PLATFORM
# WITH REINSURANCE + ENHANCED VISUALIZATION
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

st.set_page_config(layout="wide")
st.title("📊 Enterprise Actuarial Insurance Platform")

# ===============================================================
# LOAD REAL FREMTPL2 DATA
# ===============================================================

@st.cache_data
def load_data():
    freq = fetch_openml(name="freMTPL2freq", version=1, as_frame=True)
    sev = fetch_openml(name="freMTPL2sev", version=1, as_frame=True)

    df = freq.frame.merge(sev.frame, how="left", on="IDpol")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    # Exposure validation
    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)]
    return df

df = load_data()

# Portfolio Metrics
total_exposure = df["Exposure"].sum()
total_claims = df["ClaimNb"].sum()
total_loss = df["ClaimAmount"].sum()
frequency = total_claims / total_exposure
severity = total_loss / total_claims if total_claims > 0 else 0
pure_premium = total_loss / total_exposure

# ===============================================================
# SIDEBAR NAVIGATION
# ===============================================================

module = st.sidebar.selectbox(
    "Select Module",
    [
        "Dashboard",
        "Truncated Severity",
        "Inflation Adjustment",
        "IBNR Reserving",
        "ML + SHAP",
        "Reinsurance Modeling",
        "Generate Actuarial Report"
    ]
)

# ===============================================================
# 1️⃣ DASHBOARD (BIGGER GRAPHS)
# ===============================================================

if module == "Dashboard":

    col1, col2, col3, col4, col5= st.columns(5)
    col1.metric("Frequency", round(frequency,4))
    col2.metric("Severity", round(severity,2))
    col3.metric("Pure Premium", round(pure_premium,2))
    col4.metric("Total Loss", round(total_loss,2))
    col5.metric("Total Exposure", round(total_exposure,2))


    fig = px.histogram(
        df[df["ClaimAmount"]>0],
        x="ClaimAmount",
        nbins=50,
        title="Claim Severity Distribution"
    )
    fig.update_layout(
        height=600,
        width=1000,
        font=dict(size=16)
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# 2️⃣ TRUNCATED SEVERITY
# ===============================================================

elif module == "Truncated Severity":

    threshold = st.slider("Truncation Threshold", 0, 10000, 500)

    sev_df = df[df["ClaimAmount"] > threshold]

    model = smf.glm(
        "ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=sev_df,
        family=smf.families.Gamma()
    ).fit()

    st.text(model.summary())

# ===============================================================
# 3️⃣ INFLATION ADJUSTMENT 
# ===============================================================

elif module == "Inflation Adjustment":

    inflation_rate = st.slider("Inflation Rate (%)", 0.0, 15.0, 5.0)/100
    years = st.slider("Years Forward", 1, 10, 3)

    factor = (1+inflation_rate)**years
    adjusted = df["ClaimAmount"] * factor

    fig = px.histogram(
        adjusted[adjusted>0],
        nbins=150,
        title="Inflation Adjusted Severity Distribution"
    )
    fig.update_layout(height=600, width=1000)
    st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# 4️⃣ IBNR RESERVING
# ===============================================================

elif module == "IBNR Reserving":

    triangle = pd.DataFrame({
        0: [1000,1200,1500,1300],
        1: [1500,1700,2000,None],
        2: [1800,2100,None,None],
        3: [2000,None,None,None]
    })

    st.write("Loss Triangle")
    st.dataframe(triangle)

    factors = []
    for i in range(3):
        factors.append(triangle[i+1].sum(skipna=True)/triangle[i].sum(skipna=True))

    latest = triangle.iloc[:,-1].fillna(0)
    ultimate = latest.sum() * np.prod(factors)
    paid = triangle.sum().sum()
    ibnr = ultimate - paid

    st.write("IBNR Estimate:", round(ibnr,2))

# ===============================================================
# 5️⃣ ML + SHAP (CLEARER)
# ===============================================================

elif module == "ML + SHAP":

    df["pure_premium"] = df["ClaimAmount"]
    features = ["VehPower","VehAge","DrivAge","BonusMalus"]

    X = df[features]
    y = df["pure_premium"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = XGBRegressor()
    model.fit(X_train,y_train)

    preds = model.predict(X_test)
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test,preds)))

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:500])

    plt.figure(figsize=(12,6))
    shap.plots.bar(shap_values, show=False)
    st.pyplot(plt.gcf())

# ===============================================================
# 6️⃣ REINSURANCE MODELING
# ===============================================================

elif module == "Reinsurance Modeling":

    st.subheader("Quota Share Reinsurance")

    quota = st.slider("Quota Share %", 0.0, 100.0, 30.0)/100

    ceded_loss = total_loss * quota
    retained_loss = total_loss * (1-quota)

    st.write("Ceded Loss:", round(ceded_loss,2))
    st.write("Retained Loss:", round(retained_loss,2))

    st.subheader("Excess of Loss Reinsurance")

    retention = st.slider("Retention Level", 0, 20000, 5000)

    claims = df["ClaimAmount"]
    ceded_xol = np.sum(np.maximum(claims - retention,0))
    retained_xol = total_loss - ceded_xol

    st.write("Ceded (XoL):", round(ceded_xol,2))
    st.write("Retained (XoL):", round(retained_xol,2))

    fig = px.histogram(claims[claims>0], nbins=150, title="Claims Before Reinsurance")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# 7️⃣ ACTUARIAL PDF REPORT
# ===============================================================

elif module == "Generate Actuarial Report":

    if st.button("Generate Report"):

        doc = SimpleDocTemplate("Actuarial_Report.pdf")
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Enterprise Actuarial Portfolio Report", styles["Title"]))
        elements.append(Spacer(1, 0.3*inch))

        data = [
            ["Metric","Value"],
            ["Frequency", round(frequency,4)],
            ["Severity", round(severity,2)],
            ["Pure Premium", round(pure_premium,2)]
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('GRID',(0,0),(-1,-1),1,colors.black)
        ]))

        elements.append(table)
        doc.build(elements)

        with open("Actuarial_Report.pdf","rb") as f:
            st.download_button("Download Report",f,"Actuarial_Report.pdf")
