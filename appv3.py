# actuarial_pricing_platform.py

import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import TableStyle

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Actuarial Pricing Platform", layout="wide")

st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Professional Actuarial Pricing & Claims Modeling Platform")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "data" not in st.session_state:
    st.session_state.data = None

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

menu = st.sidebar.radio("Workflow", [
    "Data Input",
    "Frequency GLM",
    "Severity Modeling",
    "Deductible & Truncation",
    "Compound Modeling",
    "Portfolio Segmentation",
    "Dashboard KPIs",
    "Export Actuarial Report"
])

# --------------------------------------------------
# 1️⃣ DATA INPUT
# --------------------------------------------------

if menu == "Data Input":

    st.header("Insurance Portfolio Data")

    n = st.slider("Number of Policies", 200, 5000, 1000)

    if st.button("Generate Portfolio"):

        exposure = np.random.uniform(0.5, 1.5, n)
        region = np.random.choice(["Urban", "Rural"], n)
        vehicle_type = np.random.choice(["Sedan", "SUV", "Truck"], n)

        freq = stats.nbinom.rvs(2, 0.6, size=n)
        severity = stats.gamma.rvs(a=2, scale=3000, size=n)

        df = pd.DataFrame({
            "Exposure": exposure,
            "Region": region,
            "Vehicle_Type": vehicle_type,
            "Claim_Count": freq,
            "Claim_Severity": severity
        })

        df["Total_Loss"] = df["Claim_Count"] * df["Claim_Severity"]

        st.session_state.data = df
        st.success("Portfolio generated.")

    if st.session_state.data is not None:
        st.dataframe(st.session_state.data.head())

# --------------------------------------------------
# 2️⃣ FREQUENCY GLM WITH OFFSET
# --------------------------------------------------

if menu == "Frequency GLM" and st.session_state.data is not None:

    st.header("Frequency GLM (Poisson with Exposure Offset)")

    df = st.session_state.data.copy()
    df["log_exposure"] = np.log(df["Exposure"])

    formula = "Claim_Count ~ Region + Vehicle_Type"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=df["log_exposure"]
    ).fit()

    st.text(model.summary())

    st.session_state.freq_model = model

# --------------------------------------------------
# 3️⃣ SEVERITY MODELING
# --------------------------------------------------

if menu == "Severity Modeling" and st.session_state.data is not None:

    st.header("Severity Distribution Fitting")

    severity = st.session_state.data["Claim_Severity"]

    dist_name = st.selectbox("Select Distribution", 
                             ["Gamma", "Lognormal", "Weibull", "Pareto"])

    if dist_name == "Gamma":
        params = stats.gamma.fit(severity)
        dist = stats.gamma
    elif dist_name == "Lognormal":
        params = stats.lognorm.fit(severity)
        dist = stats.lognorm
    elif dist_name == "Weibull":
        params = stats.weibull_min.fit(severity)
        dist = stats.weibull_min
    else:
        params = stats.pareto.fit(severity)
        dist = stats.pareto

    loglik = np.sum(dist.logpdf(severity, *params))
    aic = 2*len(params) - 2*loglik
    bic = np.log(len(severity))*len(params) - 2*loglik

    st.write(f"LogLikelihood: {loglik:.2f}")
    st.write(f"AIC: {aic:.2f}")
    st.write(f"BIC: {bic:.2f}")

    x = np.linspace(min(severity), max(severity), 500)
    pdf = dist.pdf(x, *params)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=severity, histnorm='probability density'))
    fig.add_trace(go.Scatter(x=x, y=pdf))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig)

    st.session_state.sev_model = (dist, params)

# --------------------------------------------------
# 4️⃣ DEDUCTIBLE & TRUNCATION
# --------------------------------------------------

if menu == "Deductible & Truncation" and st.session_state.data is not None:

    st.header("Policy Structure Adjustments")

    deductible = st.number_input("Deductible", value=500.0)
    truncation = st.number_input("Upper Truncation Limit", value=20000.0)

    df = st.session_state.data.copy()

    df["Adjusted_Severity"] = np.clip(
        np.maximum(df["Claim_Severity"] - deductible, 0),
        0,
        truncation
    )

    st.write("Adjusted Severity Summary")
    st.write(df["Adjusted_Severity"].describe())

    st.session_state.data = df

# --------------------------------------------------
# 5️⃣ COMPOUND MODELING
# --------------------------------------------------

if menu == "Compound Modeling" and st.session_state.data is not None:

    st.header("Compound Loss Modeling")

    df = st.session_state.data

    expected_frequency = df["Claim_Count"].mean()
    expected_severity = df["Claim_Severity"].mean()

    pure_premium = expected_frequency * expected_severity

    st.metric("Expected Frequency", round(expected_frequency, 3))
    st.metric("Expected Severity", round(expected_severity, 2))
    st.metric("Pure Premium (Loss Cost)", round(pure_premium, 2))

    df["Expected_Loss"] = pure_premium * df["Exposure"]

    st.session_state.pure_premium = pure_premium

# --------------------------------------------------
# 6️⃣ PORTFOLIO SEGMENTATION
# --------------------------------------------------

if menu == "Portfolio Segmentation" and st.session_state.data is not None:

    st.header("Portfolio Segmentation")

    df = st.session_state.data

    segment = st.selectbox("Segment By", ["Region", "Vehicle_Type"])

    summary = df.groupby(segment).agg({
        "Claim_Count": "mean",
        "Claim_Severity": "mean",
        "Total_Loss": "sum"
    })

    st.dataframe(summary)

# --------------------------------------------------
# 7️⃣ DASHBOARD KPIs
# --------------------------------------------------

if menu == "Dashboard KPIs" and st.session_state.data is not None:

    st.header("Portfolio KPI Dashboard")

    df = st.session_state.data

    total_premium = df["Exposure"].sum() * st.session_state.get("pure_premium", 1)
    total_loss = df["Total_Loss"].sum()

    loss_ratio = total_loss / total_premium if total_premium > 0 else 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Premium", round(total_premium, 2))
    col2.metric("Total Loss", round(total_loss, 2))
    col3.metric("Loss Ratio", round(loss_ratio, 3))

# --------------------------------------------------
# 8️⃣ EXPORT PDF REPORT
# --------------------------------------------------

if menu == "Export Actuarial Report" and st.session_state.data is not None:

    st.header("Generate Actuarial Pricing Report")

    if st.button("Export PDF Report"):

        doc = SimpleDocTemplate("Actuarial_Report.pdf")
        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("Actuarial Pricing Report", styles["Title"]))
        elements.append(Spacer(1, 0.5 * inch))

        df = st.session_state.data

        summary_data = [
            ["Metric", "Value"],
            ["Total Policies", len(df)],
            ["Total Loss", round(df["Total_Loss"].sum(), 2)],
            ["Average Frequency", round(df["Claim_Count"].mean(), 3)],
            ["Average Severity", round(df["Claim_Severity"].mean(), 2)]
        ]

        table = Table(summary_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))

        elements.append(table)

        doc.build(elements)

        st.success("Actuarial_Report.pdf generated.")
