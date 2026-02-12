# insurance_claims_modeling_app.py

import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.discrete.count_model as cm
import json
#import os
import plotly.graph_objects as go

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Insurance Claims Modeling Platform",
    layout="wide"
)

# Dark Theme Styling
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Insurance Claims Modeling Platform")
st.markdown("Professional Frequency & Severity Modeling for Actuarial Analysis")

# -----------------------------
# Sidebar Navigation
# -----------------------------
menu = st.sidebar.radio("Workflow Navigation", [
    "Data Input",
    "Frequency Modeling",
    "Severity Modeling",
    "Model Comparison",
    "Saved Models"
])

# -----------------------------
# Session Storage
# -----------------------------
if "data" not in st.session_state:
    st.session_state.data = None
if "frequency_results" not in st.session_state:
    st.session_state.frequency_results = {}
if "severity_results" not in st.session_state:
    st.session_state.severity_results = {}

# -----------------------------
# DATA INPUT
# -----------------------------
if menu == "Data Input":

    st.header("📊 Claims Data Input")

    data_option = st.radio("Select Data Source", ["Generate Sample Data", "Manual Entry"])

    if data_option == "Generate Sample Data":
        n = st.slider("Number of Policies", 100, 5000, 1000)

        # Frequency (Negative Binomial-like)
        freq = stats.nbinom.rvs(2, 0.5, size=n)

        # Severity (Gamma)
        severity = stats.gamma.rvs(a=2, scale=2000, size=n)

        df = pd.DataFrame({
            "Claim_Count": freq,
            "Claim_Severity": severity
        })

        st.session_state.data = df
        st.success("Sample insurance dataset generated.")

    else:
        claim_count = st.text_area("Enter Claim Counts (comma separated)")
        claim_severity = st.text_area("Enter Claim Severities (comma separated)")

        if st.button("Load Data"):
            freq = np.array([int(x) for x in claim_count.split(",")])
            sev = np.array([float(x) for x in claim_severity.split(",")])
            df = pd.DataFrame({
                "Claim_Count": freq,
                "Claim_Severity": sev
            })
            st.session_state.data = df
            st.success("Data loaded successfully.")

    if st.session_state.data is not None:
        st.subheader("Preview Data")
        st.dataframe(st.session_state.data.head())

# -----------------------------
# FREQUENCY MODELING
# -----------------------------
if menu == "Frequency Modeling" and st.session_state.data is not None:

    st.header("📈 Frequency Modeling")

    y = st.session_state.data["Claim_Count"]

    model_choice = st.selectbox("Select Frequency Model", [
        "Poisson",
        "Negative Binomial",
        "Zero-Inflated Poisson",
        "Zero-Inflated Negative Binomial"
    ])

    X = np.ones((len(y), 1))

    if st.button("Fit Model"):

        if model_choice == "Poisson":
            model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

        elif model_choice == "Negative Binomial":
            model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()

        elif model_choice == "Zero-Inflated Poisson":
            model = cm.ZeroInflatedPoisson(y, X).fit()

        else:
            model = cm.ZeroInflatedNegativeBinomialP(y, X).fit()

        llf = model.llf
        aic = model.aic
        bic = model.bic

        st.session_state.frequency_results[model_choice] = {
            "LogLikelihood": llf,
            "AIC": aic,
            "BIC": bic
        }

        st.success(f"{model_choice} fitted successfully.")

        st.write("### Model Summary")
        st.text(model.summary())

# -----------------------------
# SEVERITY MODELING
# -----------------------------
if menu == "Severity Modeling" and st.session_state.data is not None:

    st.header("💰 Severity Modeling")

    severity = st.session_state.data["Claim_Severity"]

    model_choice = st.selectbox("Select Severity Distribution", [
        "Gamma",
        "Lognormal",
        "Weibull",
        "Pareto"
    ])

    if st.button("Fit Severity Model"):

        if model_choice == "Gamma":
            params = stats.gamma.fit(severity)

        elif model_choice == "Lognormal":
            params = stats.lognorm.fit(severity)

        elif model_choice == "Weibull":
            params = stats.weibull_min.fit(severity)

        else:
            params = stats.pareto.fit(severity)

        dist = getattr(stats, model_choice.lower() if model_choice != "Weibull" else "weibull_min")

        loglik = np.sum(dist.logpdf(severity, *params))
        k = len(params)
        n = len(severity)
        aic = 2*k - 2*loglik
        bic = np.log(n)*k - 2*loglik

        st.session_state.severity_results[model_choice] = {
            "Parameters": params,
            "LogLikelihood": loglik,
            "AIC": aic,
            "BIC": bic
        }

        st.success(f"{model_choice} distribution fitted.")

        # Plot
        x = np.linspace(min(severity), max(severity), 500)
        pdf = dist.pdf(x, *params)

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=severity, histnorm='probability density', name="Observed"))
        fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name="Fitted PDF"))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig)

# -----------------------------
# MODEL COMPARISON
# -----------------------------
if menu == "Model Comparison":

    st.header("📑 Model Comparison")

    if st.session_state.frequency_results:
        st.subheader("Frequency Models")
        st.dataframe(pd.DataFrame(st.session_state.frequency_results).T)

    if st.session_state.severity_results:
        st.subheader("Severity Models")
        st.dataframe(pd.DataFrame(st.session_state.severity_results).T)

# -----------------------------
# SAVE MODELS
# -----------------------------
if menu == "Saved Models":

    st.header("💾 Save Fitted Models")

    if st.button("Save All Models"):

        models = {
            "Frequency": st.session_state.frequency_results,
            "Severity": st.session_state.severity_results
        }

        with open("saved_models.json", "w") as f:
            json.dump(models, f, indent=4)

        st.success("Models saved to saved_models.json")
