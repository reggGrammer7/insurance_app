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
import shap
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import TableStyle
from reportlab.lib.units import inch

st.set_page_config(layout="wide")
st.title("📊 Actuarial Insurance Modeling Platform – Elite Edition")

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
# TABS
# =====================================================

tabs = st.tabs([
    "📂 Data",
    "📈 Pricing Models",
    "🐍 Pareto Tail Modeling",
    "💰 Premium",
    "🤖 ML",
    "⚠️ Capital Simulation",
    "🛡 Reinsurance",
    "📄 Report"
])

# =====================================================
# TAB 1 – DATA
# =====================================================

with tabs[0]:
    st.write("Records:", len(df))
    st.dataframe(df.head())

# =====================================================
# TAB 2 – GLM PRICING
# =====================================================

with tabs[1]:

    st.subheader("Frequency Model (Poisson GLM)")

    freq_model = glm(
        formula="ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=df,
        family=Poisson(),
        offset=np.log(df["Exposure"])
    ).fit()

    df["Pred_Freq"] = freq_model.predict(df, offset=np.log(df["Exposure"]))

    st.write("Average Frequency:", round(df["Pred_Freq"].mean(), 4))

    fig = plt.figure(figsize=(10,5))
    plt.hist(df["Pred_Freq"], bins=50)
    plt.title("Predicted Frequency")
    st.pyplot(fig)

    st.subheader("Severity Model (Gamma – Body)")

    sev_df = df[df["ClaimAmount"] > 0].copy()

    sev_model = glm(
        formula="ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=sev_df,
        family=Gamma()
    ).fit()

    df["Pred_Sev_Gamma"] = sev_model.predict(df)

    st.write("Average Gamma Severity:", round(df["Pred_Sev_Gamma"].mean(),2))

# =====================================================
# TAB 3 – PARETO TAIL MODELING
# =====================================================

with tabs[2]:

    st.subheader("Spliced Gamma–Pareto Severity Model")

    positive_claims = df[df["ClaimAmount"] > 0]["ClaimAmount"]

    threshold = st.slider("Tail Threshold Percentile", 90, 99, 95)
    u = np.percentile(positive_claims, threshold)

    body = positive_claims[positive_claims <= u]
    tail = positive_claims[positive_claims > u]

    # Pareto MLE
    def fit_pareto(tail_data, threshold):
        excess = tail_data / threshold
        alpha = len(excess) / np.sum(np.log(excess))
        return alpha

    alpha_hat = fit_pareto(tail, u)

    st.write("Threshold (u):", round(u,2))
    st.write("Estimated Pareto Alpha:", round(alpha_hat,4))

    # Plot Tail Fit
    fig = plt.figure(figsize=(10,6))
    plt.hist(tail, bins=30, density=True, alpha=0.6)

    x = np.linspace(u, tail.max(), 200)
    pareto_pdf = alpha_hat * u**alpha_hat / x**(alpha_hat+1)
    plt.plot(x, pareto_pdf, 'r-', linewidth=2)

    plt.title("Pareto Tail Fit")
    st.pyplot(fig)

# =====================================================
# TAB 4 – PREMIUM
# =====================================================

with tabs[3]:

    deductible = st.slider("Deductible", 0, 5000, 500)
    inflation = st.slider("Inflation %", 0.0, 0.2, 0.05)

    df["Adj_Sev"] = np.maximum(df["Pred_Sev_Gamma"] - deductible, 0)
    df["Adj_Sev"] *= (1 + inflation)

    df["Pure_Premium"] = df["Pred_Freq"] * df["Adj_Sev"]

    st.write("Average Pure Premium:", round(df["Pure_Premium"].mean(),2))

    fig = plt.figure(figsize=(10,5))
    plt.hist(np.log1p(df["Pure_Premium"]), bins=50)
    plt.title("Premium Distribution (Log Scale)")
    st.pyplot(fig)

# =====================================================
# TAB 5 – ML
# =====================================================

with tabs[4]:

    X = sev_df[["VehPower", "VehAge", "DrivAge", "BonusMalus"]]
    y = sev_df["ClaimAmount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    st.write("Random Forest RMSE:", round(rmse,2))

    explainer = shap.Explainer(rf)
    shap_values = explainer(X_test[:200])

    fig = plt.figure(figsize=(10,6))
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)

# =====================================================
# TAB 6 – CAPITAL SIMULATION (WITH PARETO)
# =====================================================

with tabs[5]:

    st.subheader("Capital Simulation with Pareto Tail")

    n_sim = st.slider("Simulations", 1000, 20000, 5000)

    p_tail = len(tail) / len(positive_claims)

    def simulate_pareto(n, alpha, threshold):
        U = np.random.uniform(size=n)
        return threshold * (1 - U) ** (-1/alpha)

    sim_losses = []

    for _ in range(n_sim):

        freq_sim = np.random.poisson(df["Pred_Freq"].mean())

        sev_total = 0

        for _ in range(freq_sim):
            if np.random.rand() < p_tail:
                sev_total += simulate_pareto(1, alpha_hat, u)[0]
            else:
                sev_total += np.random.choice(body)

        sim_losses.append(sev_total)

    var95 = np.percentile(sim_losses, 95)
    var995 = np.percentile(sim_losses, 99.5)

    st.write("VaR 95%:", round(var95,2))
    st.write("VaR 99.5% (Solvency II):", round(var995,2))

    fig = plt.figure(figsize=(10,5))
    plt.hist(np.log1p(sim_losses), bins=50)
    plt.title("Simulated Aggregate Loss Distribution")
    st.pyplot(fig)

# =====================================================
# TAB 7 – REINSURANCE
# =====================================================

with tabs[6]:

    quota = st.slider("Quota Share %", 0.0,1.0,0.3)
    retention = st.slider("Excess Retention", 1000,20000,5000)

    total_loss = sum(sim_losses)

    quota_ceded = total_loss * quota
    excess_ceded = sum([max(loss-retention,0) for loss in sim_losses])

    st.write("Quota Share Ceded:", round(quota_ceded,2))
    st.write("Excess of Loss Ceded:", round(excess_ceded,2))

# =====================================================
# TAB 8 – REPORT
# =====================================================

with tabs[7]:

    if st.button("Generate PDF"):

        doc = SimpleDocTemplate("Actuarial_Report.pdf")
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Elite Actuarial Model Report", styles['Heading1']))
        elements.append(Spacer(1, 0.3*inch))

        data = [
            ["Metric","Value"],
            ["Avg Frequency", str(round(df["Pred_Freq"].mean(),4))],
            ["Avg Gamma Severity", str(round(df["Pred_Sev_Gamma"].mean(),2))],
            ["Pareto Alpha", str(round(alpha_hat,4))],
            ["VaR 99.5%", str(round(var995,2))]
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('GRID',(0,0),(-1,-1),1,colors.black)
        ]))

        elements.append(table)
        doc.build(elements)

        st.success("PDF Generated!")
