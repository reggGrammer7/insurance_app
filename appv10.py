import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.inspection import permutation_importance

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

st.set_page_config(layout="wide")
st.title("Enterprise Insurance Pricing & Capital Platform")

# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------
@st.cache_data
def load_data():
    freq = fetch_openml(name="freMTPL2freq", version=1, as_frame=True)
    sev = fetch_openml(name="freMTPL2sev", version=1, as_frame=True)

    df = freq.frame.merge(sev.frame, how="left", on="IDpol")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)
    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)]

    df["Severity"] = df["ClaimAmount"] / df["ClaimNb"]
    df["Severity"] = df["Severity"].replace([np.inf, -np.inf], 0).fillna(0)

    return df

df = load_data()

# ---------------------------------------------------
# FEATURE SETUP
# ---------------------------------------------------
target_freq = df["ClaimNb"]
target_sev = df["Severity"]

features = df.drop(columns=["ClaimNb","ClaimAmount","Severity","IDpol"])

cat_cols = features.select_dtypes(include=["object","category"]).columns.tolist()
num_cols = features.select_dtypes(exclude=["object","category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

X_train, X_test, y_train_freq, y_test_freq = train_test_split(
    features, target_freq, test_size=0.2, random_state=42
)

_, _, y_train_sev, y_test_sev = train_test_split(
    features, target_sev, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# MODEL TOGGLE
# ---------------------------------------------------
model_choice = st.sidebar.radio("Use GLM or ML for Pricing?",
                                 ["GLM", "ML"])

if model_choice == "GLM":
    model_freq = Pipeline([
        ("prep", preprocessor),
        ("model", PoissonRegressor(alpha=0.01))
    ])
else:
    model_freq = Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor())
    ])

model_freq.fit(X_train, y_train_freq)
freq_pred = model_freq.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test_freq, freq_pred))
st.sidebar.metric("Model RMSE", round(rmse,4))

# ---------------------------------------------------
# PREMIUM CALCULATION
# ---------------------------------------------------
severity_mean = target_sev.mean()
pure_premium = freq_pred * severity_mean
technical_premium = pure_premium * 1.30

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pricing",
    "Model Validation",
    "Capital Modeling",
    "Reinsurance",
    "Board Report"
])

# ===================================================
# PRICING TAB
# ===================================================
with tab1:
    st.header("Premium Distribution")

    fig = plt.figure()
    plt.hist(technical_premium, bins=50)
    plt.title("Technical Premium Distribution")
    st.pyplot(fig)

    st.write("Average Premium:", round(np.mean(technical_premium),2))

    # CSV Upload
    st.subheader("Upload New Business CSV")
    uploaded = st.file_uploader("Upload file", type=["csv"])

    if uploaded:
        new_data = pd.read_csv(uploaded)
        new_pred = model_freq.predict(new_data)
        new_premium = new_pred * severity_mean * 1.30
        st.write("Predicted Premiums")
        st.write(new_premium[:10])

# ===================================================
# MODEL VALIDATION
# ===================================================
with tab2:
    st.header("Lift Curve")

    df_lift = pd.DataFrame({
        "Actual": y_test_freq,
        "Predicted": freq_pred
    }).sort_values("Predicted", ascending=False)

    df_lift["CumActual"] = df_lift["Actual"].cumsum()
    df_lift["Percentile"] = np.arange(len(df_lift))/len(df_lift)

    fig_lift = plt.figure()
    plt.plot(df_lift["Percentile"], df_lift["CumActual"])
    plt.title("Lift Curve")
    st.pyplot(fig_lift)

    st.subheader("Feature Importance")

    result = permutation_importance(
        model_freq, X_test, y_test_freq,
        n_repeats=5, random_state=42
    )

    fig_imp = plt.figure()
    plt.barh(range(len(result.importances_mean)),
             result.importances_mean)
    plt.title("Permutation Importance")
    st.pyplot(fig_imp)

# ===================================================
# CAPITAL MODELING
# ===================================================
with tab3:
    st.header("Monte Carlo Capital Model")

    n_sim = 5000
    simulated_losses = []

    for _ in range(n_sim):
        freq_sim = np.random.poisson(freq_pred.mean())
        sev_sim = np.random.gamma(2, severity_mean/2)
        simulated_losses.append(freq_sim * sev_sim)

    simulated_losses = np.array(simulated_losses)

    var_995 = np.percentile(simulated_losses, 99.5)
    tvar_995 = simulated_losses[simulated_losses >= var_995].mean()

    st.metric("VaR 99.5%", round(var_995,2))
    st.metric("Tail VaR 99.5%", round(tvar_995,2))

    fig_cap = plt.figure()
    plt.hist(simulated_losses, bins=60)
    plt.title("Loss Distribution")
    st.pyplot(fig_cap)

# ===================================================
# REINSURANCE OPTIMIZATION
# ===================================================
with tab4:
    st.header("Reinsurance Optimization")

    retention_levels = np.linspace(1000,20000,20)
    capital_results = []

    for retention in retention_levels:
        reins_losses = np.minimum(simulated_losses, retention)
        capital_results.append(np.percentile(reins_losses, 99.5))

    optimal_retention = retention_levels[np.argmin(capital_results)]
    st.metric("Optimal Retention", round(optimal_retention,2))

    fig_re = plt.figure()
    plt.plot(retention_levels, capital_results)
    plt.title("Retention vs VaR")
    st.pyplot(fig_re)

    # Stress Testing
    st.subheader("Stress Scenario")

    stressed_losses = simulated_losses * 1.3
    var_stress = np.percentile(stressed_losses, 99.5)

    st.metric("Stress VaR 99.5%", round(var_stress,2))

# ===================================================
# BOARD REPORT
# ===================================================
with tab5:
    st.header("Generate Board PDF Report")

    def generate_pdf():
        doc = SimpleDocTemplate("Board_Report.pdf")
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Insurance Risk & Capital Report",
                                  styles["Title"]))
        elements.append(Spacer(1, 0.5*inch))

        data = [
            ["Metric","Value"],
            ["RMSE", round(rmse,4)],
            ["Average Premium", round(np.mean(technical_premium),2)],
            ["VaR 99.5%", round(var_995,2)],
            ["TVaR 99.5%", round(tvar_995,2)],
            ["Optimal Retention", round(optimal_retention,2)],
            ["Stress VaR 99.5%", round(var_stress,2)]
        ]

        elements.append(Table(data))
        doc.build(elements)

    if st.button("Generate PDF Report"):
        generate_pdf()
        st.success("Board report generated successfully.")

