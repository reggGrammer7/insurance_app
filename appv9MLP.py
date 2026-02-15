# ===============================
# INSURANCE PRICING ENGINE APP
# PRODUCTION-READY VERSION
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import shap

st.set_page_config(layout="wide")

st.title("🚀 Insurance Pricing & Risk Platform")

# ===============================
# DATA LOADING
# ===============================
@st.cache_data
def load_data():
    freq = fetch_openml(name="freMTPL2freq", version=1, as_frame=True)
    sev = fetch_openml(name="freMTPL2sev", version=1, as_frame=True)

    df = freq.frame.merge(sev.frame, how="left", on="IDpol")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)
    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)]
    return df

df = load_data()

# ===============================
# FEATURES & MODEL PREP
# ===============================

features = ["Power", "CarAge", "DriverAge", "Brand", "Gas", "Region"]
df_model = df[features + ["ClaimNb", "ClaimAmount", "Exposure"]].copy()
df_model = pd.get_dummies(df_model, drop_first=True)

X = df_model.drop(["ClaimNb","ClaimAmount"], axis=1)
y_freq = df_model["ClaimNb"]/df_model["Exposure"]
y_sev = df_model["ClaimAmount"]

X_train, X_test, y_freq_train, y_freq_test = train_test_split(X, y_freq, test_size=0.3, random_state=42)
_, _, y_sev_train, y_sev_test = train_test_split(X, y_sev, test_size=0.3, random_state=42)

# ===============================
# TAB LAYOUT
# ===============================

tabs = st.tabs([
    "Model Training", "Pricing Engine", "Lift Curve",
    "Explainability", "New Business Upload", "Profit & Capital", "Model Validation"
])

# ===============================
# TAB 1: MODEL TRAINING
# ===============================
with tabs[0]:
    st.header("Model Training")

    # GLM Frequency
    X_train_glm = sm.add_constant(X_train)
    glm_freq = sm.GLM(y_freq_train, X_train_glm,
                      family=sm.families.Poisson())
    glm_freq_result = glm_freq.fit()

    # GLM Severity
    glm_sev = sm.GLM(y_sev_train,
                     sm.add_constant(X_train),
                     family=sm.families.Gamma(link=sm.families.links.log()))
    glm_sev_result = glm_sev.fit()

    # ML Severity
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf_model.fit(X_train, y_sev_train)

    sev_pred_glm = glm_sev_result.predict(sm.add_constant(X_test))
    sev_pred_ml = rf_model.predict(X_test)

    rmse_glm = np.sqrt(mean_squared_error(y_sev_test, sev_pred_glm))
    rmse_ml = np.sqrt(mean_squared_error(y_sev_test, sev_pred_ml))

    col1, col2 = st.columns(2)
    col1.metric("GLM Severity RMSE", round(rmse_glm,2))
    col2.metric("ML Severity RMSE", round(rmse_ml,2))

# ===============================
# TAB 2: PRICING ENGINE
# ===============================
with tabs[1]:
    st.header("Pricing Engine")

    pricing_model = st.radio("Select Pricing Engine:", ["GLM Pricing","ML Severity Pricing"])

    freq_pred = glm_freq_result.predict(sm.add_constant(X_test))
    sev_pred = sev_pred_glm if pricing_model=="GLM Pricing" else sev_pred_ml

    premium = freq_pred * sev_pred

    df_pricing = pd.DataFrame({
        "Frequency": freq_pred,
        "Severity": sev_pred,
        "Premium": premium
    })
    st.write(df_pricing.head())

    # Compare premiums
    premium_glm = freq_pred * sev_pred_glm
    premium_ml = freq_pred * sev_pred_ml
    diff = premium_ml - premium_glm

    fig = plt.figure(figsize=(10,6))
    plt.hist(diff, bins=50)
    plt.title("Premium Difference: ML - GLM")
    st.pyplot(fig)

# ===============================
# TAB 3: LIFT CURVE
# ===============================
with tabs[2]:
    st.header("Lift Curve")
    actual = y_sev_test.reset_index(drop=True)
    predicted = sev_pred_ml

    lift_df = pd.DataFrame({"Actual": actual, "Predicted": predicted})
    lift_df = lift_df.sort_values("Predicted", ascending=False)
    lift_df["Cumulative Actual"] = lift_df["Actual"].cumsum()
    lift_df["Cumulative %"] = lift_df["Cumulative Actual"] / lift_df["Actual"].sum()

    fig2 = plt.figure(figsize=(10,6))
    plt.plot(lift_df["Cumulative %"].values)
    plt.title("Lift Curve (ML Severity)")
    st.pyplot(fig2)

# ===============================
# TAB 4: EXPLAINABILITY
# ===============================
with tabs[3]:
    st.header("Feature Importance & SHAP")

    # Feature importance
    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    fig3 = plt.figure(figsize=(10,6))
    importance.plot(kind="bar")
    plt.title("Top 15 Feature Importance")
    st.pyplot(fig3)

    # SHAP summary
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test[:500])
    fig4 = plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test[:500], show=False)
    st.pyplot(fig4)

# ===============================
# TAB 5: NEW BUSINESS UPLOAD
# ===============================
with tabs[4]:
    st.header("Score New Business CSV")

    uploaded_file = st.file_uploader("Upload CSV with same columns as training data", type=["csv"])
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        new_df_encoded = pd.get_dummies(new_df, drop_first=True)
        missing_cols = set(X.columns) - set(new_df_encoded.columns)
        for col in missing_cols:
            new_df_encoded[col] = 0
        new_df_encoded = new_df_encoded[X.columns]  # align columns

        freq_new = glm_freq_result.predict(sm.add_constant(new_df_encoded))
        sev_new = rf_model.predict(new_df_encoded) if pricing_model=="ML Severity Pricing" else glm_sev_result.predict(sm.add_constant(new_df_encoded))
        premium_new = freq_new * sev_new

        new_df["Pred_Freq"] = freq_new
        new_df["Pred_Sev"] = sev_new
        new_df["Premium"] = premium_new
        st.dataframe(new_df.head())

# ===============================
# TAB 6: PROFITABILITY & CAPITAL
# ===============================
with tabs[5]:
    st.header("Portfolio Profitability & Capital Simulation")

    st.write("Simulate profitability and capital requirements based on predicted premiums")

    n_sim = st.slider("Number of Simulations", 1000, 10000, 5000)
    total_losses = []

    for _ in range(n_sim):
        freq_sim = np.random.poisson(freq_pred.mean())
        sev_sim = np.random.gamma(2, sev_pred.mean()/2)
        total_losses.append(freq_sim*sev_sim)

    total_premium = premium.sum()
    expected_loss = np.mean(total_losses)
    profit = total_premium - expected_loss

    var95 = np.percentile(total_losses,95)
    var99 = np.percentile(total_losses,99)

    st.metric("Expected Portfolio Profit", round(profit,2))
    st.metric("VaR 95%", round(var95,2))
    st.metric("VaR 99%", round(var99,2))

    fig5 = plt.figure(figsize=(10,6))
    plt.hist(total_losses, bins=50)
    plt.title("Portfolio Loss Simulation")
    st.pyplot(fig5)

# ===============================
# TAB 7: MODEL VALIDATION DASHBOARD
# ===============================
with tabs[6]:
    st.header("Model Validation")

    st.write("Compare Actual vs Predicted Severity")
    df_val = pd.DataFrame({
        "Actual": y_sev_test.reset_index(drop=True),
        "Pred_GLM": sev_pred_glm,
        "Pred_ML": sev_pred_ml
    })
    st.dataframe(df_val.head())

    fig6 = plt.figure(figsize=(10,6))
    plt.scatter(df_val["Pred_GLM"], df_val["Actual"], alpha=0.5, label="GLM")
    plt.scatter(df_val["Pred_ML"], df_val["Actual"], alpha=0.5, label="ML")
    plt.plot([0, df_val["Actual"].max()], [0, df_val["Actual"].max()], "k--")
    plt.xlabel("Predicted Severity")
    plt.ylabel("Actual Severity")
    plt.legend()
    plt.title("Actual vs Predicted Severity")
    st.pyplot(fig6)

    rmse_glm_val = np.sqrt(mean_squared_error(df_val["Actual"], df_val["Pred_GLM"]))
    rmse_ml_val = np.sqrt(mean_squared_error(df_val["Actual"], df_val["Pred_ML"]))
    st.metric("Validation RMSE GLM", round(rmse_glm_val,2))
    st.metric("Validation RMSE ML", round(rmse_ml_val,2))
