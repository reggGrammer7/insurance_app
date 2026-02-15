# ===============================
# INSURANCE PRICING ENGINE APP
# GLM vs ML Pricing Comparison
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

# ===============================
# DATA LOADING
# ===============================

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

st.title("Insurance Pricing Engine: GLM vs ML")

# ===============================
# FEATURE ENGINEERING
# ===============================

df["Frequency"] = df["ClaimNb"] / df["Exposure"]

features = ["Power", "CarAge", "DriverAge", "Brand", "Gas", "Region"]

df_model = df[features + ["Frequency", "ClaimAmount", "Exposure"]].copy()
df_model = pd.get_dummies(df_model, drop_first=True)

X = df_model.drop(["Frequency", "ClaimAmount"], axis=1)
y_freq = df_model["Frequency"]
y_sev = df_model["ClaimAmount"]

X_train, X_test, y_freq_train, y_freq_test = train_test_split(X, y_freq, test_size=0.3, random_state=42)
_, _, y_sev_train, y_sev_test = train_test_split(X, y_sev, test_size=0.3, random_state=42)

# ===============================
# TAB LAYOUT
# ===============================

tab1, tab2, tab3, tab4 = st.tabs(["Model Training", "Pricing Engine", "Lift Curve", "Explainability"])

# ===============================
# TAB 1: MODEL TRAINING
# ===============================

with tab1:

    st.header("Model Training")

    # GLM Frequency
    X_train_glm = sm.add_constant(X_train)
    glm_freq = sm.GLM(y_freq_train, X_train_glm,
                      family=sm.families.Poisson(),
                      offset=np.log(X_train["Exposure"]))
    glm_freq_result = glm_freq.fit()

    # GLM Severity
    glm_sev = sm.GLM(y_sev_train,
                     sm.add_constant(X_train),
                     family=sm.families.Gamma(link=sm.families.links.log()))
    glm_sev_result = glm_sev.fit()

    # ML Severity Model
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf_model.fit(X_train, y_sev_train)

    # Evaluate
    sev_pred_glm = glm_sev_result.predict(sm.add_constant(X_test))
    sev_pred_ml = rf_model.predict(X_test)

    rmse_glm = np.sqrt(mean_squared_error(y_sev_test, sev_pred_glm))
    rmse_ml = np.sqrt(mean_squared_error(y_sev_test, sev_pred_ml))

    col1, col2 = st.columns(2)
    col1.metric("GLM Severity RMSE", round(rmse_glm, 2))
    col2.metric("ML Severity RMSE", round(rmse_ml, 2))

# ===============================
# TAB 2: PRICING ENGINE
# ===============================

with tab2:

    st.header("Pricing Engine")

    pricing_model = st.radio("Select Pricing Engine:",
                             ["GLM Pricing", "ML Severity Pricing"])

    # Frequency prediction (always GLM)
    freq_pred = glm_freq_result.predict(sm.add_constant(X_test),
                                        offset=np.log(X_test["Exposure"]))

    # Severity selection
    if pricing_model == "GLM Pricing":
        sev_pred = sev_pred_glm
    else:
        sev_pred = sev_pred_ml

    premium = freq_pred * sev_pred

    df_pricing = pd.DataFrame({
        "Frequency": freq_pred,
        "Severity": sev_pred,
        "Premium": premium
    })

    st.write(df_pricing.head())

    # Premium comparison
    premium_glm = freq_pred * sev_pred_glm
    premium_ml = freq_pred * sev_pred_ml

    diff = premium_ml - premium_glm

    fig = plt.figure(figsize=(10,6))
    plt.hist(diff, bins=50)
    plt.title("Premium Difference: ML - GLM", fontsize=14)
    plt.xlabel("Premium Difference")
    plt.ylabel("Count")
    st.pyplot(fig)

# ===============================
# TAB 3: LIFT CURVE
# ===============================

with tab3:

    st.header("Lift Curve")

    actual = y_sev_test.reset_index(drop=True)
    predicted = sev_pred_ml

    lift_df = pd.DataFrame({
        "Actual": actual,
        "Predicted": predicted
    })

    lift_df = lift_df.sort_values("Predicted", ascending=False)
    lift_df["Cumulative Actual"] = lift_df["Actual"].cumsum()
    lift_df["Cumulative %"] = lift_df["Cumulative Actual"] / lift_df["Actual"].sum()

    fig2 = plt.figure(figsize=(10,6))
    plt.plot(lift_df["Cumulative %"].values)
    plt.title("Lift Curve (ML Severity)", fontsize=14)
    plt.xlabel("Policies (sorted by predicted risk)")
    plt.ylabel("Cumulative % of Loss")
    st.pyplot(fig2)

# ===============================
# TAB 4: EXPLAINABILITY
# ===============================

with tab4:

    st.header("Feature Importance & SHAP")

    # Feature Importance
    importance = pd.Series(rf_model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False).head(15)

    fig3 = plt.figure(figsize=(10,6))
    importance.plot(kind="bar")
    plt.title("Top 15 Feature Importance (Random Forest)", fontsize=14)
    st.pyplot(fig3)

    # SHAP
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test[:500])

    fig4 = plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test[:500], show=False)
    st.pyplot(fig4)

