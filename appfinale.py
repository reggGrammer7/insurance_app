import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import glm, ols
from statsmodels.genmod.families import Gamma, NegativeBinomial, Poisson
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


st.set_page_config(layout="wide")
st.title("Actuarial Insurance Modeling Platform - Final")


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

    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)].dropna()
    df["Observed_Freq"] = df["ClaimNb"] / df["Exposure"]
    df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)
    return df


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_lift_table(actual, predicted):
    lift_df = pd.DataFrame({"actual": actual, "predicted": predicted}).sort_values(
        "predicted", ascending=False
    )
    lift_df["cum_actual"] = lift_df["actual"].cumsum()
    total_actual = max(lift_df["actual"].sum(), 1e-9)
    lift_df["cum_actual_pct"] = lift_df["cum_actual"] / total_actual
    lift_df["cum_book_pct"] = np.arange(1, len(lift_df) + 1) / len(lift_df)
    return lift_df


def chain_ladder_ibnr(triangle_df):
    tri = triangle_df.copy()
    n = tri.shape[1]
    dev_factors = []
    for j in range(n - 1):
        num = tri.iloc[:, j + 1].dropna().sum()
        den = tri.iloc[:, j].dropna().sum()
        dev_factors.append((num / den) if den > 0 else 1.0)

    completed = tri.copy()
    for i in range(completed.shape[0]):
        row = completed.iloc[i, :]
        last_idx = row.last_valid_index()
        if last_idx is None:
            continue
        last_pos = completed.columns.get_loc(last_idx)
        for j in range(last_pos + 1, n):
            prev_val = completed.iloc[i, j - 1]
            completed.iloc[i, j] = prev_val * dev_factors[j - 1]

    latest = []
    ultimate = []
    ibnr = []
    for i in range(completed.shape[0]):
        obs = tri.iloc[i, :].dropna()
        latest_val = obs.iloc[-1] if len(obs) > 0 else 0.0
        ult_val = completed.iloc[i, -1] if pd.notna(completed.iloc[i, -1]) else latest_val
        latest.append(float(latest_val))
        ultimate.append(float(ult_val))
        ibnr.append(float(ult_val - latest_val))

    summary = pd.DataFrame(
        {"Latest": latest, "Ultimate": ultimate, "IBNR": ibnr},
        index=tri.index
    )
    return dev_factors, completed, summary


df = load_data()
features = ["VehPower", "VehAge", "DrivAge", "BonusMalus"]

# Shared train/test split for comparisons
train_idx, test_idx = train_test_split(df.index, test_size=0.3, random_state=42)
train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()

# Frequency models: Poisson vs Negative Binomial
freq_poisson = glm(
    formula="ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=train_df,
    family=Poisson(),
    offset=np.log(train_df["Exposure"])
).fit()

freq_nb = glm(
    formula="ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=train_df,
    family=NegativeBinomial(),
    offset=np.log(train_df["Exposure"])
).fit()

freq_poisson_test_count = freq_poisson.predict(test_df, offset=np.log(test_df["Exposure"]))
freq_poisson_test_annual = np.clip(freq_poisson_test_count / test_df["Exposure"], 0, None)
freq_nb_test_count = freq_nb.predict(test_df, offset=np.log(test_df["Exposure"]))
freq_nb_test_annual = np.clip(freq_nb_test_count / test_df["Exposure"], 0, None)

freq_results = pd.DataFrame(
    [
        {
            "Model": "Poisson GLM",
            "RMSE": rmse(test_df["Observed_Freq"], freq_poisson_test_annual),
            "MAE": float(mean_absolute_error(test_df["Observed_Freq"], freq_poisson_test_annual)),
        },
        {
            "Model": "Negative Binomial GLM",
            "RMSE": rmse(test_df["Observed_Freq"], freq_nb_test_annual),
            "MAE": float(mean_absolute_error(test_df["Observed_Freq"], freq_nb_test_annual)),
        },
    ]
).sort_values("RMSE")

best_freq_model_name = freq_results.iloc[0]["Model"]
if best_freq_model_name == "Poisson GLM":
    df["Pred_Freq_Annual"] = np.clip(
        freq_poisson.predict(df, offset=np.log(df["Exposure"])) / df["Exposure"], 0, None
    )
else:
    df["Pred_Freq_Annual"] = np.clip(
        freq_nb.predict(df, offset=np.log(df["Exposure"])) / df["Exposure"], 0, None
    )
df["Pred_Claim_Count"] = df["Pred_Freq_Annual"] * df["Exposure"]

# Severity models: Gamma vs Lognormal vs Gamma-Pareto
sev_full = df[df["ClaimAmount"] > 0].copy()
sev_train = sev_full.loc[sev_full.index.intersection(train_idx)].copy()
sev_test = sev_full.loc[sev_full.index.intersection(test_idx)].copy()

sev_gamma = glm(
    formula="ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=sev_train,
    family=Gamma()
).fit()
sev_gamma_pred = np.clip(sev_gamma.predict(sev_test), 0, None)

sev_lognorm = ols(
    formula="np.log(ClaimAmount) ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=sev_train
).fit()
log_sigma2 = float(sev_lognorm.mse_resid)
sev_lognorm_pred = np.clip(
    np.exp(sev_lognorm.predict(sev_test) + 0.5 * log_sigma2), 0, None
)

tail_threshold = float(sev_train["ClaimAmount"].quantile(0.90))
tail_train = sev_train.loc[sev_train["ClaimAmount"] >= tail_threshold, "ClaimAmount"].values
if len(tail_train) >= 30 and tail_threshold > 0:
    alpha = float(len(tail_train) / np.sum(np.log(tail_train / tail_threshold)))
    alpha = max(alpha, 1.01)
    pareto_tail_mean = float(alpha * tail_threshold / (alpha - 1))
else:
    pareto_tail_mean = float(np.percentile(sev_train["ClaimAmount"], 95))

# Smooth blend: gamma body + pareto tail
tail_scale = max(0.10 * tail_threshold, 1.0)
tail_weight_test = 1.0 / (1.0 + np.exp(-(sev_gamma_pred - tail_threshold) / tail_scale))
sev_gp_pred = np.clip(
    (1.0 - tail_weight_test) * sev_gamma_pred + tail_weight_test * pareto_tail_mean,
    0,
    None,
)

sev_models = [
    {
        "Model": "Gamma GLM",
        "RMSE": rmse(sev_test["ClaimAmount"], sev_gamma_pred),
        "MAE": float(mean_absolute_error(sev_test["ClaimAmount"], sev_gamma_pred)),
    },
    {
        "Model": "Lognormal (Log-OLS)",
        "RMSE": rmse(sev_test["ClaimAmount"], sev_lognorm_pred),
        "MAE": float(mean_absolute_error(sev_test["ClaimAmount"], sev_lognorm_pred)),
    },
    {
        "Model": "Gamma-Pareto Hybrid",
        "RMSE": rmse(sev_test["ClaimAmount"], sev_gp_pred),
        "MAE": float(mean_absolute_error(sev_test["ClaimAmount"], sev_gp_pred)),
    },
]

sev_results = pd.DataFrame(sev_models).sort_values("RMSE")
best_sev_model_name = sev_results.iloc[0]["Model"]
if best_sev_model_name == "Gamma GLM":
    df["Pred_Sev"] = np.clip(sev_gamma.predict(df), 0, None)
elif best_sev_model_name == "Lognormal (Log-OLS)":
    df["Pred_Sev"] = np.clip(np.exp(sev_lognorm.predict(df) + 0.5 * log_sigma2), 0, None)
else:
    gamma_all = np.clip(sev_gamma.predict(df), 0, None)
    tail_weight_all = 1.0 / (1.0 + np.exp(-(gamma_all - tail_threshold) / tail_scale))
    df["Pred_Sev"] = np.clip(
        (1.0 - tail_weight_all) * gamma_all + tail_weight_all * pareto_tail_mean,
        0,
        None,
    )

# ML benchmark for explainability tab
rf_sev = RandomForestRegressor(
    n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
)
rf_sev.fit(sev_train[features], sev_train["ClaimAmount"])

df["Pure_Premium_Base"] = df["Pred_Claim_Count"] * df["Pred_Sev"]

# Portfolio KPIs
total_policies = len(df)
total_exposure = float(df["Exposure"].sum())
total_claim_count = float(df["ClaimNb"].sum())
total_incurred = float(df["ClaimAmount"].sum())
claim_frequency = total_claim_count / total_exposure if total_exposure > 0 else 0.0
claim_severity = total_incurred / total_claim_count if total_claim_count > 0 else 0.0
pure_premium_empirical = total_incurred / total_exposure if total_exposure > 0 else 0.0
claim_rate = float(df["HasClaim"].mean())
zero_claim_ratio = 1.0 - claim_rate
predicted_pure_premium = float(df["Pure_Premium_Base"].mean())

# Tabs
tabs = st.tabs([
    "Data Overview",
    "Pricing Models",
    "Premium & Adjustments",
    "ML & Explainability",
    "IBNR Reserving",
    "Risk Simulation",
    "Reinsurance",
    "Reporting",
])


with tabs[0]:
    st.header("Data Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Policies", f"{total_policies:,}")
    k2.metric("Total Exposure", f"{total_exposure:,.2f}")
    k3.metric("Total Claims", f"{total_claim_count:,.0f}")
    k4.metric("Total Incurred Loss", f"{total_incurred:,.2f}")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Claim Frequency", f"{claim_frequency:.4f}")
    k6.metric("Claim Severity", f"{claim_severity:,.2f}")
    k7.metric("Pure Premium (Empirical)", f"{pure_premium_empirical:,.2f}")
    k8.metric("Claim Rate", f"{claim_rate:.2%}")

    k9, k10 = st.columns(2)
    k9.metric("Zero-Claim Ratio", f"{zero_claim_ratio:.2%}")
    k10.metric("Predicted Pure Premium", f"{predicted_pure_premium:,.2f}")

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = plt.figure(figsize=(8, 4))
        plt.hist(df["Exposure"], bins=40)
        plt.title("Exposure Distribution")
        st.pyplot(fig)
    with c2:
        fig = plt.figure(figsize=(8, 4))
        plt.hist(df["ClaimNb"], bins=np.arange(df["ClaimNb"].max() + 2) - 0.5)
        plt.title("Claim Count Distribution")
        st.pyplot(fig)

    sev_non_zero = df.loc[df["ClaimAmount"] > 0, "ClaimAmount"]
    d1, d2 = st.columns(2)
    with d1:
        fig = plt.figure(figsize=(8, 4))
        plt.hist(sev_non_zero, bins=60)
        plt.title("Observed Severity Histogram")
        st.pyplot(fig)
    with d2:
        fig = plt.figure(figsize=(8, 4))
        plt.scatter(df["DrivAge"], np.log1p(df["ClaimAmount"]), s=3, alpha=0.3)
        plt.title("Driver Age vs log(1+ClaimAmount)")
        plt.xlabel("Driver Age")
        plt.ylabel("log(1 + ClaimAmount)")
        st.pyplot(fig)


with tabs[1]:
    st.header("Pricing Models")
    st.subheader("Frequency Model Comparison")
    st.dataframe(freq_results.round(4), use_container_width=True)
    st.success(f"Best Frequency Model: {best_freq_model_name}")

    fig = plt.figure(figsize=(9, 4))
    plt.hist(test_df["Observed_Freq"], bins=60, alpha=0.5, label="Observed")
    plt.hist(freq_poisson_test_annual, bins=60, alpha=0.5, label="Poisson GLM")
    plt.hist(freq_nb_test_annual, bins=60, alpha=0.5, label="Negative Binomial GLM")
    plt.legend()
    plt.title("Frequency Comparison on Test Set")
    st.pyplot(fig)

    st.subheader("Severity Model Comparison")
    st.dataframe(sev_results.round(2), use_container_width=True)
    st.success(f"Best Severity Model: {best_sev_model_name}")

    fig = plt.figure(figsize=(9, 4))
    plt.hist(sev_test["ClaimAmount"], bins=60, alpha=0.5, label="Observed")
    plt.hist(sev_gamma_pred, bins=60, alpha=0.5, label="Gamma GLM")
    plt.hist(sev_lognorm_pred, bins=60, alpha=0.5, label="Lognormal")
    plt.hist(sev_gp_pred, bins=60, alpha=0.5, label="Gamma-Pareto")
    plt.legend()
    plt.title("Severity Comparison on Test Set")
    st.pyplot(fig)


with tabs[2]:
    st.header("Premium & Adjustments")
    deductible = st.slider("Deductible", 0, 5000, 500, 100)
    inflation_rate = st.slider("Annual Inflation Rate", 0.0, 0.2, 0.05, 0.005)
    years = st.slider("Projection Years", 1, 10, 2)
    expense_loading = st.slider("Expense Loading", 0.0, 0.5, 0.15, 0.01)
    profit_loading = st.slider("Profit Loading", 0.0, 0.4, 0.05, 0.01)

    inflation_factor = (1 + inflation_rate) ** years
    df["Adj_Sev"] = np.maximum(df["Pred_Sev"] - deductible, 0) * inflation_factor
    df["Pure_Premium_Adj"] = df["Pred_Claim_Count"] * df["Adj_Sev"]
    df["Technical_Premium"] = df["Pure_Premium_Adj"] * (1 + expense_loading + profit_loading)

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Adjusted Severity", f"{df['Adj_Sev'].mean():,.2f}")
    m2.metric("Avg Pure Premium", f"{df['Pure_Premium_Adj'].mean():,.2f}")
    m3.metric("Avg Technical Premium", f"{df['Technical_Premium'].mean():,.2f}")

    fig = plt.figure(figsize=(9, 4))
    plt.hist(df["Technical_Premium"], bins=60)
    plt.title("Technical Premium Distribution")
    st.pyplot(fig)


with tabs[3]:
    st.header("ML & Explainability")
    st.subheader("Lift Curve (Best Severity Model)")
    if best_sev_model_name == "Gamma GLM":
        sev_test_pred_for_lift = sev_gamma_pred
    elif best_sev_model_name == "Lognormal (Log-OLS)":
        sev_test_pred_for_lift = sev_lognorm_pred
    else:
        sev_test_pred_for_lift = sev_gp_pred

    lift_df = build_lift_table(
        actual=sev_test["ClaimAmount"].reset_index(drop=True),
        predicted=pd.Series(sev_test_pred_for_lift).reset_index(drop=True),
    )

    fig = plt.figure(figsize=(9, 4))
    plt.plot(lift_df["cum_book_pct"], lift_df["cum_actual_pct"], label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("Lift Curve")
    plt.xlabel("Cumulative % of Portfolio")
    plt.ylabel("Cumulative % of Claims")
    plt.legend()
    st.pyplot(fig)

    st.subheader("Feature Importance")
    feat_imp = pd.Series(rf_sev.feature_importances_, index=features).sort_values(ascending=False)
    fig = plt.figure(figsize=(8, 4))
    feat_imp.plot(kind="bar")
    plt.title("Random Forest Severity Feature Importance")
    st.pyplot(fig)

    st.subheader("SHAP Explainability")
    if SHAP_AVAILABLE:
        sample = sev_test[features].head(300)
        explainer = shap.Explainer(rf_sev)
        shap_values = explainer(sample)
        fig = plt.figure(figsize=(8, 4))
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)
    else:
        st.info("SHAP not installed in this environment. Feature importance is displayed instead.")


with tabs[4]:
    st.header("IBNR Reserving")
    st.write("Edit the cumulative triangle and run chain-ladder IBNR.")

    default_triangle = pd.DataFrame(
        {
            "Dev_12": [120000, 140000, 160000, 180000],
            "Dev_24": [170000, 200000, 225000, np.nan],
            "Dev_36": [210000, 245000, np.nan, np.nan],
            "Dev_48": [235000, np.nan, np.nan, np.nan],
        },
        index=["AY_2021", "AY_2022", "AY_2023", "AY_2024"],
    )

    tri_input = st.data_editor(default_triangle, num_rows="fixed", use_container_width=True)
    tri_input = tri_input.apply(pd.to_numeric, errors="coerce")

    dev_factors, completed_tri, ibnr_summary = chain_ladder_ibnr(tri_input)
    st.write("Development Factors", [round(x, 4) for x in dev_factors])
    st.subheader("Completed Triangle")
    st.dataframe(completed_tri.round(2), use_container_width=True)
    st.subheader("IBNR Summary")
    st.dataframe(ibnr_summary.round(2), use_container_width=True)
    st.metric("Total IBNR Reserve", f"{ibnr_summary['IBNR'].sum():,.2f}")


with tabs[5]:
    st.header("Risk Simulation")
    n_sim = st.slider("Number of Simulations", 2000, 50000, 10000, 1000)
    seed = st.number_input("Random Seed", min_value=1, max_value=99999, value=42)
    rng = np.random.default_rng(seed)

    lambda_claims = max(df["Pred_Claim_Count"].mean(), 1e-6)
    sev_mean = max(df["Pred_Sev"].mean(), 1e-6)
    sev_var = max(df["Pred_Sev"].var(), sev_mean)
    shape = max((sev_mean ** 2) / sev_var, 1e-3)
    scale = max(sev_var / sev_mean, 1e-3)

    freq_sim = rng.poisson(lam=lambda_claims, size=n_sim)
    sev_sim = rng.gamma(shape=shape, scale=scale, size=n_sim)
    agg_loss = freq_sim * sev_sim

    var95 = float(np.percentile(agg_loss, 95))
    var99 = float(np.percentile(agg_loss, 99))
    tvar95 = float(agg_loss[agg_loss >= var95].mean())
    tvar99 = float(agg_loss[agg_loss >= var99].mean())

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("VaR 95%", f"{var95:,.2f}")
    r2.metric("Tail VaR 95%", f"{tvar95:,.2f}")
    r3.metric("VaR 99%", f"{var99:,.2f}")
    r4.metric("Tail VaR 99%", f"{tvar99:,.2f}")

    fig = plt.figure(figsize=(9, 4))
    plt.hist(agg_loss, bins=80)
    plt.axvline(var95, color="orange", linestyle="--", label="VaR95")
    plt.axvline(var99, color="red", linestyle="--", label="VaR99")
    plt.title("Simulated Aggregate Loss Distribution")
    plt.legend()
    st.pyplot(fig)


with tabs[6]:
    st.header("Reinsurance")
    st.write("Apply quota-share and excess-of-loss reinsurance to simulated losses.")

    if "agg_loss" not in locals():
        st.info("Run the Risk Simulation tab first.")
    else:
        quota = st.slider("Quota Share %", 0.0, 1.0, 0.30, 0.01)
        retention = st.slider("Excess of Loss Retention", 1000, 50000, 10000, 500)

        gross_total = float(agg_loss.sum())
        quota_ceded = gross_total * quota
        quota_net = gross_total - quota_ceded

        xol_ceded = float(np.maximum(agg_loss - retention, 0).sum())
        xol_net = gross_total - xol_ceded

        c1, c2, c3 = st.columns(3)
        c1.metric("Gross Simulated Loss", f"{gross_total:,.2f}")
        c2.metric("Quota Ceded", f"{quota_ceded:,.2f}")
        c3.metric("Quota Net", f"{quota_net:,.2f}")

        c4, c5 = st.columns(2)
        c4.metric("XoL Ceded", f"{xol_ceded:,.2f}")
        c5.metric("XoL Net", f"{xol_net:,.2f}")


with tabs[7]:
    st.header("Reporting")
    st.write("Generate a compact actuarial report with key portfolio and risk metrics.")

    if st.button("Generate Actuarial PDF Report"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Actuarial Insurance Report", styles["Heading1"]))
        elements.append(Spacer(1, 0.2 * inch))

        report_metrics = [
            ["Metric", "Value"],
            ["Policies", f"{total_policies:,}"],
            ["Total Exposure", f"{total_exposure:,.2f}"],
            ["Claim Frequency", f"{claim_frequency:.4f}"],
            ["Claim Severity", f"{claim_severity:,.2f}"],
            ["Empirical Pure Premium", f"{pure_premium_empirical:,.2f}"],
            ["Predicted Pure Premium", f"{predicted_pure_premium:,.2f}"],
            ["Best Frequency Model", best_freq_model_name],
            ["Best Severity Model", best_sev_model_name],
        ]

        if "var95" in locals():
            report_metrics.extend(
                [
                    ["VaR 95%", f"{var95:,.2f}"],
                    ["Tail VaR 95%", f"{tvar95:,.2f}"],
                    ["VaR 99%", f"{var99:,.2f}"],
                    ["Tail VaR 99%", f"{tvar99:,.2f}"],
                ]
            )

        if "ibnr_summary" in locals():
            report_metrics.append(["Total IBNR Reserve", f"{ibnr_summary['IBNR'].sum():,.2f}"])

        table = Table(report_metrics)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )

        elements.append(table)
        doc.build(elements)
        buffer.seek(0)

        st.download_button(
            label="Download Report",
            data=buffer,
            file_name="Actuarial_Report.pdf",
            mime="application/pdf",
        )
        st.success("Report generated.")
