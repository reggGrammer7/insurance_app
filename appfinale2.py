import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import glm, ols
from statsmodels.genmod.families import Gamma, NegativeBinomial, Poisson
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


st.set_page_config(layout="wide")
st.title("Actuarial Insurance Modeling Platform - Simple Final")


@st.cache_data
def load_data():
    freq = fetch_openml(name="freMTPL2freq", version=1, as_frame=True)
    sev = fetch_openml(name="freMTPL2sev", version=1, as_frame=True)
    df = freq.frame.merge(sev.frame, how="left", on="IDpol")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    numeric_cols = ["ClaimNb", "Exposure", "ClaimAmount", "VehPower", "VehAge", "DrivAge", "BonusMalus"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)].dropna().copy()
    df["Observed_Freq"] = df["ClaimNb"] / df["Exposure"]
    return df


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calc_chain_ladder(triangle):
    dev_factors = []
    n_cols = triangle.shape[1]
    for j in range(n_cols - 1):
        num = triangle.iloc[:, j + 1].dropna().sum()
        den = triangle.iloc[:, j].dropna().sum()
        dev_factors.append((num / den) if den > 0 else 1.0)

    completed = triangle.copy()
    for i in range(completed.shape[0]):
        row = completed.iloc[i, :]
        last_idx = row.last_valid_index()
        if last_idx is None:
            continue
        last_pos = completed.columns.get_loc(last_idx)
        for j in range(last_pos + 1, n_cols):
            completed.iloc[i, j] = completed.iloc[i, j - 1] * dev_factors[j - 1]

    latest = []
    ultimate = []
    ibnr = []
    for i in range(completed.shape[0]):
        observed = triangle.iloc[i, :].dropna()
        latest_val = float(observed.iloc[-1]) if len(observed) > 0 else 0.0
        ult_val = float(completed.iloc[i, -1]) if pd.notna(completed.iloc[i, -1]) else latest_val
        latest.append(latest_val)
        ultimate.append(ult_val)
        ibnr.append(ult_val - latest_val)

    summary = pd.DataFrame({"Latest": latest, "Ultimate": ultimate, "IBNR": ibnr}, index=triangle.index)
    return dev_factors, completed, summary


def lift_table(actual, predicted):
    x = pd.DataFrame({"actual": actual, "pred": predicted}).sort_values("pred", ascending=False)
    x["cum_actual"] = x["actual"].cumsum()
    total = max(x["actual"].sum(), 1e-9)
    x["cum_actual_pct"] = x["cum_actual"] / total
    x["cum_book_pct"] = np.arange(1, len(x) + 1) / len(x)
    return x


# ---------------------------------------------------------
# Data and model training (run once, reused by all tabs)
# ---------------------------------------------------------
df = load_data()
features = ["VehPower", "VehAge", "DrivAge", "BonusMalus"]

train_idx, test_idx = train_test_split(df.index, test_size=0.30, random_state=42)
train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()

# Frequency: Poisson vs Negative Binomial
freq_poisson = glm(
    "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=train_df,
    family=Poisson(),
    offset=np.log(train_df["Exposure"]),
).fit()

freq_nb = glm(
    "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=train_df,
    family=NegativeBinomial(),
    offset=np.log(train_df["Exposure"]),
).fit()

freq_poisson_test = np.clip(
    freq_poisson.predict(test_df, offset=np.log(test_df["Exposure"])) / test_df["Exposure"], 0, None
)
freq_nb_test = np.clip(
    freq_nb.predict(test_df, offset=np.log(test_df["Exposure"])) / test_df["Exposure"], 0, None
)

freq_results = pd.DataFrame(
    [
        {
            "Model": "Poisson GLM",
            "RMSE": rmse(test_df["Observed_Freq"], freq_poisson_test),
            "MAE": float(mean_absolute_error(test_df["Observed_Freq"], freq_poisson_test)),
        },
        {
            "Model": "Negative Binomial GLM",
            "RMSE": rmse(test_df["Observed_Freq"], freq_nb_test),
            "MAE": float(mean_absolute_error(test_df["Observed_Freq"], freq_nb_test)),
        },
    ]
).sort_values("RMSE")

best_freq_model = freq_results.iloc[0]["Model"]
if best_freq_model == "Poisson GLM":
    df["Pred_Freq"] = np.clip(freq_poisson.predict(df, offset=np.log(df["Exposure"])) / df["Exposure"], 0, None)
else:
    df["Pred_Freq"] = np.clip(freq_nb.predict(df, offset=np.log(df["Exposure"])) / df["Exposure"], 0, None)
df["Pred_Claim_Count"] = df["Pred_Freq"] * df["Exposure"]

# Severity: Gamma vs Lognormal vs Gamma-Pareto
sev_full = df[df["ClaimAmount"] > 0].copy()
sev_train = sev_full.loc[sev_full.index.intersection(train_idx)].copy()
sev_test = sev_full.loc[sev_full.index.intersection(test_idx)].copy()

sev_gamma = glm(
    "ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=sev_train,
    family=Gamma(),
).fit()
sev_gamma_test = np.clip(sev_gamma.predict(sev_test), 0, None)

sev_lognorm = ols(
    "np.log(ClaimAmount) ~ VehPower + VehAge + DrivAge + BonusMalus",
    data=sev_train,
).fit()
sigma2 = float(sev_lognorm.mse_resid)
sev_lognorm_test = np.clip(np.exp(sev_lognorm.predict(sev_test) + 0.5 * sigma2), 0, None)

u = float(sev_train["ClaimAmount"].quantile(0.90))
tail = sev_train.loc[sev_train["ClaimAmount"] >= u, "ClaimAmount"].values
if len(tail) >= 30 and u > 0:
    alpha = float(len(tail) / np.sum(np.log(tail / u)))
    alpha = max(alpha, 1.01)
    pareto_mean = float(alpha * u / (alpha - 1))
else:
    pareto_mean = float(np.percentile(sev_train["ClaimAmount"], 95))

blend_scale = max(0.10 * u, 1.0)
tail_w_test = 1.0 / (1.0 + np.exp(-(sev_gamma_test - u) / blend_scale))
sev_gp_test = np.clip((1.0 - tail_w_test) * sev_gamma_test + tail_w_test * pareto_mean, 0, None)

sev_results = pd.DataFrame(
    [
        {
            "Model": "Gamma GLM",
            "RMSE": rmse(sev_test["ClaimAmount"], sev_gamma_test),
            "MAE": float(mean_absolute_error(sev_test["ClaimAmount"], sev_gamma_test)),
        },
        {
            "Model": "Lognormal (Log-OLS)",
            "RMSE": rmse(sev_test["ClaimAmount"], sev_lognorm_test),
            "MAE": float(mean_absolute_error(sev_test["ClaimAmount"], sev_lognorm_test)),
        },
        {
            "Model": "Gamma-Pareto Hybrid",
            "RMSE": rmse(sev_test["ClaimAmount"], sev_gp_test),
            "MAE": float(mean_absolute_error(sev_test["ClaimAmount"], sev_gp_test)),
        },
    ]
).sort_values("RMSE")

best_sev_model = sev_results.iloc[0]["Model"]
if best_sev_model == "Gamma GLM":
    df["Pred_Sev"] = np.clip(sev_gamma.predict(df), 0, None)
elif best_sev_model == "Lognormal (Log-OLS)":
    df["Pred_Sev"] = np.clip(np.exp(sev_lognorm.predict(df) + 0.5 * sigma2), 0, None)
else:
    sev_gamma_all = np.clip(sev_gamma.predict(df), 0, None)
    tail_w_all = 1.0 / (1.0 + np.exp(-(sev_gamma_all - u) / blend_scale))
    df["Pred_Sev"] = np.clip((1.0 - tail_w_all) * sev_gamma_all + tail_w_all * pareto_mean, 0, None)

df["Pure_Premium_Base"] = df["Pred_Claim_Count"] * df["Pred_Sev"]

# KPI values
total_policies = len(df)
total_exposure = float(df["Exposure"].sum())
total_claims = float(df["ClaimNb"].sum())
total_loss = float(df["ClaimAmount"].sum())
freq_kpi = total_claims / total_exposure if total_exposure > 0 else 0.0
sev_kpi = total_loss / total_claims if total_claims > 0 else 0.0
pure_premium_emp = total_loss / total_exposure if total_exposure > 0 else 0.0


tabs = st.tabs(
    [
        "Data Overview",
        "Pricing Models",
        "Premium & Adjustments",
        "ML & Explainability",
        "IBNR Reserving",
        "Risk Simulation",
        "Reinsurance",
        "Reporting",
    ]
)


with tabs[0]:
    st.header("Data Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Policies", f"{total_policies:,}")
    c2.metric("Total Exposure", f"{total_exposure:,.2f}")
    c3.metric("Total Claims", f"{total_claims:,.0f}")
    c4.metric("Total Incurred", f"{total_loss:,.2f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Claim Frequency", f"{freq_kpi:.4f}")
    c6.metric("Claim Severity", f"{sev_kpi:,.2f}")
    c7.metric("Pure Premium (Empirical)", f"{pure_premium_emp:,.2f}")

    st.subheader("Raw Data Sample")
    display_cols = ["IDpol", "ClaimNb", "Exposure", "ClaimAmount", "VehPower", "VehAge", "DrivAge", "BonusMalus"]
    existing_cols = [x for x in display_cols if x in df.columns]
    st.dataframe(df[existing_cols].head(20), use_container_width=True)

    p1, p2 = st.columns(2)
    with p1:
        fig = plt.figure(figsize=(8, 4))
        plt.hist(df["ClaimNb"], bins=np.arange(df["ClaimNb"].max() + 2) - 0.5)
        plt.title("Claim Count Histogram")
        st.pyplot(fig)
    with p2:
        fig = plt.figure(figsize=(8, 4))
        plt.hist(df.loc[df["ClaimAmount"] > 0, "ClaimAmount"], bins=60)
        plt.title("Severity Histogram (Observed)")
        st.pyplot(fig)


with tabs[1]:
    st.header("Pricing Models")

    st.subheader("Frequency: Poisson vs Negative Binomial")
    st.dataframe(freq_results.round(4), use_container_width=True)
    st.success(f"Best Frequency Model: {best_freq_model}")

    fig = plt.figure(figsize=(9, 4))
    plt.hist(test_df["Observed_Freq"], bins=60, alpha=0.5, label="Observed")
    plt.hist(freq_poisson_test, bins=60, alpha=0.5, label="Poisson")
    plt.hist(freq_nb_test, bins=60, alpha=0.5, label="NegBin")
    plt.title("Frequency Model Comparison")
    plt.legend()
    st.pyplot(fig)

    st.subheader("Severity: Gamma vs Lognormal vs Gamma-Pareto")
    st.dataframe(sev_results.round(2), use_container_width=True)
    st.success(f"Best Severity Model: {best_sev_model}")

    fig = plt.figure(figsize=(9, 4))
    plt.hist(sev_test["ClaimAmount"], bins=60, alpha=0.45, label="Observed")
    plt.hist(sev_gamma_test, bins=60, alpha=0.45, label="Gamma")
    plt.hist(sev_lognorm_test, bins=60, alpha=0.45, label="Lognormal")
    plt.hist(sev_gp_test, bins=60, alpha=0.45, label="Gamma-Pareto")
    plt.title("Severity Model Comparison")
    plt.legend()
    st.pyplot(fig)


with tabs[2]:
    st.header("Premium & Adjustments")
    deductible = st.slider("Deductible", 0, 5000, 500, 100)
    inflation = st.slider("Annual Inflation", 0.0, 0.20, 0.05, 0.005)
    years = st.slider("Projection Years", 1, 10, 2)
    expense_load = st.slider("Expense Loading", 0.0, 0.50, 0.15, 0.01)
    profit_load = st.slider("Profit Loading", 0.0, 0.30, 0.05, 0.01)

    infl_factor = (1 + inflation) ** years
    df["Adj_Sev"] = np.maximum(df["Pred_Sev"] - deductible, 0) * infl_factor
    df["Pure_Premium_Adj"] = df["Pred_Claim_Count"] * df["Adj_Sev"]
    df["Technical_Premium"] = df["Pure_Premium_Adj"] * (1 + expense_load + profit_load)

    t1, t2, t3 = st.columns(3)
    t1.metric("Avg Adjusted Severity", f"{df['Adj_Sev'].mean():,.2f}")
    t2.metric("Avg Pure Premium", f"{df['Pure_Premium_Adj'].mean():,.2f}")
    t3.metric("Avg Technical Premium", f"{df['Technical_Premium'].mean():,.2f}")

    fig = plt.figure(figsize=(9, 4))
    plt.hist(df["Technical_Premium"], bins=60)
    plt.title("Technical Premium Distribution")
    st.pyplot(fig)


with tabs[3]:
    st.header("ML & Explainability")
    st.subheader("Lift Curve for Best Severity Model")

    if best_sev_model == "Gamma GLM":
        pred_for_lift = sev_gamma_test
    elif best_sev_model == "Lognormal (Log-OLS)":
        pred_for_lift = sev_lognorm_test
    else:
        pred_for_lift = sev_gp_test

    lift_df = lift_table(sev_test["ClaimAmount"].reset_index(drop=True), pd.Series(pred_for_lift).reset_index(drop=True))

    fig = plt.figure(figsize=(9, 4))
    plt.plot(lift_df["cum_book_pct"], lift_df["cum_actual_pct"], label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("Lift Curve")
    plt.xlabel("Cumulative % of Portfolio")
    plt.ylabel("Cumulative % of Loss")
    plt.legend()
    st.pyplot(fig)

    st.subheader("Simple Explainability")
    st.write("Frequency model coefficients (Poisson):")
    st.dataframe(pd.DataFrame({"Feature": freq_poisson.params.index, "Coef": freq_poisson.params.values}).round(4))
    st.write("Severity model coefficients (Gamma):")
    st.dataframe(pd.DataFrame({"Feature": sev_gamma.params.index, "Coef": sev_gamma.params.values}).round(4))


with tabs[4]:
    st.header("IBNR Reserving")
    default_triangle = pd.DataFrame(
        {
            "Dev12": [120000, 140000, 160000, 180000],
            "Dev24": [170000, 200000, 225000, np.nan],
            "Dev36": [210000, 245000, np.nan, np.nan],
            "Dev48": [235000, np.nan, np.nan, np.nan],
        },
        index=["AY2021", "AY2022", "AY2023", "AY2024"],
    )
    tri = st.data_editor(default_triangle, num_rows="fixed", use_container_width=True)
    tri = tri.apply(pd.to_numeric, errors="coerce")

    dev_factors, completed_triangle, ibnr_summary = calc_chain_ladder(tri)
    st.write("Development Factors:", [round(x, 4) for x in dev_factors])
    st.subheader("Completed Triangle")
    st.dataframe(completed_triangle.round(2), use_container_width=True)
    st.subheader("IBNR Summary")
    st.dataframe(ibnr_summary.round(2), use_container_width=True)
    st.metric("Total IBNR", f"{ibnr_summary['IBNR'].sum():,.2f}")


with tabs[5]:
    st.header("Risk Simulation")
    n_sim = st.slider("Simulations", 2000, 50000, 10000, 1000)
    seed = st.number_input("Random Seed", min_value=1, max_value=99999, value=42)
    rng = np.random.default_rng(seed)

    lam = max(df["Pred_Claim_Count"].mean(), 1e-6)
    sev_mean = max(df["Pred_Sev"].mean(), 1e-6)
    sev_var = max(df["Pred_Sev"].var(), sev_mean)
    shape = max((sev_mean ** 2) / sev_var, 1e-3)
    scale = max(sev_var / sev_mean, 1e-3)

    sim_freq = rng.poisson(lam=lam, size=n_sim)
    sim_sev = rng.gamma(shape=shape, scale=scale, size=n_sim)
    sim_loss = sim_freq * sim_sev

    var95 = float(np.percentile(sim_loss, 95))
    var99 = float(np.percentile(sim_loss, 99))
    tvar95 = float(sim_loss[sim_loss >= var95].mean())
    tvar99 = float(sim_loss[sim_loss >= var99].mean())

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("VaR 95%", f"{var95:,.2f}")
    r2.metric("Tail VaR 95%", f"{tvar95:,.2f}")
    r3.metric("VaR 99%", f"{var99:,.2f}")
    r4.metric("Tail VaR 99%", f"{tvar99:,.2f}")

    fig = plt.figure(figsize=(9, 4))
    plt.hist(sim_loss, bins=80)
    plt.axvline(var95, linestyle="--", color="orange", label="VaR95")
    plt.axvline(var99, linestyle="--", color="red", label="VaR99")
    plt.title("Simulated Aggregate Loss")
    plt.legend()
    st.pyplot(fig)


with tabs[6]:
    st.header("Reinsurance")
    st.write("Applies to the current simulated aggregate loss distribution.")

    if "sim_loss" not in locals():
        st.info("Run Risk Simulation tab first.")
    else:
        quota = st.slider("Quota Share %", 0.0, 1.0, 0.30, 0.01)
        retention = st.slider("XoL Retention", 1000, 50000, 10000, 500)

        gross = float(sim_loss.sum())
        quota_ceded = gross * quota
        quota_net = gross - quota_ceded

        xol_ceded = float(np.maximum(sim_loss - retention, 0).sum())
        xol_net = gross - xol_ceded

        q1, q2, q3 = st.columns(3)
        q1.metric("Gross Loss", f"{gross:,.2f}")
        q2.metric("Quota Ceded", f"{quota_ceded:,.2f}")
        q3.metric("Quota Net", f"{quota_net:,.2f}")

        q4, q5 = st.columns(2)
        q4.metric("XoL Ceded", f"{xol_ceded:,.2f}")
        q5.metric("XoL Net", f"{xol_net:,.2f}")


with tabs[7]:
    st.header("Reporting")
    st.write("Generate PDF summary report.")

    if st.button("Generate Actuarial PDF Report"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Actuarial Report Summary", styles["Heading1"]))
        elements.append(Spacer(1, 0.2 * inch))

        table_data = [
            ["Metric", "Value"],
            ["Policies", f"{total_policies:,}"],
            ["Total Exposure", f"{total_exposure:,.2f}"],
            ["Claim Frequency", f"{freq_kpi:.4f}"],
            ["Claim Severity", f"{sev_kpi:,.2f}"],
            ["Pure Premium (Empirical)", f"{pure_premium_emp:,.2f}"],
            ["Best Frequency Model", best_freq_model],
            ["Best Severity Model", best_sev_model],
        ]

        if "var95" in locals():
            table_data.extend(
                [
                    ["VaR 95%", f"{var95:,.2f}"],
                    ["Tail VaR 95%", f"{tvar95:,.2f}"],
                    ["VaR 99%", f"{var99:,.2f}"],
                    ["Tail VaR 99%", f"{tvar99:,.2f}"],
                ]
            )

        if "ibnr_summary" in locals():
            table_data.append(["Total IBNR", f"{ibnr_summary['IBNR'].sum():,.2f}"])

        table = Table(table_data)
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
            "Download Report",
            data=buffer,
            file_name="Actuarial_Report_Simple.pdf",
            mime="application/pdf",
        )
        st.success("Report generated.")
