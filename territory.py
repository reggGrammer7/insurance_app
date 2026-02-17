import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gamma as gamma_dist
from scipy.stats import genpareto
from sklearn.datasets import fetch_openml
from statsmodels.api import add_constant
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedPoisson,
)
from statsmodels.formula.api import glm, ols
from statsmodels.genmod.families import Binomial, Gamma, InverseGaussian, NegativeBinomial, Poisson

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("Territorial Frequency and Severity Dashboard")


@st.cache_data
def load_data() -> pd.DataFrame:
    freq = fetch_openml(name="freMTPL2freq", version=1, as_frame=True)
    sev = fetch_openml(name="freMTPL2sev", version=1, as_frame=True)

    df = freq.frame.merge(sev.frame, how="left", on="IDpol")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)

    numeric_cols = [
        "ClaimNb",
        "Exposure",
        "ClaimAmount",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "Density",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)].copy()
    df = df.dropna().copy()

    df["log_density"] = np.log(np.clip(df["Density"], 1e-9, None))
    df["log_exposure"] = np.log(np.clip(df["Exposure"], 1e-9, None))
    return df


def metric_name_and_direction(model_rows: list[dict], preferred: str = "AIC") -> tuple[str, bool]:
    if preferred == "AIC" and all(np.isfinite(x.get("aic", np.nan)) for x in model_rows):
        return "AIC", True
    if all(np.isfinite(x.get("llf", np.nan)) for x in model_rows):
        return "LogLik", False
    return "AIC", True


def fit_frequency_models(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, str, str]:
    formula = "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + log_density"
    offset = df["log_exposure"]

    pois = glm(formula=formula, data=df, family=Poisson(), offset=offset).fit()
    nb = glm(formula=formula, data=df, family=NegativeBinomial(), offset=offset).fit()

    x = pd.get_dummies(
        df[["VehPower", "VehAge", "DrivAge", "BonusMalus", "log_density", "Region"]],
        columns=["Region"],
        drop_first=True,
    )
    x = add_constant(x, has_constant="add")
    y = df["ClaimNb"].astype(int)

    zip_model = ZeroInflatedPoisson(
        endog=y,
        exog=x,
        exog_infl=x,
        offset=offset,
        inflation="logit",
    ).fit(method="bfgs", maxiter=200, disp=0)

    zinb_model = ZeroInflatedNegativeBinomialP(
        endog=y,
        exog=x,
        exog_infl=x,
        offset=offset,
        inflation="logit",
    ).fit(method="bfgs", maxiter=250, disp=0)

    candidates = [
        {"model": "Poisson", "fit": pois, "aic": pois.aic, "llf": pois.llf},
        {"model": "Negative Binomial", "fit": nb, "aic": nb.aic, "llf": nb.llf},
        {"model": "ZIP", "fit": zip_model, "aic": zip_model.aic, "llf": zip_model.llf},
        {"model": "ZINB", "fit": zinb_model, "aic": zinb_model.aic, "llf": zinb_model.llf},
    ]

    metric, lower_is_better = metric_name_and_direction(candidates, preferred="AIC")
    metric_key = "aic" if metric == "AIC" else "llf"
    best = (
        min(candidates, key=lambda r: r[metric_key])
        if lower_is_better
        else max(candidates, key=lambda r: r[metric_key])
    )

    freq_scores = pd.DataFrame(
        {
            "Model": [c["model"] for c in candidates],
            "AIC": [c["aic"] for c in candidates],
            "LogLik": [c["llf"] for c in candidates],
        }
    ).sort_values(by=metric, ascending=lower_is_better)

    df["pred_freq_poisson"] = pois.predict(df, offset=offset)
    df["pred_freq_negbin"] = nb.predict(df, offset=offset)
    df["pred_freq_zip"] = zip_model.predict(exog=x, exog_infl=x, offset=offset, which="mean")
    df["pred_freq_zinb"] = zinb_model.predict(exog=x, exog_infl=x, offset=offset, which="mean")

    best_map = {
        "Poisson": "pred_freq_poisson",
        "Negative Binomial": "pred_freq_negbin",
        "ZIP": "pred_freq_zip",
        "ZINB": "pred_freq_zinb",
    }
    df["pred_freq_best"] = df[best_map[best["model"]]]
    return best, freq_scores, metric, metric_key


def fit_severity_models(df: pd.DataFrame, tail_quantile: float) -> tuple[dict, pd.DataFrame, str, str, dict]:
    sev_df = df[df["ClaimAmount"] > 0].copy()
    sev_formula = "ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + log_density"

    gamma_fit = glm(sev_formula, data=sev_df, family=Gamma()).fit()
    ig_fit = glm(sev_formula, data=sev_df, family=InverseGaussian()).fit()
    lognorm_fit = ols("np.log(ClaimAmount) ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + log_density", data=sev_df).fit()

    threshold = sev_df["ClaimAmount"].quantile(tail_quantile)
    sev_df["is_tail"] = (sev_df["ClaimAmount"] > threshold).astype(int)

    body_df = sev_df[sev_df["ClaimAmount"] <= threshold].copy()
    body_fit = glm(sev_formula, data=body_df, family=Gamma()).fit()
    tail_prob_fit = glm(
        "is_tail ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + log_density",
        data=sev_df,
        family=Binomial(),
    ).fit()

    excess = sev_df.loc[sev_df["is_tail"] == 1, "ClaimAmount"] - threshold
    if len(excess) > 10 and float(excess.mean()) > 0:
        gpd_shape, _, gpd_scale = genpareto.fit(excess, floc=0)
        if gpd_shape >= 1:
            gpd_shape = 0.95
    else:
        gpd_shape, gpd_scale = 0.2, max(float(excess.mean()), 1.0)

    phi = float(body_fit.scale)
    shape_body = 1.0 / max(phi, 1e-9)
    mu_body = np.clip(body_fit.predict(sev_df), 1e-9, None)
    scale_body = np.clip(mu_body * phi, 1e-9, None)
    p_tail = np.clip(tail_prob_fit.predict(sev_df), 1e-8, 1 - 1e-8)
    y = np.clip(sev_df["ClaimAmount"].to_numpy(), 1e-9, None)

    logpdf_body = gamma_dist.logpdf(y, a=shape_body, scale=scale_body)
    excess_y = np.clip(y - threshold, 0, None)
    logpdf_tail = genpareto.logpdf(excess_y, c=gpd_shape, loc=0, scale=max(gpd_scale, 1e-9))

    is_tail = y > threshold
    comp_ll = np.where(
        is_tail,
        np.log(p_tail) + logpdf_tail,
        np.log(1 - p_tail) + logpdf_body,
    )
    gp_llf = float(np.sum(comp_ll))

    if gpd_shape < 1:
        expected_excess = gpd_scale / (1 - gpd_shape)
    else:
        expected_excess = float(excess.mean()) if len(excess) > 0 else gpd_scale
    expected_tail_mean = threshold + expected_excess

    candidates = [
        {"model": "Gamma GLM", "fit": gamma_fit, "aic": gamma_fit.aic, "llf": gamma_fit.llf},
        {"model": "Inverse Gaussian GLM", "fit": ig_fit, "aic": ig_fit.aic, "llf": ig_fit.llf},
        {"model": "Lognormal OLS", "fit": lognorm_fit, "aic": lognorm_fit.aic, "llf": lognorm_fit.llf},
        {"model": "Gamma + Pareto Tail", "fit": None, "aic": np.nan, "llf": gp_llf},
    ]

    metric, lower_is_better = metric_name_and_direction(candidates, preferred="AIC")
    metric_key = "aic" if metric == "AIC" else "llf"
    best = (
        min(candidates, key=lambda r: r[metric_key])
        if lower_is_better
        else max(candidates, key=lambda r: r[metric_key])
    )

    sev_scores = pd.DataFrame(
        {
            "Model": [c["model"] for c in candidates],
            "AIC": [c["aic"] for c in candidates],
            "LogLik": [c["llf"] for c in candidates],
        }
    ).sort_values(by=metric, ascending=lower_is_better)

    df["pred_sev_gamma"] = np.clip(gamma_fit.predict(df), 1e-9, None)
    df["pred_sev_invgauss"] = np.clip(ig_fit.predict(df), 1e-9, None)
    df["pred_sev_lognorm"] = np.exp(lognorm_fit.predict(df) + 0.5 * lognorm_fit.mse_resid)
    full_p_tail = np.clip(tail_prob_fit.predict(df), 1e-8, 1 - 1e-8)
    df["pred_sev_gamma_pareto"] = np.clip(
        (1 - full_p_tail) * np.clip(body_fit.predict(df), 1e-9, None) + full_p_tail * expected_tail_mean,
        1e-9,
        None,
    )

    best_map = {
        "Gamma GLM": "pred_sev_gamma",
        "Inverse Gaussian GLM": "pred_sev_invgauss",
        "Lognormal OLS": "pred_sev_lognorm",
        "Gamma + Pareto Tail": "pred_sev_gamma_pareto",
    }
    df["pred_sev_best"] = df[best_map[best["model"]]]

    tail_info = {
        "threshold": threshold,
        "shape": gpd_shape,
        "scale": gpd_scale,
        "tail_count": int((sev_df["ClaimAmount"] > threshold).sum()),
    }
    return best, sev_scores, metric, metric_key, tail_info


df = load_data()
st.success(f"Dataset loaded. Records: {len(df):,}")

st.sidebar.header("Tail Settings")
tail_quantile = st.sidebar.slider("Tail threshold quantile", 0.85, 0.99, 0.95, 0.01)

with st.spinner("Fitting frequency models (Poisson, NB, ZIP, ZINB)..."):
    best_freq, freq_scores, freq_metric, _ = fit_frequency_models(df)

with st.spinner("Fitting severity models (Gamma, InvGaussian, Lognormal, Gamma+Pareto)..."):
    best_sev, sev_scores, sev_metric, _, tail_info = fit_severity_models(df, tail_quantile)

df["pure_premium_best"] = df["pred_freq_best"] * df["pred_sev_best"]

st.header("1. Frequency Model Dashboard")
col1, col2 = st.columns(2)
with col1:
    st.write(f"Common metric used: **{freq_metric}**")
    st.write(f"Selected frequency model: **{best_freq['model']}**")
with col2:
    chosen_value = best_freq["aic"] if freq_metric == "AIC" else best_freq["llf"]
    st.metric(f"Best {freq_metric}", f"{chosen_value:,.2f}")

st.dataframe(freq_scores, use_container_width=True)

st.header("2. Severity Model Dashboard")
col3, col4 = st.columns(2)
with col3:
    st.write(f"Common metric used: **{sev_metric}**")
    st.write(f"Selected severity model: **{best_sev['model']}**")
with col4:
    chosen_value = best_sev["aic"] if sev_metric == "AIC" else best_sev["llf"]
    if np.isfinite(chosen_value):
        st.metric(f"Best {sev_metric}", f"{chosen_value:,.2f}")
    else:
        st.metric(f"Best {sev_metric}", "N/A")

st.dataframe(sev_scores, use_container_width=True)

st.caption(
    "Tail diagnostics (Gamma+Pareto): "
    f"threshold={tail_info['threshold']:.2f}, "
    f"tail claims={tail_info['tail_count']}, "
    f"shape={tail_info['shape']:.3f}, scale={tail_info['scale']:.2f}"
)

st.header("3. Pure Premium (Best Frequency x Best Severity)")
col5, col6, col7 = st.columns(3)
with col5:
    st.metric("Avg Pred Frequency", f"{df['pred_freq_best'].mean():.4f}")
with col6:
    st.metric("Avg Pred Severity", f"{df['pred_sev_best'].mean():.2f}")
with col7:
    st.metric("Avg Pure Premium", f"{df['pure_premium_best'].mean():.2f}")

region_summary = (
    df.groupby("Region", as_index=False)["pure_premium_best"]
    .mean()
    .sort_values("pure_premium_best", ascending=False)
)

st.subheader("Average Pure Premium by Region")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(region_summary["Region"], region_summary["pure_premium_best"], color="steelblue")
ax.set_ylabel("Average Pure Premium")
ax.set_xlabel("Region")
ax.set_title("Territorial Pure Premium - Best Models")
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)

st.subheader("Sample Portfolio Predictions")
show_cols = [
    "ClaimNb",
    "ClaimAmount",
    "pred_freq_best",
    "pred_sev_best",
    "pure_premium_best",
]
st.dataframe(df[show_cols].head(20), use_container_width=True)
