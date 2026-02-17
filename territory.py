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
from statsmodels.genmod.families.links import Log

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


def stratified_sample(df: pd.DataFrame, max_rows: int, group_col: str, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    base = (
        df.groupby(group_col, group_keys=False)
        .apply(lambda g: g.sample(n=1, random_state=random_state))
        .reset_index(drop=True)
    )
    need = max_rows - len(base)
    if need <= 0:
        return base.copy()
    remaining = df.drop(index=base.index, errors="ignore")
    extra = remaining.sample(n=min(need, len(remaining)), random_state=random_state)
    return pd.concat([base, extra], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fit_frequency_models(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    max_zi_rows: int,
    include_zinb: bool,
    zi_maxiter: int,
) -> tuple[dict, pd.DataFrame, str, str, pd.DataFrame]:
    formula = "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + log_density"
    train_offset = train_df["log_exposure"].astype(float)
    full_offset = full_df["log_exposure"].astype(float)

    pois = glm(formula=formula, data=train_df, family=Poisson(), offset=train_offset).fit()
    nb = glm(formula=formula, data=train_df, family=NegativeBinomial(), offset=train_offset).fit()

    zi_df = stratified_sample(train_df, max_zi_rows, "Region", random_state=42)
    x_train = pd.get_dummies(
        zi_df[["VehPower", "VehAge", "DrivAge", "BonusMalus", "log_density", "Region"]],
        columns=["Region"],
        drop_first=True,
        dtype=float,
    )
    x_train = add_constant(x_train, has_constant="add").astype(float)
    y_train = zi_df["ClaimNb"].astype(int)
    train_zi_offset = zi_df["log_exposure"].astype(float)
    x_train_np = np.asarray(x_train, dtype=float)
    y_train_np = np.asarray(y_train, dtype=np.int64)
    train_zi_offset_np = np.asarray(train_zi_offset, dtype=float)

    zip_model = ZeroInflatedPoisson(
        endog=y_train_np,
        exog=x_train_np,
        exog_infl=x_train_np,
        offset=train_zi_offset_np,
        inflation="logit",
    ).fit(method="bfgs", maxiter=zi_maxiter, disp=0)

    zinb_model = None
    zinb_aic = np.nan
    zinb_llf = np.nan
    if include_zinb:
        zinb_model = ZeroInflatedNegativeBinomialP(
            endog=y_train_np,
            exog=x_train_np,
            exog_infl=x_train_np,
            offset=train_zi_offset_np,
            inflation="logit",
        ).fit(method="bfgs", maxiter=zi_maxiter, disp=0)
        zinb_aic = zinb_model.aic
        zinb_llf = zinb_model.llf

    x_full = pd.get_dummies(
        full_df[["VehPower", "VehAge", "DrivAge", "BonusMalus", "log_density", "Region"]],
        columns=["Region"],
        drop_first=True,
        dtype=float,
    )
    x_full = add_constant(x_full, has_constant="add").astype(float)
    x_full = x_full.reindex(columns=x_train.columns, fill_value=0.0)
    x_full_np = np.asarray(x_full, dtype=float)
    full_offset_np = np.asarray(full_offset, dtype=float)

    candidates = [
        {"model": "Poisson", "fit": pois, "aic": pois.aic, "llf": pois.llf},
        {"model": "Negative Binomial", "fit": nb, "aic": nb.aic, "llf": nb.llf},
        {"model": "ZIP", "fit": zip_model, "aic": zip_model.aic, "llf": zip_model.llf},
    ]
    if include_zinb:
        candidates.append({"model": "ZINB", "fit": zinb_model, "aic": zinb_aic, "llf": zinb_llf})

    scored = [c for c in candidates if np.isfinite(c["aic"]) and np.isfinite(c["llf"])]
    if not scored:
        scored = [c for c in candidates if np.isfinite(c["llf"])]
    if not scored:
        scored = candidates

    metric, lower_is_better = metric_name_and_direction(scored, preferred="AIC")
    metric_key = "aic" if metric == "AIC" else "llf"
    best = (
        min(scored, key=lambda r: r[metric_key])
        if lower_is_better
        else max(scored, key=lambda r: r[metric_key])
    )

    freq_scores = pd.DataFrame(
        {
            "Model": [c["model"] for c in candidates],
            "AIC": [c["aic"] for c in candidates],
            "LogLik": [c["llf"] for c in candidates],
        }
    ).sort_values(by=metric, ascending=lower_is_better)

    pred_df = full_df.copy()
    pred_df["pred_freq_poisson"] = pois.predict(full_df, offset=full_offset)
    pred_df["pred_freq_negbin"] = nb.predict(full_df, offset=full_offset)
    pred_df["pred_freq_zip"] = zip_model.predict(exog=x_full_np, exog_infl=x_full_np, offset=full_offset_np, which="mean")
    if include_zinb and zinb_model is not None:
        pred_df["pred_freq_zinb"] = zinb_model.predict(exog=x_full_np, exog_infl=x_full_np, offset=full_offset_np, which="mean")
    else:
        pred_df["pred_freq_zinb"] = np.nan

    best_map = {
        "Poisson": "pred_freq_poisson",
        "Negative Binomial": "pred_freq_negbin",
        "ZIP": "pred_freq_zip",
        "ZINB": "pred_freq_zinb",
    }
    pred_df["pred_freq_best"] = pred_df[best_map[best["model"]]]
    return best, freq_scores, metric, metric_key, pred_df


@st.cache_data(show_spinner=False)
def fit_severity_models(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    tail_quantile: float,
) -> tuple[dict, pd.DataFrame, str, str, dict, pd.DataFrame]:
    sev_df = train_df[train_df["ClaimAmount"] > 0].copy()
    sev_formula = "ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + log_density"

    gamma_fit = glm(sev_formula, data=sev_df, family=Gamma()).fit()
    ig_fit = None
    ig_aic = np.nan
    ig_llf = np.nan
    ig_failed = False
    try:
        ig_fit = glm(sev_formula, data=sev_df, family=InverseGaussian(link=Log())).fit()
        ig_aic = ig_fit.aic
        ig_llf = ig_fit.llf
    except Exception:
        ig_failed = True
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
        {"model": "Inverse Gaussian GLM", "fit": ig_fit, "aic": ig_aic, "llf": ig_llf},
        {"model": "Lognormal OLS", "fit": lognorm_fit, "aic": lognorm_fit.aic, "llf": lognorm_fit.llf},
        {"model": "Gamma + Pareto Tail", "fit": None, "aic": np.nan, "llf": gp_llf},
    ]

    scored = [c for c in candidates if np.isfinite(c["aic"]) and np.isfinite(c["llf"])]
    if not scored:
        scored = [c for c in candidates if np.isfinite(c["llf"])]
    metric, lower_is_better = metric_name_and_direction(scored, preferred="AIC")
    metric_key = "aic" if metric == "AIC" else "llf"
    best = (
        min(scored, key=lambda r: r[metric_key])
        if lower_is_better
        else max(scored, key=lambda r: r[metric_key])
    )

    sev_scores = pd.DataFrame(
        {
            "Model": [c["model"] for c in candidates],
            "AIC": [c["aic"] for c in candidates],
            "LogLik": [c["llf"] for c in candidates],
        }
    ).sort_values(by=metric, ascending=lower_is_better)

    pred_df = full_df.copy()
    pred_df["pred_sev_gamma"] = np.clip(gamma_fit.predict(full_df), 1e-9, None)
    if ig_fit is not None:
        pred_df["pred_sev_invgauss"] = np.clip(ig_fit.predict(full_df), 1e-9, None)
    else:
        pred_df["pred_sev_invgauss"] = np.nan
    pred_df["pred_sev_lognorm"] = np.exp(lognorm_fit.predict(full_df) + 0.5 * lognorm_fit.mse_resid)
    full_p_tail = np.clip(tail_prob_fit.predict(full_df), 1e-8, 1 - 1e-8)
    pred_df["pred_sev_gamma_pareto"] = np.clip(
        (1 - full_p_tail) * np.clip(body_fit.predict(full_df), 1e-9, None) + full_p_tail * expected_tail_mean,
        1e-9,
        None,
    )

    best_map = {
        "Gamma GLM": "pred_sev_gamma",
        "Inverse Gaussian GLM": "pred_sev_invgauss",
        "Lognormal OLS": "pred_sev_lognorm",
        "Gamma + Pareto Tail": "pred_sev_gamma_pareto",
    }
    pred_df["pred_sev_best"] = pred_df[best_map[best["model"]]]

    tail_info = {
        "threshold": threshold,
        "shape": gpd_shape,
        "scale": gpd_scale,
        "tail_count": int((sev_df["ClaimAmount"] > threshold).sum()),
        "ig_failed": ig_failed,
    }
    return best, sev_scores, metric, metric_key, tail_info, pred_df


df = load_data()
st.success(f"Dataset loaded. Records: {len(df):,}")

st.sidebar.header("Performance Settings")
fast_mode = st.sidebar.checkbox("Fast mode (recommended)", value=True)
max_train_rows = st.sidebar.slider("Max training rows", 30000, 300000, 120000, 10000)
max_zi_rows = st.sidebar.slider("Max ZIP/ZINB rows", 10000, 120000, 35000, 5000)
include_zinb = st.sidebar.checkbox("Fit ZINB", value=not fast_mode)
zi_maxiter = st.sidebar.slider("ZIP/ZINB max iterations", 30, 200, 60 if fast_mode else 120, 10)

st.sidebar.header("Tail Settings")
tail_quantile = st.sidebar.slider("Tail threshold quantile", 0.85, 0.99, 0.95, 0.01)

train_df = stratified_sample(df, max_train_rows if fast_mode else len(df), "Region", random_state=42)
st.caption(
    f"Training sample: {len(train_df):,} rows (full data: {len(df):,}). "
    f"ZIP/ZINB capped at {min(max_zi_rows, len(train_df)):,} rows."
)

with st.spinner("Fitting frequency models (Poisson, NB, ZIP, ZINB)..."):
    best_freq, freq_scores, freq_metric, _, freq_pred_df = fit_frequency_models(
        train_df=train_df,
        full_df=df,
        max_zi_rows=max_zi_rows if fast_mode else len(train_df),
        include_zinb=include_zinb,
        zi_maxiter=zi_maxiter,
    )

with st.spinner("Fitting severity models (Gamma, InvGaussian, Lognormal, Gamma+Pareto)..."):
    best_sev, sev_scores, sev_metric, _, tail_info, sev_pred_df = fit_severity_models(train_df, df, tail_quantile)

df = df.copy()
df["pred_freq_best"] = freq_pred_df["pred_freq_best"].to_numpy()
df["pred_sev_best"] = sev_pred_df["pred_sev_best"].to_numpy()
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

if tail_info.get("ig_failed", False):
    st.warning("Inverse Gaussian severity fit was numerically unstable and excluded from model selection.")

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
