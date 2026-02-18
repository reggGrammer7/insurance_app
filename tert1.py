import warnings

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gamma as gamma_dist
from scipy.stats import genpareto
from sklearn.datasets import fetch_openml
from statsmodels.api import add_constant
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial, Gamma, NegativeBinomial, Poisson

warnings.filterwarnings("ignore") 

st.set_page_config(layout="wide")
st.title("Frequency (ZIP/Poisson/NB) + Severity (Gamma/Gamma-Pareto)")


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
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1)].dropna().copy()
    df["log_density"] = np.log(np.clip(df["Density"], 1e-9, None))
    df["log_exposure"] = np.log(np.clip(df["Exposure"], 1e-9, None))
    return df


def choose_metric(rows: list[dict]) -> tuple[str, str, bool]:
    if all(np.isfinite(r.get("aic", np.nan)) for r in rows):
        return "AIC", "aic", True
    return "LogLik", "llf", False


@st.cache_data(show_spinner=False)
def fit_frequency(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, str, pd.DataFrame]:
    formula = "ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + log_density"
    offset = df["log_exposure"].astype(float)

    pois = glm(formula=formula, data=df, family=Poisson(), offset=offset).fit()
    nb = glm(formula=formula, data=df, family=NegativeBinomial(), offset=offset).fit()

    x = pd.get_dummies(
        df[["VehPower", "VehAge", "DrivAge", "BonusMalus", "log_density", "Region"]],
        columns=["Region"],
        drop_first=True,
        dtype=float,
    )
    x = add_constant(x, has_constant="add").astype(float)
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(df["ClaimNb"].astype(int), dtype=np.int64)
    offset_np = np.asarray(offset, dtype=float)

    zip_fit = ZeroInflatedPoisson(
        endog=y_np,
        exog=x_np,
        exog_infl=x_np,
        offset=offset_np,
        inflation="logit",
    ).fit(method="bfgs", maxiter=120, disp=0)

    candidates = [
        {"model": "Poisson", "aic": pois.aic, "llf": pois.llf},
        {"model": "Negative Binomial", "aic": nb.aic, "llf": nb.llf},
        {"model": "ZIP", "aic": zip_fit.aic, "llf": zip_fit.llf},
    ]
    metric_name, metric_key, lower_is_better = choose_metric(candidates)
    best = (
        min(candidates, key=lambda r: r[metric_key])
        if lower_is_better
        else max(candidates, key=lambda r: r[metric_key])
    )

    out = df.copy()
    out["pred_freq_poisson"] = pois.predict(df, offset=offset)
    out["pred_freq_nb"] = nb.predict(df, offset=offset)
    out["pred_freq_zip"] = zip_fit.predict(exog=x_np, exog_infl=x_np, offset=offset_np, which="mean")
    best_map = {
        "Poisson": "pred_freq_poisson",
        "Negative Binomial": "pred_freq_nb",
        "ZIP": "pred_freq_zip",
    }
    out["pred_freq_best"] = out[best_map[best["model"]]]

    score_df = pd.DataFrame(candidates).rename(columns={"model": "Model", "aic": "AIC", "llf": "LogLik"})
    score_df = score_df.sort_values(metric_name, ascending=lower_is_better).reset_index(drop=True)
    return out, best, metric_name, score_df


@st.cache_data(show_spinner=False)
def fit_severity(df: pd.DataFrame, tail_q: float) -> tuple[pd.DataFrame, dict, str, dict, pd.DataFrame]:
    sev_train = df[df["ClaimAmount"] > 0].copy()
    sev_formula = "ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + log_density"

    gamma_fit = glm(sev_formula, data=sev_train, family=Gamma()).fit()

    threshold = sev_train["ClaimAmount"].quantile(tail_q)
    sev_train["is_tail"] = (sev_train["ClaimAmount"] > threshold).astype(int)

    body = sev_train[sev_train["ClaimAmount"] <= threshold].copy()
    body_fit = glm(sev_formula, data=body, family=Gamma()).fit()
    tail_prob_fit = glm(
        "is_tail ~ VehPower + VehAge + DrivAge + BonusMalus + C(Region) + log_density",
        data=sev_train,
        family=Binomial(),
    ).fit()

    excess = sev_train.loc[sev_train["is_tail"] == 1, "ClaimAmount"] - threshold
    if len(excess) > 10 and float(excess.mean()) > 0:
        xi, _, beta = genpareto.fit(excess, floc=0)
        if xi >= 1:
            xi = 0.95
    else:
        xi, beta = 0.2, max(float(excess.mean()), 1.0)

    phi = float(body_fit.scale)
    shape_body = 1.0 / max(phi, 1e-9)
    mu_body = np.clip(body_fit.predict(sev_train), 1e-9, None)
    scale_body = np.clip(mu_body * phi, 1e-9, None)
    p_tail = np.clip(tail_prob_fit.predict(sev_train), 1e-8, 1 - 1e-8)
    y = np.clip(sev_train["ClaimAmount"].to_numpy(), 1e-9, None)

    logpdf_body = gamma_dist.logpdf(y, a=shape_body, scale=scale_body)
    excess_y = np.clip(y - threshold, 0, None)
    logpdf_tail = genpareto.logpdf(excess_y, c=xi, loc=0, scale=max(beta, 1e-9))
    is_tail = y > threshold
    gp_llf = float(np.sum(np.where(is_tail, np.log(p_tail) + logpdf_tail, np.log(1 - p_tail) + logpdf_body)))

    expected_excess = beta / (1 - xi) if xi < 1 else (float(excess.mean()) if len(excess) else beta)
    tail_mean = threshold + expected_excess

    candidates = [
        {"model": "Gamma", "aic": gamma_fit.aic, "llf": gamma_fit.llf},
        {"model": "Gamma-Pareto", "aic": np.nan, "llf": gp_llf},
    ]
    metric_name, metric_key, lower_is_better = choose_metric(candidates)
    best = (
        min(candidates, key=lambda r: r[metric_key])
        if lower_is_better
        else max(candidates, key=lambda r: r[metric_key])
    )

    out = df.copy()
    out["pred_sev_gamma"] = np.clip(gamma_fit.predict(df), 1e-9, None)
    full_tail_prob = np.clip(tail_prob_fit.predict(df), 1e-8, 1 - 1e-8)
    out["pred_sev_gamma_pareto"] = np.clip(
        (1 - full_tail_prob) * np.clip(body_fit.predict(df), 1e-9, None) + full_tail_prob * tail_mean,
        1e-9,
        None,
    )
    out["pred_sev_best"] = np.where(best["model"] == "Gamma", out["pred_sev_gamma"], out["pred_sev_gamma_pareto"])

    score_df = pd.DataFrame(candidates).rename(columns={"model": "Model", "aic": "AIC", "llf": "LogLik"})
    score_df = score_df.sort_values(metric_name, ascending=lower_is_better).reset_index(drop=True)
    tail_info = {
        "threshold": float(threshold),
        "tail_count": int((sev_train["ClaimAmount"] > threshold).sum()),
        "shape": float(xi),
        "scale": float(beta),
    }
    return out, best, metric_name, tail_info, score_df


df = load_data()
st.success(f"Loaded {len(df):,} records.")

tail_q = st.sidebar.slider("Tail quantile (Gamma-Pareto)", 0.85, 0.99, 0.95, 0.01)

with st.spinner("Training frequency models..."):
    freq_df, best_freq, freq_metric, freq_scores = fit_frequency(df)

with st.spinner("Training severity models..."):
    sev_df, best_sev, sev_metric, tail_info, sev_scores = fit_severity(df, tail_q)

result = df.copy()
result["pred_freq_best"] = freq_df["pred_freq_best"].to_numpy()
result["pred_sev_best"] = sev_df["pred_sev_best"].to_numpy()
result["pure_premium_best"] = result["pred_freq_best"] * result["pred_sev_best"]

st.header("Frequency Comparison")
st.write(f"Metric used: **{freq_metric}**")
st.write(f"Selected model: **{best_freq['model']}**")
st.dataframe(freq_scores, use_container_width=True)
st.dataframe(
    pd.DataFrame(
        {
            "Model": ["Poisson", "Negative Binomial", "ZIP"],
            "Predicted Mean Frequency": [
                float(freq_df["pred_freq_poisson"].mean()),
                float(freq_df["pred_freq_nb"].mean()),
                float(freq_df["pred_freq_zip"].mean()),
            ],
        }
    ),
    use_container_width=True,
)

st.header("Severity Comparison")
st.write(f"Metric used: **{sev_metric}**")
st.write(f"Selected model: **{best_sev['model']}**")
st.dataframe(sev_scores, use_container_width=True)
st.caption(
    f"Tail diagnostics: threshold={tail_info['threshold']:.2f}, "
    f"tail_count={tail_info['tail_count']}, xi={tail_info['shape']:.3f}, beta={tail_info['scale']:.2f}"
)
st.dataframe(
    pd.DataFrame(
        {
            "Model": ["Gamma", "Gamma-Pareto"],
            "Predicted Mean Severity": [
                float(sev_df["pred_sev_gamma"].mean()),
                float(sev_df["pred_sev_gamma_pareto"].mean()),
            ],
        }
    ),
    use_container_width=True,
)

st.header("Best-Model Output")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Avg Frequency", f"{result['pred_freq_best'].mean():.4f}")
with col2:
    st.metric("Avg Severity", f"{result['pred_sev_best'].mean():.2f}")
with col3:
    st.metric("Avg Pure Premium", f"{result['pure_premium_best'].mean():.2f}")

st.subheader("Sample Predictions")
st.dataframe(
    result[["ClaimNb", "ClaimAmount", "pred_freq_best", "pred_sev_best", "pure_premium_best"]].head(25),
    use_container_width=True,
)
