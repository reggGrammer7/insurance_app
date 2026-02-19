import numpy as np
import pandas as pd
import streamlit as st

from data import load_data
from frequency import train_frequency_models, predict_frequency
from severity import train_gamma_model, train_lognormal_model
from reserving import chain_ladder
from capitalsim import simulate_aggregate_losses
from premium import apply_adjustments
from reinsurance import apply_quota_share, apply_xol


def _split_train_test(df: pd.DataFrame, train_frac: float = 0.8, seed: int = 42):
    train_df = df.sample(frac=train_frac, random_state=seed)
    test_df = df.drop(train_df.index)
    return train_df.copy(), test_df.copy()


def _build_proxy_triangle(df: pd.DataFrame, n_accident_years: int = 6, n_dev_periods: int = 6):
    """Build a simple cumulative triangle from claim data for reserving demo."""
    work = df[df["ClaimAmount"] > 0].copy()
    if work.empty:
        idx = [f"AY{i + 1}" for i in range(n_accident_years)]
        cols = [f"Dev{j + 1}" for j in range(n_dev_periods)]
        base = pd.DataFrame(0.0, index=idx, columns=cols)
        for i in range(n_accident_years):
            for j in range(n_dev_periods):
                if j <= n_dev_periods - i - 1:
                    base.iloc[i, j] = (i + 1) * (j + 1) * 10.0
                else:
                    base.iloc[i, j] = np.nan
        return base

    work = work.reset_index(drop=True)
    work["AY"] = work.index % n_accident_years
    work["Dev"] = (work.index // n_accident_years) % n_dev_periods

    incremental = (
        work.groupby(["AY", "Dev"], as_index=False)["ClaimAmount"]
        .sum()
        .pivot(index="AY", columns="Dev", values="ClaimAmount")
        .reindex(index=range(n_accident_years), columns=range(n_dev_periods), fill_value=0.0)
    )

    cumulative = incremental.cumsum(axis=1)
    cumulative.index = [f"AY{i + 1}" for i in cumulative.index]
    cumulative.columns = [f"Dev{j + 1}" for j in range(n_dev_periods)]

    for i in range(n_accident_years):
        max_observed_dev = n_dev_periods - i - 1
        if max_observed_dev < n_dev_periods - 1:
            cumulative.iloc[i, max_observed_dev + 1 :] = np.nan

    return cumulative


@st.cache_data(show_spinner=False)
def load_data_cached():
    return load_data()


def run_pipeline(
    train_frac: float,
    deductible: float,
    inflation: float,
    years: int,
    expense_loading: float,
    profit_loading: float,
    quota: float,
    n_sim: int,
    seed: int,
):
    df = load_data_cached()

    train_df, test_df = _split_train_test(df, train_frac=train_frac)

    poisson_model, nb_model = train_frequency_models(train_df)
    test_df["Pred_Freq_Pois"] = predict_frequency(poisson_model, test_df)
    test_df["Pred_Freq_NB"] = predict_frequency(nb_model, test_df)
    test_df["Pred_Freq"] = test_df["Pred_Freq_NB"].clip(lower=0)

    sev_train = train_df[train_df["ClaimAmount"] > 0].copy()
    gamma_model = train_gamma_model(sev_train)
    lognorm_model, lognorm_mse = train_lognormal_model(sev_train)

    sev_gamma = gamma_model.predict(test_df).clip(lower=0)
    sev_lognorm = np.exp(lognorm_model.predict(test_df) + 0.5 * lognorm_mse)

    test_df["Pred_Sev_Gamma"] = sev_gamma
    test_df["Pred_Sev_Lognorm"] = sev_lognorm
    test_df["Pred_Sev"] = test_df["Pred_Sev_Gamma"]

    priced_df = apply_adjustments(
        test_df,
        deductible=200.0,
        inflation=0.05,
        years=1,
        expense_loading=0.20,
        profit_loading=0.05,
    )

    triangle = _build_proxy_triangle(df)
    reserve_result = chain_ladder(triangle)

    lambda_mean = float((priced_df["Pred_Freq"] * priced_df["Exposure"]).mean())
    positive_sev = priced_df.loc[priced_df["Adj_Sev"] > 0, "Adj_Sev"]
    sev_mean = float(positive_sev.mean()) if not positive_sev.empty else 1.0
    sev_var = float(positive_sev.var()) if len(positive_sev) > 1 else max(sev_mean, 1.0)

    agg_losses = simulate_aggregate_losses(
        lambda_mean=lambda_mean,
        sev_mean=max(sev_mean, 1e-6),
        sev_var=max(sev_var, 1e-6),
        n_sim=n_sim,
        seed=seed,
    )

    qs_ceded, qs_net = apply_quota_share(agg_losses, quota=quota)
    xol_ceded, xol_net = apply_xol(agg_losses, retention=np.percentile(agg_losses, 90))

    return {
        "rows_loaded": len(df),
        "priced_df": priced_df,
        "triangle": triangle,
        "reserve_result": reserve_result,
        "agg_losses": agg_losses,
        "summary": {
            "avg_technical_premium": float(priced_df["Technical_Premium"].mean()),
            "total_ibnr": float(reserve_result["total_ibnr"]),
            "qs_ceded": float(qs_ceded),
            "qs_net": float(qs_net),
            "xol_ceded": float(xol_ceded),
            "xol_net": float(xol_net),
        },
    }


def main():
    st.set_page_config(page_title="Insurance Modular App", layout="wide")
    st.title("Insurance Modular App")
    st.caption("Frequency, severity, premium, reserving, capital simulation, and reinsurance")

    with st.sidebar:
        st.header("Parameters")
        train_frac = st.slider("Train fraction", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
        deductible = st.number_input("Deductible", min_value=0.0, value=200.0, step=50.0)
        inflation = st.number_input("Inflation", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        years = st.number_input("Projection years", min_value=0, max_value=30, value=1, step=1)
        expense_loading = st.number_input("Expense loading", min_value=0.0, max_value=2.0, value=0.20, step=0.01)
        profit_loading = st.number_input("Profit loading", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
        quota = st.slider("Quota share", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
        n_sim = st.number_input("Simulation runs", min_value=1000, max_value=200000, value=10000, step=1000)
        seed = st.number_input("Random seed", min_value=0, max_value=1000000, value=42, step=1)
        run_clicked = st.button("Run Analysis", type="primary")

    if not run_clicked:
        st.info("Set parameters in the sidebar and click Run Analysis.")
        return

    try:
        with st.spinner("Running actuarial pipeline..."):
            result = run_pipeline(
                train_frac=float(train_frac),
                deductible=float(deductible),
                inflation=float(inflation),
                years=int(years),
                expense_loading=float(expense_loading),
                profit_loading=float(profit_loading),
                quota=float(quota),
                n_sim=int(n_sim),
                seed=int(seed),
            )
    except Exception as exc:
        st.error("Pipeline failed. See details below.")
        st.exception(exc)
        st.stop()

    summary = result["summary"]
    st.success(f"Loaded rows: {result['rows_loaded']:,}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Technical Premium", f"{summary['avg_technical_premium']:,.2f}")
    c2.metric("Total IBNR", f"{summary['total_ibnr']:,.2f}")
    c3.metric("Quota Share Net", f"{summary['qs_net']:,.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Quota Share Ceded", f"{summary['qs_ceded']:,.2f}")
    c5.metric("XoL Net", f"{summary['xol_net']:,.2f}")
    c6.metric("XoL Ceded", f"{summary['xol_ceded']:,.2f}")

    st.subheader("Premium Output Sample")
    st.dataframe(
        result["priced_df"][
            [
                "Exposure",
                "ClaimNb",
                "ClaimAmount",
                "Pred_Freq",
                "Pred_Sev",
                "Adj_Sev",
                "Pure_Premium",
                "Technical_Premium",
            ]
        ].head(50)
    )

    st.subheader("Reserving")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Observed Cumulative Triangle")
        st.dataframe(result["triangle"])
    with col_b:
        st.write("IBNR Summary")
        st.dataframe(result["reserve_result"]["ibnr_summary"])

    st.subheader("Aggregate Loss Simulation")
    sim_series = pd.Series(result["agg_losses"], name="aggregate_loss")
    st.line_chart(sim_series.rolling(window=100, min_periods=1).mean())
    st.caption("Chart shows rolling mean of simulated aggregate losses (window=100).")


if __name__ == "__main__":
    main()
