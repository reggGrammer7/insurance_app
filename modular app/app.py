import numpy as np
import pandas as pd

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


def run_pipeline():
    df = load_data()
    print(f"Loaded rows: {len(df):,}")

    train_df, test_df = _split_train_test(df)

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
        n_sim=10000,
        seed=42,
    )

    qs_ceded, qs_net = apply_quota_share(agg_losses, quota=0.30)
    xol_ceded, xol_net = apply_xol(agg_losses, retention=np.percentile(agg_losses, 90))

    print("\nPipeline summary")
    print(f"Average technical premium: {priced_df['Technical_Premium'].mean():,.2f}")
    print(f"Total IBNR (proxy triangle): {reserve_result['total_ibnr']:,.2f}")
    print(f"Quota Share ceded / net: {qs_ceded:,.2f} / {qs_net:,.2f}")
    print(f"XoL ceded / net: {xol_ceded:,.2f} / {xol_net:,.2f}")

    return {
        "priced_df": priced_df,
        "triangle": triangle,
        "reserve_result": reserve_result,
        "agg_losses": agg_losses,
    }


if __name__ == "__main__":
    run_pipeline()
