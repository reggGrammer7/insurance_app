import numpy as np
import pandas as pd


def calculate_development_factors(triangle: pd.DataFrame):
    """
    Calculate age-to-age development factors.
    """
    n_dev = triangle.shape[1]
    dev_factors = []

    for j in range(n_dev - 1):
        col_current = triangle.iloc[:, j]
        col_next = triangle.iloc[:, j + 1]

        mask = col_current.notna() & col_next.notna()

        numerator = col_next[mask].sum()
        denominator = col_current[mask].sum()

        factor = numerator / denominator if denominator > 0 else 1.0
        dev_factors.append(float(factor))

    return dev_factors


def complete_triangle(triangle: pd.DataFrame, dev_factors):
    """
    Complete cumulative triangle using development factors.
    """
    completed = triangle.copy()
    n_dev = completed.shape[1]

    for i in range(completed.shape[0]):
        row = completed.iloc[i]

        # Find last observed development
        last_valid = row.last_valid_index()
        if last_valid is None:
            continue

        last_pos = completed.columns.get_loc(last_valid)

        for j in range(last_pos + 1, n_dev):
            prev_value = completed.iloc[i, j - 1]
            completed.iloc[i, j] = prev_value * dev_factors[j - 1]

    return completed


def calculate_ibnr(original_triangle: pd.DataFrame, completed_triangle: pd.DataFrame):
    """
    Compute Latest, Ultimate, and IBNR for each accident year.
    """
    latest_list = []
    ultimate_list = []
    ibnr_list = []

    for i in range(original_triangle.shape[0]):

        observed_row = original_triangle.iloc[i].dropna()
        latest = observed_row.iloc[-1] if len(observed_row) > 0 else 0.0
        ultimate = completed_triangle.iloc[i, -1]

        latest_list.append(float(latest))
        ultimate_list.append(float(ultimate))
        ibnr_list.append(float(ultimate - latest))

    summary = pd.DataFrame(
        {
            "Latest": latest_list,
            "Ultimate": ultimate_list,
            "IBNR": ibnr_list,
        },
        index=original_triangle.index,
    )

    return summary


def chain_ladder(triangle: pd.DataFrame):
    """
    Full Chain Ladder pipeline:
    1. Calculate development factors
    2. Complete triangle
    3. Calculate IBNR summary
    """

    triangle = triangle.apply(pd.to_numeric, errors="coerce")

    dev_factors = calculate_development_factors(triangle)
    completed = complete_triangle(triangle, dev_factors)
    summary = calculate_ibnr(triangle, completed)

    return {
        "development_factors": dev_factors,
        "completed_triangle": completed,
        "ibnr_summary": summary,
        "total_ibnr": float(summary["IBNR"].sum()),
    }
