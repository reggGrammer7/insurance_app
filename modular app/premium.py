import numpy as np

def apply_adjustments(df, deductible, inflation, years,
                      expense_loading, profit_loading):

    inflation_factor = (1 + inflation) ** years

    df["Adj_Sev"] = np.maximum(df["Pred_Sev"] - deductible, 0) * inflation_factor
    df["Pure_Premium"] = df["Pred_Freq"] * df["Adj_Sev"]

    df["Technical_Premium"] = df["Pure_Premium"] * (
        1 + expense_loading + profit_loading
    )

    return df
