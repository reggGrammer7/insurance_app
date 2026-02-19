import numpy as np
from statsmodels.formula.api import glm, ols
from statsmodels.genmod.families import Gamma

def train_gamma_model(train_df):
    return glm(
        formula="ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=train_df,
        family=Gamma()
    ).fit()


def train_lognormal_model(train_df):
    model = ols(
        formula="np.log(ClaimAmount) ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=train_df
    ).fit()
    return model, model.mse_resid
