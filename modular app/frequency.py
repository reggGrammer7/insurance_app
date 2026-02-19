import numpy as np
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson, NegativeBinomial

def train_frequency_models(train_df):

    poisson_model = glm(
        formula="ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=train_df,
        family=Poisson(),
        offset=np.log(train_df["Exposure"])
    ).fit()

    nb_model = glm(
        formula="ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus",
        data=train_df,
        family=NegativeBinomial(),
        offset=np.log(train_df["Exposure"])
    ).fit()

    return poisson_model, nb_model


def predict_frequency(model, df):
    return model.predict(df, offset=np.log(df["Exposure"]))
