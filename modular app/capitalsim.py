import numpy as np

def simulate_aggregate_losses(lambda_mean, sev_mean, sev_var,
                              n_sim=10000, seed=42):

    rng = np.random.default_rng(seed)

    shape = (sev_mean**2) / sev_var
    scale = sev_var / sev_mean

    freq = rng.poisson(lambda_mean, n_sim)
    sev = rng.gamma(shape, scale, n_sim)

    agg = freq * sev

    return agg
