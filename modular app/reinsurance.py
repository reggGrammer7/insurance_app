import numpy as np

def apply_quota_share(losses, quota):
    ceded = losses * quota
    net = losses - ceded
    return ceded.sum(), net.sum()


def apply_xol(losses, retention):
    ceded = np.maximum(losses - retention, 0)
    net = losses - ceded
    return ceded.sum(), net.sum()
