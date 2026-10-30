"""Idea 4: analog forecasting. Per station, find the K most similar historical 10-day anomaly
trajectories (same season +-SEAS days) and return the median of what followed."""
import numpy as np
from common import *

STRIDE = 1
K, SEAS, DECAY = 40, 25, 0.85
W = (DECAY ** np.arange(IN - 1, -1, -1))[:, None] * np.ones((1, 4))  # recent days matter most
W = (W / W.mean()).reshape(-1)


def fit_predict(train, test, chunk=800):
    X, Y, d, s = train
    Xt, dt, st = test
    out = np.zeros((len(Xt), OUT, 4))
    for stn in np.unique(st):
        m = s == stn
        Fa = X[m].reshape(m.sum(), -1) * np.sqrt(W); Ya = Y[m]; da = d[m]
        q = np.where(st == stn)[0]
        Fq = Xt[q].reshape(len(q), -1) * np.sqrt(W)
        na = (Fa ** 2).sum(1)
        for i in range(0, len(q), chunk):
            sl = q[i:i + chunk]; fq = Fq[i:i + chunk]
            D = na[None, :] - 2 * fq @ Fa.T + (fq ** 2).sum(1)[:, None]
            dd = np.abs(((da[None, :] - dt[sl][:, None] + 182.6) % 365.25) - 182.6)
            D[dd > SEAS] = np.inf
            idx = np.argpartition(D, K, axis=1)[:, :K]
            out[sl] = np.median(Ya[idx], axis=1)
    return out
