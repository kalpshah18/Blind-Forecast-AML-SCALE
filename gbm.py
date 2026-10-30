"""Idea 1: pooled LightGBM (L1) on scaled-anomaly features, one model per (variable, step).
Run: python gbm.py            -> CV table (same folds as linear.py)
     python gbm.py submit     -> fit on all data, write submission_gbm.csv
"""
import sys
import numpy as np
import lightgbm as lgb
from common import *

PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=31, min_child_samples=100,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
              n_estimators=300, verbose=-1, n_jobs=-1)
STRIDE = 2  # adjacent windows are near-duplicates; halves training cost


def feats(X, d, st):
    """X (N,10,4) scaled anomalies, d (N,) doy of last day, st (N,) station idx."""
    N = len(X)
    last = X[:, -1]
    f = [X.reshape(N, -1),                       # all 40 lags
         last - X[:, -2], last - X[:, -4],       # recent tendencies
         X[:, -3:].mean(1), X[:, -5:].mean(1), X.mean(1), X.std(1),
         X[:, -5:].mean(1) - X[:, :5].mean(1),
         (X[:, :, 0] - X[:, :, 1])[:, -3:],      # tmax-tmin range (cloud/dryness proxy)
         np.abs(X).mean(1),
         season_feats(d, 2)[:, 1:], st[:, None]]
    return np.hstack(f)


def build(A, doy, spans, stations, stride=STRIDE):
    Xs, Ys, ds, ss = [], [], [], []
    for s in stations:
        for lo, hi in spans:
            if hi - lo <= IN + OUT:
                continue
            X, Y, d = make_windows(A[s], doy[s], lo, hi)
            Xs.append(X[::stride]); Ys.append(Y[::stride]); ds.append(d[::stride])
            ss.append(np.full(len(X[::stride]), s))
    return [np.concatenate(z) for z in (Xs, Ys, ds, ss)]


def fit_predict(train, test):
    X, Y, d, s = train
    F = feats(X, d, s)
    Ft = feats(*test)
    pred = np.zeros((len(Ft), OUT, 4))
    for j in range(OUT):
        for v in range(4):
            m = lgb.LGBMRegressor(**PARAMS)
            m.fit(F, Y[:, j, v], categorical_feature=[F.shape[1] - 1])
            pred[:, j, v] = m.predict(Ft)
    return pred


def cv(n_folds=5, k_clim=4):
    raw, doy = load_train()
    S, T, _ = raw.shape
    allp, allt = [], []
    rows = []
    for lo, hi in cv_folds(T, n_folds):
        mask = np.ones((S, T), bool); mask[:, lo:hi] = False
        coefs = fit_climatology(raw, doy, mask, k_clim)
        A = to_anom(raw, doy, coefs)
        tr = build(A, doy, [(0, lo), (hi, T)], range(S))
        Xv, Yv, dv, sv = build(A, doy, [(lo, hi)], range(S))
        p = fit_predict(tr, (Xv, dv, sv))
        v, st = score_cols(p, Yv)
        rows.append(v); allp.append(st)
        print(f"fold {lo}-{hi}: {v.mean():.4f}  " + " ".join(f"{x:.3f}" for x in v), flush=True)
    r = np.array(rows)
    print(f"gbm  {r.mean():.4f}  " + " ".join(f"{x:.3f}" for x in r.mean(0)))
    print("per-step:", " ".join(f"{x:.3f}" for x in np.array(allp).mean(0).mean(1)))


def submit(k_clim=4):
    raw, doy = load_train()
    S, T, _ = raw.shape
    coefs = fit_climatology(raw, doy, None, k_clim)
    A = to_anom(raw, doy, coefs)
    wids, st, X, d, tdoy = load_test(coefs)
    pred = fit_predict(build(A, doy, [(0, T)], range(S)), (X, d, st))
    write_submission(wids, st, pred, tdoy, coefs, "submission_gbm.csv")


if __name__ == "__main__":
    submit() if len(sys.argv) > 1 and sys.argv[1] == "submit" else cv()
