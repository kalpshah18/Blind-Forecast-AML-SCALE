"""Idea 3: per-station ridge / LAD (median) regression on scaled anomalies.
Run: python linear.py            -> CV table
     python linear.py submit     -> fit on all data, write submission_linear.csv
"""
import sys
import numpy as np
from common import *


def feats(X, d, seasonal):
    """X (N,10,4) anomalies, d (N,) doy -> design matrix."""
    N = len(X)
    lags = X.reshape(N, -1)
    S = season_feats(d, 2)  # 1, s1, c1, s2, c2
    cols = [lags, S]
    if seasonal:  # lag x first-harmonic interactions let dynamics vary with season
        cols += [lags * S[:, [1]], lags * S[:, [2]]]
    return np.hstack(cols)


def ridge(F, Y, lam):
    P = np.eye(F.shape[1]) * lam * len(F)
    return np.linalg.solve(F.T @ F + P, F.T @ Y)


def lad(F, Y, lam, iters=8, eps=1e-3, stride=3):
    """IRLS for least absolute deviation, one weight vector per output column.
    Stride-subsamples rows (adjacent windows are near-duplicates) to keep it fast."""
    F, Y = F[::stride], Y[::stride]
    W0 = ridge(F, Y, lam)
    out = np.empty_like(W0)
    P = np.eye(F.shape[1]) * lam * len(F)
    for j in range(Y.shape[1]):
        w = W0[:, j]
        for _ in range(iters):
            r = Y[:, j] - F @ w
            u = 1.0 / np.maximum(np.abs(r), eps)
            Fu = F * u[:, None]
            w = np.linalg.solve(F.T @ Fu + P, Fu.T @ Y[:, j])
        out[:, j] = w
    return out


MODELS = {
    "ridge":      lambda F, Y: ridge(F, Y, 1e-3),
    "ridge_seas": lambda F, Y: ridge(F, Y, 1e-3),
    "lad":        lambda F, Y: lad(F, Y, 1e-4),
    "lad_seas":   lambda F, Y: lad(F, Y, 1e-4),
}


def station_windows(A, doy, spans):
    parts = [make_windows(A, doy, lo, hi) for lo, hi in spans]
    return [np.concatenate(z) for z in zip(*parts)]


def fit_predict(name, train, test_X_d):
    """train: (X,Y,d); test_X_d: (X,d) -> predictions (N,5,4)."""
    seasonal = name.endswith("_seas")
    X, Y, d = train
    W = MODELS[name](feats(X, d, seasonal), Y.reshape(len(Y), -1))
    Xt, dt = test_X_d
    return (feats(Xt, dt, seasonal) @ W).reshape(-1, OUT, 4)


def cv(names, n_folds=5, k_clim=4):
    raw, doy = load_train()
    S, T, _ = raw.shape
    folds = cv_folds(T, n_folds)
    res = {n: [] for n in names + ["clim", "persist"]}
    steps = {n: [] for n in names + ["persist"]}
    for lo, hi in folds:
        mask = np.ones((S, T), bool); mask[:, lo:hi] = False
        coefs = fit_climatology(raw, doy, mask, k_clim)
        A = to_anom(raw, doy, coefs)
        train_spans = [(a, b) for a, b in [(0, lo), (hi, T)] if b - a > IN + OUT]
        preds = {n: [] for n in names}; trues, pers = [], []
        for s in range(S):
            tr = station_windows(A[s], doy[s], train_spans)
            Xv, Yv, dv = make_windows(A[s], doy[s], lo, hi)
            trues.append(Yv); pers.append(np.repeat(Xv[:, -1:], OUT, 1))
            for n in names:
                preds[n].append(fit_predict(n, tr, (Xv, dv)))
        true = np.concatenate(trues)
        res["clim"].append(score_cols(np.zeros_like(true), true)[0])
        res["persist"].append(score_cols(np.concatenate(pers), true)[0])
        steps["persist"].append(score_cols(np.concatenate(pers), true)[1])
        for n in names:
            p = np.concatenate(preds[n])
            res[n].append(score_cols(p, true)[0]); steps[n].append(score_cols(p, true)[1])
    print(f"{'model':12s} {'score':>7s}  " + " ".join(f"{v:>6s}" for v in VARS) + "   per-fold score")
    for n, r in res.items():
        r = np.array(r)
        print(f"{n:12s} {r.mean(1).mean():7.4f}  " + " ".join(f"{x:6.3f}" for x in r.mean(0))
              + "   " + " ".join(f"{x:.3f}" for x in r.mean(1)))
    print("\nper-step score (mean over vars), steps 1..5")
    for n, s in steps.items():
        print(f"{n:12s} " + " ".join(f"{x:.3f}" for x in np.array(s).mean(0).mean(1)))


def submit(name, k_clim=4):
    raw, doy = load_train()
    S, T, _ = raw.shape
    coefs = fit_climatology(raw, doy, None, k_clim)
    A = to_anom(raw, doy, coefs)
    wids, st, X, d, tdoy = load_test(coefs)
    pred = np.zeros((len(wids), OUT, 4))
    for s in range(S):
        tr = station_windows(A[s], doy[s], [(0, T)])
        m = st == s
        pred[m] = fit_predict(name, tr, (X[m], d[m]))
    write_submission(wids, st, pred, tdoy, coefs, f"submission_{name}.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        submit(sys.argv[2] if len(sys.argv) > 2 else "lad_seas")
    else:
        cv(["ridge", "ridge_seas", "lad", "lad_seas"])
