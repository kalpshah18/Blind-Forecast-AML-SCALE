"""Test-time climatology adaptation + blend.
Test windows are from a shifted regime (per-station level shifts vs. train climatology). We fit a
seasonal (k harmonics) per-station offset from the unlabelled test inputs, add it to the climatology,
predict anomalies relative to the adapted climatology, and blend with OOF-fit weights.
Run: python adapt.py [out.csv]
"""
import sys
import numpy as np
import linear, gbm, blend, nn, analog
from common import *

K_OFF = 1
KSEL = False  # per-cell offset order by held-out residual: scored worse on the public LB (0.6798 vs 0.6753), off by default
MODELS = ["lad_seas", "ridge_seas", "gbm"]
DEADBAND = (0.8, 1.25)          # leave scale alone when the test/train dispersion ratio is inside this
POOLED_CELLS = [(0, 3)]         # (station, variable) whose own train dynamics differ from test (S0 wind)
NORMAL = [1, 2, 3, 4, 5, 7, 9]  # stations whose own-vs-test behaviour matches; used to train pooled models


def dispersion_ratio(A, doy, X, d, st, w=10):
    """Season-matched ratio of test mean|resid| to train mean|anom| per (station, var) -> (S,4)."""
    S = A.shape[0]
    din = d[:, None] + np.arange(-(IN - 1), 1)
    r = np.ones((S, 4))
    for s in range(S):
        m = st == s
        days = din[m].reshape(-1)
        resid = np.abs(X[m]).reshape(-1, 4)
        for v in range(4):
            dd = np.abs(((doy[s][None, :] - np.unique(days)[:, None] + 182.6) % 365.25) - 182.6)
            prof = np.array([np.abs(A[s][row <= w, v]).mean() for row in dd])
            cnt = np.array([(days == c).sum() for c in np.unique(days)])
            r[s, v] = resid[:, v].mean() / ((prof * cnt).sum() / cnt.sum())
    return np.clip(r, 0.5, 4.0)


def select_k(coefs, ks=(0, 1, 2), reps=20, margin=0.02, lam=1e-2):
    """Per-(station, var) offset harmonic order by repeated 50/50 held-out |resid| on the test inputs.
    Stays at k=1 unless another order wins by more than `margin` (relative)."""
    wids, st, X, d, _ = load_test(coefs)
    din = d[:, None] + np.arange(-(IN - 1), 1)
    rng = np.random.default_rng(0)
    err = {k: np.zeros((coefs.shape[0], 4)) for k in ks}
    for _ in range(reps):
        for s in range(coefs.shape[0]):
            idx = np.where(st == s)[0]; rng.shuffle(idx)
            a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
            for k in ks:
                Ha = harmonics(din[a].reshape(-1), k)
                beta = np.linalg.solve(Ha.T @ Ha + lam * np.eye(2 * k + 1), Ha.T @ X[a].reshape(-1, 4))
                err[k][s] += np.abs(X[b].reshape(-1, 4) - harmonics(din[b].reshape(-1), k) @ beta).mean(0)
    kmat = np.full((coefs.shape[0], 4), K_OFF)
    for s in range(kmat.shape[0]):
        for v in range(4):
            best = min(ks, key=lambda k: err[k][s, v])
            if err[best][s, v] < (1 - margin) * err[K_OFF][s, v]:
                kmat[s, v] = best
    return kmat


def adapted_coefs(coefs, k=K_OFF, lam=1e-2, kmat=None):
    wids, st, X, d, _ = load_test(coefs)
    din = d[:, None] + np.arange(-(IN - 1), 1)
    kmat = np.full((coefs.shape[0], 4), k) if kmat is None else kmat
    new = coefs.copy()
    for s in range(coefs.shape[0]):
        m = st == s
        for v in range(4):
            kk = kmat[s, v]
            H = harmonics(din[m].reshape(-1), kk)
            y = X[m][:, :, v].reshape(-1)
            beta = np.linalg.solve(H.T @ H + lam * np.eye(H.shape[1]), H.T @ y)
            new[s, : 2 * kk + 1, v] += beta * SCALE[v]
    return new


def main(out, models=MODELS, weights=None):
    raw, doy = load_train(); S, T, _ = raw.shape
    coefs = fit_climatology(raw, doy, None, 4)
    A = to_anom(raw, doy, coefs)
    kmat = select_k(coefs) if KSEL else None
    if KSEL:
        print("offset harmonic order per cell (rows=stations; cols=tmax,tmin,rh,wind):\n", kmat)
    ca = adapted_coefs(coefs, kmat=kmat)
    wids, st, X, d, tdoy = load_test(ca)
    print("adapted test input mean anomaly:", X.mean((0, 1)).round(3), "(was nonzero before)")
    sig = np.abs(A).mean(1)                                  # (S,4) train mean|anom|
    r_full = dispersion_ratio(A, doy, X, d, st)
    r = np.where((r_full > DEADBAND[0]) & (r_full < DEADBAND[1]), 1.0, r_full)
    print("scale ratios used (rows=stations, cols=tmax,tmin,rh,wind):\n", r.round(2))
    Xraw = X
    X = X / r[st][:, None, :]                                # inputs in train-like dispersion
    P = {}
    pooled = {"gbm": (gbm.fit_predict, gbm.STRIDE), "nn": (nn.fit_predict, nn.STRIDE),
              "analog": (analog.fit_predict, analog.STRIDE),
              "nn6": (nn.fit_predict6, nn.STRIDE),
              "nn6a": (nn.fit_predict6a, nn.STRIDE),
              "nn12b": (nn.fit_predict12b, nn.STRIDE)}
    for name in models:
        if name in pooled:
            fp, stride = pooled[name]
            P[name] = fp(gbm.build(A, doy, [(0, T)], range(S), stride), (X, d, st))
        else:
            p = np.zeros((len(wids), OUT, 4))
            for s in range(S):
                m = st == s
                p[m] = linear.fit_predict(name, linear.station_windows(A[s], doy[s], [(0, T)]), (X[m], d[m]))
            P[name] = p
    # blend weights: stored file if given, else fitted on the cached OOF (in-distribution) predictions
    if not weights:
        import os
        proxy = [f"blend_weights_{m}_proxy.json" for m in models if not os.path.exists(f"oof/{m}_oof.npy")]
        if proxy and os.path.exists(proxy[0]):   # no OOF cache for this model: use stored proxy weights
            weights = proxy[0]
            print("no OOF cache for the requested models; using proxy blend weights from", weights)
    if weights:
        W, G = blend.load_weights(models, weights)
    else:
        O = np.stack([np.load(f"oof/{m}_oof.npy") for m in models]); Y = np.load("oof/truth.npy")
        W, G = blend.fit_weights(O, Y, blend.simplex(len(models)), blend.SHRINKS)
    pred = blend.apply(np.stack([P[m] for m in models]), W, G) * r[st][:, None, :]
    # pooled station-normalised model for cells whose own train dynamics are unrepresentative
    if POOLED_CELLS:
        parts = [linear.station_windows(A[o] / sig[o], doy[o], [(0, T)]) for o in NORMAL]
        pool = [np.concatenate(z) for z in zip(*parts)]
        for s_, v_ in POOLED_CELLS:
            m = st == s_
            Xz = Xraw[m] / (sig[s_] * r_full[s_])
            pz = linear.fit_predict("lad_seas", pool, (Xz, d[m])) * (sig[s_] * r_full[s_])
            pred[m, :, v_] = pz[:, :, v_]
    write_submission(wids, st, pred, tdoy, ca, out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "submission_adapt_blend.csv", a[1:] or MODELS)
