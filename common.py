"""Shared data, climatology, windowing, CV and submission helpers."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

VARS = ["tmax", "tmin", "rh", "wind"]
IN, OUT = 10, 5
STATIONS = [f"S{i}" for i in range(10)]
_scales = pd.read_csv("scales.csv").set_index("variable")["scale"]
SCALE = np.array([_scales[v] for v in VARS])


def load_train():
    """-> raw (S, T, 4) array and doy (S, T) array, stations in STATIONS order."""
    tr = pd.read_csv("train.csv").sort_values(["station", "day_index"])
    raw = np.stack([tr[tr.station == s][VARS].to_numpy() for s in STATIONS])
    doy = np.stack([tr[tr.station == s]["doy"].to_numpy() for s in STATIONS])
    return raw, doy


def harmonics(doy, k):
    a = 2 * np.pi * np.asarray(doy, float) / 365.25
    cols = [np.ones_like(a)]
    for j in range(1, k + 1):
        cols += [np.sin(j * a), np.cos(j * a)]
    return np.stack(cols, -1)


def fit_climatology(raw, doy, mask=None, k=4):
    """Harmonic regression per station/variable. Returns coefs (S, 2k+1, 4).
    mask (S, T) bool restricts which days are used for fitting."""
    S = raw.shape[0]
    coefs = np.zeros((S, 2 * k + 1, 4))
    for s in range(S):
        m = np.ones(raw.shape[1], bool) if mask is None else mask[s]
        H = harmonics(doy[s][m], k)
        coefs[s] = np.linalg.lstsq(H, raw[s][m], rcond=None)[0]
    return coefs


def climatology(coefs, s, doy):
    k = (coefs.shape[1] - 1) // 2
    return harmonics(doy, k) @ coefs[s]


def to_anom(raw, doy, coefs):
    """Scaled anomalies (S, T, 4)."""
    out = np.empty_like(raw, dtype=float)
    for s in range(raw.shape[0]):
        out[s] = (raw[s] - climatology(coefs, s, doy[s])) / SCALE
    return out


def make_windows(A, doy, t_lo=0, t_hi=None):
    """All stride-1 windows fully inside [t_lo, t_hi) for station arrays.
    Returns X (N,10,4), Y (N,5,4), d (N,) doy of last input day."""
    t_hi = A.shape[0] if t_hi is None else t_hi
    a = A[t_lo:t_hi]
    w = sliding_window_view(a, IN + OUT, axis=0).transpose(0, 2, 1)  # (N,15,4)
    d = doy[t_lo:t_hi][IN - 1 : IN - 1 + len(w)]
    return w[:, :IN], w[:, IN:], d


def season_feats(d, k=2):
    return harmonics(d, k)  # (N, 2k+1) incl. constant


def cv_folds(T, n=5):
    edges = np.linspace(0, T, n + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(n)]


def score_cols(pred, true):
    """pred,true (N,5,4) scaled anomalies -> per-var MAE (4,) and per-step-var (5,4)."""
    e = np.abs(pred - true)
    return e.mean((0, 1)), e.mean(0)


def load_test(coefs):
    """-> ids list of window ids, stations idx, X (N,10,4) scaled anomalies, d (N,), target doy (N,5)."""
    te = pd.read_csv("test.csv")
    obs = te[te.step <= 0].sort_values(["window_id", "step"])
    wids = obs.window_id.unique()
    st = obs.groupby("window_id").station.first().loc[wids].map(STATIONS.index).to_numpy()
    raw = obs[VARS].to_numpy().reshape(len(wids), IN, 4)
    doys = obs["doy"].to_numpy().reshape(len(wids), IN)
    X = np.empty_like(raw)
    for i, s in enumerate(st):
        X[i] = (raw[i] - climatology(coefs, s, doys[i])) / SCALE
    d = doys[:, -1]
    tdoy = d[:, None] + np.arange(1, OUT + 1)
    return wids, st, X, d, tdoy


def write_submission(wids, st, pred_anom, tdoy, coefs, path):
    """pred_anom (N,5,4) scaled anomalies -> submission csv of value/scale (= clim/scale + anom)."""
    rows = []
    for i, w in enumerate(wids):
        for j in range(OUT):
            clim = climatology(coefs, st[i], tdoy[i, j]) / SCALE
            val = clim + pred_anom[i, j]
            for v, name in enumerate(VARS):
                rows.append((f"{w}_{j+1}_{name}", val[v]))
    sub = pd.DataFrame(rows, columns=["id", "value"])
    ref = pd.read_csv("sample_submission.csv")
    assert len(sub) == len(ref) and set(sub.id) == set(ref.id), "id mismatch"
    assert not sub.value.isna().any()
    sub.set_index("id").loc[ref.id].reset_index().to_csv(path, index=False)
    print("wrote", path, len(sub))
