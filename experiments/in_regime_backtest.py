"""In-regime backtest: inside each test window, days -9..-5 forecast days -4..0, whose values are known.
Gives labelled examples from the shifted test regime. Measures whether forecast gains differ from train,
with held-out window splits. Control: the same procedure on a held-out block of train.
Run: python -m experiments.in_regime_backtest"""
import numpy as np, linear, adapt
from common import *

GRID = np.arange(0.5, 1.55, 0.05)
raw, doy = load_train(); S, T, _ = raw.shape
coefs = fit_climatology(raw, doy, None, 4); A0 = to_anom(raw, doy, coefs)
ca = adapt.adapted_coefs(coefs); wids, st, X, d, _ = load_test(ca)
rf = adapt.dispersion_ratio(A0, doy, X, d, st)
r = np.where((rf > 0.8) & (rf < 1.25), 1.0, rf)
X = X / r[st][:, None, :]
cut = int(T * 0.8); rng = np.random.default_rng(0)
mask = np.ones((S, T), bool); mask[:, cut:] = False
A = to_anom(raw, doy, fit_climatology(raw, doy, mask, 4))


def bt(Xw, dw, sw):
    """10-day windows -> pred and truth for the last 5 days from the first 5 (per-station ridge)."""
    P = np.zeros_like(Xw[:, 5:])
    for s in range(S):
        m = sw == s
        if m.any():
            Xa, Ya, da = make_windows(A[s], doy[s], 0, cut)
            P[m] = linear.fit_predict("ridge_seas", (Xa[:, 5:], Ya, da), (Xw[m, :5], dw[m] - 5))
    return P, Xw[:, 5:]


def gain_cv(P, Y, sw, groups, reps=30):
    """Held-out MAE at g=1 vs gain fitted per group on half of the windows."""
    base, fit = [], []
    for _ in range(reps):
        i = rng.permutation(len(P)); a, b = i[:len(i) // 2], i[len(i) // 2:]
        Pg = P.copy()
        G = {}
        for key, sel in groups(sw, P.shape):
            ma = np.zeros(P.shape, bool); ma[a] = True; ma &= sel
            g = GRID[np.argmin([np.abs(x * P[ma] - Y[ma]).mean() for x in GRID])]
            Pg[sel] = P[sel] * g; G[key] = g
        base.append(np.abs(P[b] - Y[b]).mean()); fit.append(np.abs(Pg[b] - Y[b]).mean())
    return np.mean(base), np.mean(fit), G


def by_var(sw, sh):
    for v in range(4):
        m = np.zeros(sh, bool); m[:, :, v] = True; yield VARS[v], m


def by_var_step(sw, sh):
    for v in range(4):
        for j in range(5):
            m = np.zeros(sh, bool); m[:, j, v] = True; yield (VARS[v], j + 1), m


def by_station(sw, sh):
    for s in range(S):
        m = np.zeros(sh, bool); m[sw == s] = True; yield f"S{s}", m


def by_station_var(sw, sh):
    for s in range(S):
        for v in range(4):
            m = np.zeros(sh, bool); m[sw == s, :, v] = True; yield (f"S{s}", VARS[v]), m


# control: same-sized sample from the held-out block of train
Xc, dc, sc = [], [], []
for s in range(S):
    x, _, dd = make_windows(A[s], doy[s], cut, T); i = rng.choice(len(x), 89, replace=False)
    Xc.append(x[i]); dc.append(dd[i]); sc.append(np.full(89, s))
Xc, dc, sc = map(np.concatenate, (Xc, dc, sc))
for name, (Xw, dw, sw) in {"TEST regime": (X, d, st), "control (train block)": (Xc, dc, sc)}.items():
    P, Y = bt(Xw, dw, sw)
    print(f"\n{name}: backtest MAE {np.abs(P - Y).mean():.4f}, zero-forecast {np.abs(Y).mean():.4f}")
    for gname, fn in [("per variable", by_var), ("per var x step", by_var_step), ("per station", by_station), ("per station x var", by_station_var)]:
        b, f, G = gain_cv(P, Y, sw, fn)
        print(f"  {gname:18s} held-out MAE g=1 {b:.4f} -> fitted {f:.4f} ({f - b:+.4f})")
    if name.startswith("TEST"):
        print("  gains per variable:", {k: round(float(v), 2) for k, v in gain_cv(P, Y, sw, by_var)[2].items()})
        print("  gains per station :", {k: round(float(v), 2) for k, v in gain_cv(P, Y, sw, by_station)[2].items()})
