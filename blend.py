"""Blend cached model predictions. Weights are fit per (step, variable) on OOF predictions
(simplex grid + a scalar shrink toward climatology), evaluated with fold-wise cross-fitting
so the reported score is honest.
Run: python blend.py [model ...]            -> report
     python blend.py submit [model ...]     -> write submission_blend.csv
"""
import itertools
import sys
import numpy as np
from common import *

N_FOLDS = 5
SHRINKS = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15]


def simplex(k, step=0.1):
    n = int(round(1 / step))
    return [np.array(c) / n for c in itertools.product(range(n + 1), repeat=k) if sum(c) == n]


def fit_weights(P, Y, grid, shrinks):
    """P (M,N,5,4) preds, Y (N,5,4) -> weights (5,4,M), shrink (5,4)."""
    M = len(P)
    W = np.zeros((OUT, 4, M)); G = np.ones((OUT, 4))
    for j in range(OUT):
        for v in range(4):
            Pj = P[:, :, j, v]; y = Y[:, j, v]
            best = (9, None)
            for w in grid:
                e = np.abs(w @ Pj - y).mean()
                if e < best[0]:
                    best = (e, w)
            W[j, v] = best[1]
            b = best[1] @ Pj
            G[j, v] = min(shrinks, key=lambda g: np.abs(g * b - y).mean())
    return W, G


def save_weights(models, W, G, path):
    import json
    json.dump({"models": list(models), "W": W.tolist(), "G": G.tolist()}, open(path, "w"))


def load_weights(models, path):
    import json
    d = json.load(open(path))
    assert d["models"] == list(models), f"weights file is for {d['models']}, not {list(models)}"
    return np.array(d["W"]), np.array(d["G"])


def export(models, path):
    """Fit weights on the cached OOF predictions (oof/*.npy) and store them."""
    P = np.stack([np.load(f"oof/{m}_oof.npy") for m in models]); Y = np.load("oof/truth.npy")
    W, G = fit_weights(P, Y, simplex(len(models)), SHRINKS)
    save_weights(models, W, G, path)
    print("wrote", path)


def apply(P, W, G):
    return np.einsum("mnjv,jvm->njv", P, W) * G


def main(models, submit):
    P = np.stack([np.load(f"oof/{m}_oof.npy") for m in models])
    Y = np.load("oof/truth.npy")
    N = Y.shape[0]
    fold_id = np.zeros(N, int)  # folds were concatenated in order; recover boundaries from sizes
    sizes = []
    raw, doy = load_train()
    for lo, hi in cv_folds(raw.shape[1], N_FOLDS):
        sizes.append(10 * (hi - lo - (IN + OUT) + 1))
    assert sum(sizes) == N, (sum(sizes), N)
    fold_id = np.repeat(np.arange(N_FOLDS), sizes)
    grid = simplex(len(models)); shrinks = SHRINKS

    print("single models:")
    for m, p in zip(models, P):
        print(f"  {m:12s} {np.abs(p - Y).mean():.4f}")
    print(f"  equal-avg    {np.abs(P.mean(0) - Y).mean():.4f}")

    cross = np.zeros_like(Y)
    for f in range(N_FOLDS):
        tr, va = fold_id != f, fold_id == f
        W, G = fit_weights(P[:, tr], Y[tr], grid, shrinks)
        cross[va] = apply(P[:, va], W, G)
    print(f"cross-fitted blend (per step/var weights + shrink): {np.abs(cross - Y).mean():.4f}")
    print("  per-var:", " ".join(f"{x:.3f}" for x in np.abs(cross - Y).mean((0, 1))))

    W, G = fit_weights(P, Y, grid, shrinks)
    print("full-fit mean weights per model:", dict(zip(models, W.mean((0, 1)).round(2))))
    print("shrink (steps x vars):\n", G.round(2))
    if submit:
        T = np.stack([np.load(f"oof/{m}_test.npy") for m in models])
        coefs = fit_climatology(raw, doy, None, 4)
        wids, st, _, _, tdoy = load_test(coefs)
        write_submission(wids, st, apply(T, W, G), tdoy, coefs, "submission_blend.csv")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "export":          # python blend.py export blend_weights.json model [model ...]
        export(a[2:], a[1]); sys.exit()
    sub = bool(a) and a[0] == "submit"
    models = (a[1:] if sub else a) or ["lad_seas", "ridge_seas", "gbm"]
    main(models, sub)
