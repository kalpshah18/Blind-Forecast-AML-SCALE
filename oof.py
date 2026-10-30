"""Cache out-of-fold (CV) and test predictions of each model so blending needs no refits.
Run: python oof.py [model ...]   (models: ridge_seas lad_seas gbm)
Writes oof/{model}_oof.npy (N,5,4), oof/truth.npy, oof/{model}_test.npy (890,5,4).
OOF rows are station-major within each fold, folds concatenated in order.
"""
import os
import sys
import numpy as np
import linear
import gbm, nn, analog
from common import *

os.makedirs("oof", exist_ok=True)


POOLED = {"gbm": (gbm.fit_predict, gbm.STRIDE), "nn": (nn.fit_predict, nn.STRIDE),
          "analog": (analog.fit_predict, analog.STRIDE),
          "nn6": (nn.fit_predict6, nn.STRIDE),
          "nn6a": (nn.fit_predict6a, nn.STRIDE)}


def run(model, n_folds=5, k_clim=4):
    raw, doy = load_train()
    S, T, _ = raw.shape
    oof, truth = [], []
    for lo, hi in cv_folds(T, n_folds):
        mask = np.ones((S, T), bool); mask[:, lo:hi] = False
        A = to_anom(raw, doy, fit_climatology(raw, doy, mask, k_clim))
        if model in POOLED:
            fp, stride = POOLED[model]
            tr = gbm.build(A, doy, [(0, lo), (hi, T)], range(S), stride)
            parts = [make_windows(A[s], doy[s], lo, hi) for s in range(S)]
            Xv, Yv, dv = [np.concatenate(z) for z in zip(*parts)]
            sv = np.concatenate([np.full(len(p[0]), s) for s, p in enumerate(parts)])
            oof.append(fp(tr, (Xv, dv, sv)))
        else:
            ps, ys = [], []
            spans = [(a, b) for a, b in [(0, lo), (hi, T)] if b - a > IN + OUT]
            for s in range(S):
                trn = linear.station_windows(A[s], doy[s], spans)
                Xv, Yv, dv = make_windows(A[s], doy[s], lo, hi)
                ps.append(linear.fit_predict(model, trn, (Xv, dv)))
            oof.append(np.concatenate(ps))
        if not os.path.exists("oof/truth.npy"):
            ys = [make_windows(A[s], doy[s], lo, hi)[1] for s in range(S)]
            truth.append(np.concatenate(ys))
        print(model, "fold", lo, hi, flush=True)
    np.save(f"oof/{model}_oof.npy", np.concatenate(oof))
    if truth:
        np.save("oof/truth.npy", np.concatenate(truth))

    # test predictions from full-data fit
    coefs = fit_climatology(raw, doy, None, k_clim)
    A = to_anom(raw, doy, coefs)
    wids, st, X, d, tdoy = load_test(coefs)
    if model in POOLED:
        fp, stride = POOLED[model]
        pred = fp(gbm.build(A, doy, [(0, T)], range(S), stride), (X, d, st))
    else:
        pred = np.zeros((len(wids), OUT, 4))
        for s in range(S):
            m = st == s
            pred[m] = linear.fit_predict(model, linear.station_windows(A[s], doy[s], [(0, T)]), (X[m], d[m]))
    np.save(f"oof/{model}_test.npy", pred)
    print(model, "done", flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:] or ["lad_seas", "ridge_seas", "gbm"]:
        run(m)
