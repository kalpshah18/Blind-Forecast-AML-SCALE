"""End-to-end reproduction of the final submission from the provided competition files only.

    python reproduce.py [--out submission.csv] [--compare other_submission.csv]
                        [--refit-weights] [--no-ksel]

Pipeline (see README.md for details):
  1. climatology per station (4-harmonic regression) and scaled anomalies from train.csv
  2. test-time adaptation from the unlabelled test inputs: per-cell seasonal offset, dispersion
     ratio correction, pooled station-normalised model for S0 wind
  3. fit the three models on all of train (per-station LAD regression, LightGBM, 6-seed MLP with
     shift/scale augmentation) and predict the adapted test windows
  4. blend with the stored per-(step, variable) weights in blend_weights.json and write the csv

blend_weights.json was produced by this repository's own code (`blend.py export`) from five-fold,
time-blocked out-of-fold predictions of the same three models on train.csv. Regenerating it takes
much longer than the 30-minute budget (it trains every model five more times), so it is stored.
`--refit-weights` regenerates it from scratch first, for verification; this is NOT the fast path.

The neural net uses a CUDA GPU when one is available (Kaggle/Colab) and the CPU otherwise.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

MODELS = ["lad_seas", "gbm", "nn6a"]
WEIGHTS = "blend_weights.json"
DATA = ["train.csv", "test.csv", "scales.csv", "sample_submission.csv"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--compare", help="existing submission csv to compare against (mean |difference|)")
    ap.add_argument("--refit-weights", action="store_true",
                    help="regenerate blend_weights.json via the full five-fold out-of-fold procedure (slow)")
    ap.add_argument("--no-ksel", action="store_true",
                    help="use a fixed first-harmonic offset for every cell instead of the per-cell selection")
    args = ap.parse_args()

    missing = [f for f in DATA if not os.path.exists(f)]
    if missing:
        sys.exit(f"missing competition files in the working directory: {missing}")

    import adapt, blend, nn
    import torch
    adapt.KSEL = not args.no_ksel
    print(f"device for the neural net: {nn.DEVICE}  (torch {torch.__version__})", flush=True)
    t0 = time.time()

    if args.refit_weights:
        import oof
        for m in MODELS:
            print(f"[refit-weights] out-of-fold run for {m}", flush=True)
            oof.run(m)
        blend.export(MODELS, WEIGHTS)
        print(f"[refit-weights] done after {(time.time() - t0) / 60:.1f} min", flush=True)

    t1 = time.time()
    adapt.main(args.out, MODELS, weights=WEIGHTS)
    print(f"\nfinal fit + prediction + blend: {(time.time() - t1) / 60:.1f} min; "
          f"total: {(time.time() - t0) / 60:.1f} min", flush=True)

    sub = pd.read_csv(args.out)
    ref = pd.read_csv("sample_submission.csv")
    assert len(sub) == 17800 and list(sub.id) == list(ref.id) and sub.value.notna().all()
    print(f"wrote {args.out}: {len(sub)} rows, ids match sample_submission.csv, no NaNs")
    if args.compare:
        other = pd.read_csv(args.compare).set_index("id").value
        d = (sub.set_index("id").value - other).abs()
        print(f"vs {args.compare}: mean |diff| = {d.mean():.5f}, max = {d.max():.4f} (scaled units)")


if __name__ == "__main__":
    main()
